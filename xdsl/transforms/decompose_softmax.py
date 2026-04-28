from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    FloatAttr,
    ModuleOp,
    TensorType,
    i64,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def propagate_acc_bound(softmax_bound: FloatAttr) -> FloatAttr:
    """Compute the per-`math.exp` accuracy bound from the per-output softmax accuracy bound.

    Currently the identity is used as placeholder. The right rule depends on a forward error analysis
    of  exp(x_i - m) / sum_j exp(x_j - m); a conservative placeholder for an
    N-element reduction would be  softmax_bound / (2N + 1).
    """
    return softmax_bound


def _identity_map(rank: int) -> AffineMap:
    return AffineMap.identity(rank)


def _drop_dim_map(rank: int, dim: int) -> AffineMap:
    """Identity map of `rank` dims, dropping result `dim`."""
    results = tuple(AffineExpr.dimension(d) for d in range(rank) if d != dim)
    return AffineMap(rank, 0, results)


def _scalar_const(rewriter: PatternRewriter, value: float, ty) -> SSAValue:
    return rewriter.insert(arith.ConstantOp(FloatAttr(value, ty))).result


def _empty_with_dim_dropped(
    rewriter: PatternRewriter, input_ty: TensorType, dim: int
) -> SSAValue:
    shape = list(input_ty.get_shape())
    del shape[dim]
    reduced_ty = TensorType(input_ty.get_element_type(), shape)
    return rewriter.insert(tensor.EmptyOp((), reduced_ty)).tensor


def _filled(rewriter: PatternRewriter, fill_value: SSAValue, out: SSAValue) -> SSAValue:
    fill = linalg.FillOp(inputs=(fill_value,), outputs=(out,), res=(out.type,))
    return rewriter.insert(fill).results[0]


def _build_reduce(
    rewriter: PatternRewriter,
    input: SSAValue,
    init: SSAValue,
    reduce_dim: int,
    binop_cls: type,
) -> SSAValue:
    """Emit a `linalg.reduce` with a single-arith-op body."""
    input_ty = input.type
    assert isinstance(input_ty, TensorType)
    elem_ty = input_ty.get_element_type()

    body = Region(Block(arg_types=[elem_ty, elem_ty]))
    a, b = body.block.args
    op = binop_cls(a, b)
    body.block.add_op(op)
    body.block.add_op(linalg.YieldOp(op.result))

    reduce = linalg.ReduceOp(
        input=input,
        init=init,
        dimensions=DenseArrayBase.from_list(i64, [reduce_dim]),
        region=body,
    )
    return rewriter.insert(reduce).result[0]


def _build_sub_and_exp(
    rewriter: PatternRewriter,
    input: SSAValue,
    max_val: SSAValue,
    output: SSAValue,
    reduce_dim: int,
    acc_bound: FloatAttr | None,
) -> SSAValue:
    """Emit  num_i = exp(input_i - max)  via linalg.generic.

    The inner `math.exp` gets `acc_bound` as a discardable attribute.
    """
    input_ty = input.type
    assert isinstance(input_ty, TensorType)
    rank = len(input_ty.get_shape())
    elem_ty = input_ty.get_element_type()

    body = Region(Block(arg_types=[elem_ty, elem_ty, elem_ty]))
    in_arg, max_arg, _ = body.block.args
    sub = arith.SubfOp(in_arg, max_arg)
    body.block.add_op(sub)
    exp = math.ExpOp(sub.result)
    if acc_bound is not None:
        exp.attributes["acc_bound"] = acc_bound
    body.block.add_op(exp)
    body.block.add_op(linalg.YieldOp(exp.result))

    indexing_maps = ArrayAttr(
        [
            AffineMapAttr(_identity_map(rank)),
            AffineMapAttr(_drop_dim_map(rank, reduce_dim)),
            AffineMapAttr(_identity_map(rank)),
        ]
    )
    iterator_types = ArrayAttr(
        [linalg.IteratorTypeAttr.parallel() for _ in range(rank)]
    )

    generic = linalg.GenericOp(
        inputs=(input, max_val),
        outputs=(output,),
        body=body,
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
        result_types=(output.type,),
    )
    return rewriter.insert(generic).results[0]


def _build_div(
    rewriter: PatternRewriter,
    numerator: SSAValue,
    denominator: SSAValue,
    output: SSAValue,
    reduce_dim: int,
) -> SSAValue:
    """Emit  out_i = numerator_i / denominator  via linalg.generic."""
    input_ty = numerator.type
    assert isinstance(input_ty, TensorType)
    rank = len(input_ty.get_shape())
    elem_ty = input_ty.get_element_type()

    body = Region(Block(arg_types=[elem_ty, elem_ty, elem_ty]))
    num_arg, denom_arg, _ = body.block.args
    div = arith.DivfOp(num_arg, denom_arg)
    body.block.add_op(div)
    body.block.add_op(linalg.YieldOp(div.result))

    indexing_maps = ArrayAttr(
        [
            AffineMapAttr(_identity_map(rank)),
            AffineMapAttr(_drop_dim_map(rank, reduce_dim)),
            AffineMapAttr(_identity_map(rank)),
        ]
    )
    iterator_types = ArrayAttr(
        [linalg.IteratorTypeAttr.parallel() for _ in range(rank)]
    )

    generic = linalg.GenericOp(
        inputs=(numerator, denominator),
        outputs=(output,),
        body=body,
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
        result_types=(output.type,),
    )
    return rewriter.insert(generic).results[0]


class DecomposeSoftmaxPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.SoftmaxOp, rewriter: PatternRewriter
    ) -> None:
        input_val = op.input
        output_val = op.output
        input_ty = input_val.type
        if not isinstance(input_ty, TensorType):
            return
        elem_ty = input_ty.get_element_type()
        reduce_dim = op.dimension.value.data

        softmax_bound = op.attributes.get("acc_bound")
        per_exp_bound = (
            propagate_acc_bound(softmax_bound)
            if isinstance(softmax_bound, FloatAttr)
            else None
        )

        # Step 1: max along dim.
        neg_inf = _scalar_const(rewriter, float("-inf"), elem_ty)
        max_init = _empty_with_dim_dropped(rewriter, input_ty, reduce_dim)
        max_filled = _filled(rewriter, neg_inf, max_init)
        max_val = _build_reduce(
            rewriter, input_val, max_filled, reduce_dim, arith.MaxnumfOp
        )

        # Step 2: numerator = exp(input - max).  math.exp carries acc_bound.
        numerator = _build_sub_and_exp(
            rewriter, input_val, max_val, output_val, reduce_dim, per_exp_bound
        )

        # Step 3: denominator = sum along dim.
        zero = _scalar_const(rewriter, 0.0, elem_ty)
        sum_init = _empty_with_dim_dropped(rewriter, input_ty, reduce_dim)
        sum_filled = _filled(rewriter, zero, sum_init)
        denominator = _build_reduce(
            rewriter, numerator, sum_filled, reduce_dim, arith.AddfOp
        )

        # Step 4: result = numerator / denominator.
        result = _build_div(rewriter, numerator, denominator, output_val, reduce_dim)

        rewriter.replace_matched_op([], new_results=[result])


@dataclass(frozen=True)
class DecomposeSoftmaxPass(ModulePass):
    """Decompose `linalg.softmax` to elementwise primitive form.

    Mirrors upstream MLIR's `SoftmaxOp::decomposeOperation`. The`acc_bound`
    attribute from softmax is propagatedto each emitted `math.exp`
    via `propagate_acc_bound`.
    """

    name = "decompose-softmax"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(DecomposeSoftmaxPattern()).rewrite_module(op)
