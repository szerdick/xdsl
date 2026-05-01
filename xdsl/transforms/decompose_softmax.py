from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, scf, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    ModuleOp,
    TensorType,
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


def propagate_acc_bound(softmax_bound: FloatAttr, N: int) -> FloatAttr:
    """Compute the per-`math.exp` accuracy bound from the per-output softmax accuracy bound.

    `N` is the length of the dimension softmax reduces over.
    Currently assuming the softmax_bound is l_infinity.
    """
    ty = softmax_bound.type
    eps = softmax_bound.value.data
    # return FloatAttr(eps / (N + 1 + N * eps), ty)
    return FloatAttr(eps, ty)


def _identity_map(rank: int) -> AffineMap:
    return AffineMap.identity(rank)


def _drop_dim_map(rank: int, dim: int) -> AffineMap:
    """Identity map of `rank` dims, dropping result `dim`."""
    results = tuple(AffineExpr.dimension(d) for d in range(rank) if d != dim)
    return AffineMap(rank, 0, results)


def _build_reduce(
    rewriter: PatternRewriter,
    input: SSAValue,
    init_value: float,
    binop_cls: type,
) -> SSAValue:
    """Reduce a 1-D tensor along its single axis using `scf.for` + iter_args.

    Returns the scalar reduction result.
    """
    input_ty = input.type
    assert isinstance(input_ty, TensorType)
    elem_ty = input_ty.get_element_type()
    shape = list(input_ty.get_shape())
    if len(shape) != 1:
        raise NotImplementedError(
            "decompose-softmax: scf.for-based reduction only supports 1-D "
            f"inputs; got shape {shape}"
        )
    n = shape[0]

    init = rewriter.insert(arith.ConstantOp(FloatAttr(init_value, elem_ty))).result
    c0 = rewriter.insert(arith.ConstantOp(IntegerAttr(0, IndexType()))).result
    cN = rewriter.insert(arith.ConstantOp(IntegerAttr(n, IndexType()))).result
    c1 = rewriter.insert(arith.ConstantOp(IntegerAttr(1, IndexType()))).result

    body_block = Block(arg_types=[IndexType(), elem_ty])
    iv, acc = body_block.args
    extract = tensor.ExtractOp(input, (iv,), elem_ty)
    body_block.add_op(extract)
    binop = binop_cls(extract.result, acc)
    body_block.add_op(binop)
    body_block.add_op(scf.YieldOp(binop.result))

    for_op = rewriter.insert(scf.ForOp(c0, cN, c1, (init,), Region(body_block)))
    return for_op.res[0]


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
        reduce_dim = op.dimension.value.data

        softmax_bound = op.attributes.get("acc_bound")
        N = input_ty.get_shape()[reduce_dim]
        per_exp_bound = (
            propagate_acc_bound(softmax_bound, N)
            if isinstance(softmax_bound, FloatAttr) and N > 0
            else None
        )

        # Step 1: max along dim — scalar accumulator via scf.for.
        max_val = _build_reduce(rewriter, input_val, float("-inf"), arith.MaximumfOp)

        # Step 2: numerator = exp(input - max).  math.exp carries acc_bound.
        numerator = _build_sub_and_exp(
            rewriter, input_val, max_val, output_val, reduce_dim, per_exp_bound
        )

        # Step 3: denominator = sum along dim — scalar accumulator via scf.for.
        denominator = _build_reduce(rewriter, numerator, 0.0, arith.AddfOp)

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
