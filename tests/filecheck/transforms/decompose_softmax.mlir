// RUN: xdsl-opt -p decompose-softmax %s | filecheck %s

builtin.module {
  // 1D softmax with acc_bound: bound is propagated identity-style to math.exp.
  // Reductions are emitted as scf.for with scalar iter_args (no scratch tensor).
  func.func @softmax_1d_with_bound(%x: tensor<16xf32>, %y: tensor<16xf32>) -> tensor<16xf32> {
    %r = linalg.softmax dimension(0) ins(%x : tensor<16xf32>) outs(%y : tensor<16xf32>) {acc_bound = 1.000000e-06 : f32} -> tensor<16xf32>
    func.return %r : tensor<16xf32>
  }

  // 1D softmax without acc_bound: math.exp must NOT carry the attribute.
  func.func @softmax_1d_no_bound(%x: tensor<8xf32>, %y: tensor<8xf32>) -> tensor<8xf32> {
    %r = linalg.softmax dimension(0) ins(%x : tensor<8xf32>) outs(%y : tensor<8xf32>) -> tensor<8xf32>
    func.return %r : tensor<8xf32>
  }
}

// CHECK: builtin.module {

// ===== 1D, acc_bound = 1e-6 =====
// CHECK:      func.func @softmax_1d_with_bound(%[[X:.*]]: tensor<16xf32>, %[[Y:.*]]: tensor<16xf32>) -> tensor<16xf32> {
//             Step 1: max via scf.for with scalar iter_args, init = -inf.
// CHECK-NEXT:   %[[NEG_INF:.*]] = arith.constant 0xff800000 : f32
// CHECK-NEXT:   %[[C0_M:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[CN_M:.*]] = arith.constant 16 : index
// CHECK-NEXT:   %[[C1_M:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[MAX:.*]] = scf.for %{{.*}} = %[[C0_M]] to %[[CN_M]] step %[[C1_M]] iter_args(%{{.*}} = %[[NEG_INF]]) -> (f32) {
// CHECK-NEXT:     %{{.*}} = tensor.extract %[[X]][%{{.*}}] : tensor<16xf32>
// CHECK-NEXT:     %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     scf.yield %{{.*}} : f32
// CHECK-NEXT:   }
//             Step 2: numerator = exp(input - max). math.exp carries acc_bound.
//             max is a scalar f32 with broadcast map (d0) -> ().
// CHECK-NEXT:   %[[NUMER:.*]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%[[X]], %[[MAX]] : tensor<16xf32>, f32) outs(%[[Y]] : tensor<16xf32>) {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     %{{.*}} = math.exp %{{.*}} {acc_bound = 1.000000e-06 : f32} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   } -> tensor<16xf32>
//             Step 3: sum via scf.for with scalar iter_args, init = 0.
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[C0_S:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[CN_S:.*]] = arith.constant 16 : index
// CHECK-NEXT:   %[[C1_S:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[SUM:.*]] = scf.for %{{.*}} = %[[C0_S]] to %[[CN_S]] step %[[C1_S]] iter_args(%{{.*}} = %[[ZERO]]) -> (f32) {
// CHECK-NEXT:     %{{.*}} = tensor.extract %[[NUMER]][%{{.*}}] : tensor<16xf32>
// CHECK-NEXT:     %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     scf.yield %{{.*}} : f32
// CHECK-NEXT:   }
//             Step 4: divide.
// CHECK-NEXT:   %[[R:.*]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%[[NUMER]], %[[SUM]] : tensor<16xf32>, f32) outs(%[[Y]] : tensor<16xf32>) {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     %{{.*}} = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   } -> tensor<16xf32>
// CHECK-NEXT:   func.return %[[R]] : tensor<16xf32>

// ===== 1D, no acc_bound =====
// CHECK:      func.func @softmax_1d_no_bound(%[[XN:.*]]: tensor<8xf32>, %[[YN:.*]]: tensor<8xf32>) -> tensor<8xf32> {
// CHECK:        scf.for
//             math.exp must NOT carry an acc_bound attribute when softmax has none.
// CHECK:        math.exp %{{[^ ]+}} : f32
// CHECK:        linalg.yield
