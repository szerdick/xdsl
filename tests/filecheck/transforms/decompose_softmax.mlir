// RUN: xdsl-opt -p decompose-softmax %s | filecheck %s

builtin.module {
  // 1D softmax with acc_bound: bound is propagated identity-style to math.exp.
  func.func @softmax_1d_with_bound(%x: tensor<16xf32>, %y: tensor<16xf32>) -> tensor<16xf32> {
    %r = linalg.softmax dimension(0) ins(%x : tensor<16xf32>) outs(%y : tensor<16xf32>) {acc_bound = 1.000000e-06 : f32} -> tensor<16xf32>
    func.return %r : tensor<16xf32>
  }

  // 2D softmax along dim 1, no acc_bound: math.exp must NOT carry the attribute.
  func.func @softmax_2d_no_bound(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %r = linalg.softmax dimension(1) ins(%x : tensor<4x8xf32>) outs(%y : tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %r : tensor<4x8xf32>
  }
}

// CHECK: builtin.module {

// ===== 1D, acc_bound = 1e-6 =====
// CHECK:      func.func @softmax_1d_with_bound(%[[X:.*]]: tensor<16xf32>, %[[Y:.*]]: tensor<16xf32>) -> tensor<16xf32> {
//             Step 1: max reduction (init: -inf, then linalg.reduce with arith.maximumf).
// CHECK-NEXT:   %[[NEG_INF:.*]] = arith.constant 0xff800000 : f32
// CHECK-NEXT:   %[[MAX_INIT:.*]] = tensor.empty() : tensor<f32>
// CHECK-NEXT:   %[[MAX_FILLED:.*]] = linalg.fill ins(%[[NEG_INF]] : f32) outs(%[[MAX_INIT]] : tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   %[[MAX:.*]] = linalg.reduce ins(%[[X]]:tensor<16xf32>) outs(%[[MAX_FILLED]]:tensor<f32>) dimensions = [0]
// CHECK-NEXT:   (%{{.*}}: f32, %{{.*}}: f32) {
// CHECK-NEXT:     %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   }
//             Step 2: numerator = exp(input - max). math.exp carries acc_bound.
// CHECK-NEXT:   %[[NUMER:.*]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%[[X]], %[[MAX]] : tensor<16xf32>, tensor<f32>) outs(%[[Y]] : tensor<16xf32>) {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     %{{.*}} = math.exp %{{.*}} {acc_bound = 1.000000e-06 : f32} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   } -> tensor<16xf32>
//             Step 3: sum reduction.
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[SUM_INIT:.*]] = tensor.empty() : tensor<f32>
// CHECK-NEXT:   %[[SUM_FILLED:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[SUM_INIT]] : tensor<f32>) -> tensor<f32>
// CHECK-NEXT:   %[[DENOM:.*]] = linalg.reduce ins(%[[NUMER]]:tensor<16xf32>) outs(%[[SUM_FILLED]]:tensor<f32>) dimensions = [0]
// CHECK-NEXT:   (%{{.*}}: f32, %{{.*}}: f32) {
// CHECK-NEXT:     %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   }
//             Step 4: divide.
// CHECK-NEXT:   %[[R:.*]] = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%[[NUMER]], %[[DENOM]] : tensor<16xf32>, tensor<f32>) outs(%[[Y]] : tensor<16xf32>) {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     %{{.*}} = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:     linalg.yield %{{.*}} : f32
// CHECK-NEXT:   } -> tensor<16xf32>
// CHECK-NEXT:   func.return %[[R]] : tensor<16xf32>

// ===== 2D dim=1, no acc_bound =====
// CHECK:      func.func @softmax_2d_no_bound(%[[X2:.*]]: tensor<4x8xf32>, %[[Y2:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
//             Reduced shape is tensor<4xf32> (drop dim 1). Iterators are 2 parallel dims.
// CHECK:        %{{.*}} = tensor.empty() : tensor<4xf32>
// CHECK:        linalg.reduce ins(%[[X2]]:tensor<4x8xf32>) outs(%{{.*}}:tensor<4xf32>) dimensions = [1]
//             math.exp must NOT carry an acc_bound attribute when softmax has none.
// CHECK:        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
// CHECK:          arith.subf
// CHECK-NEXT:     math.exp %{{[^ ]+}} : f32
// CHECK-NEXT:     linalg.yield
