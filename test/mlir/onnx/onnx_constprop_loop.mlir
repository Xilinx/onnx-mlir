// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// LoopUnroll: constant-trip-count loops with NoneType condition are physically
// unrolled so that standard constprop can fold the resulting ops.
//
// Match conditions (see LoopUnroll in ConstProp.cpp):
//   • Loop condition operand is NoneType (loop always runs exactly M times)
//   • Trip count M is a dense scalar constant in (0, 64]
//===----------------------------------------------------------------------===//

// -----

// Simplest case: accumulate an integer sum across 3 iterations.
// Expected: onnx.Loop disappears; result folds to onnx.Constant dense<3>.

func.func @test_loop_unroll_carried_only() -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_unroll_carried_only() -> tensor<i64>
// CHECK-NOT:   onnx.Loop
// CHECK:       [[RES:%.+]] = onnx.Constant dense<3> : tensor<i64>
// CHECK:       onnx.Return [[RES]] : tensor<i64>

// -----

// Loop with one scan output: collect the running sum at each iteration.
// Trip count = 3, body: next = carried + 1, scan_elem = next.
// Expected scan output: [1, 2, 3] folds to a single constant tensor.

func.func @test_loop_unroll_scan_output() -> tensor<3xi64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %carried_out, %scan = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next, %next : tensor<i1>, tensor<i64>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> (tensor<i64>, tensor<3xi64>)
  onnx.Return %scan : tensor<3xi64>
}
// CHECK-LABEL: @test_loop_unroll_scan_output() -> tensor<3xi64>
// CHECK-NOT:   onnx.Loop
// CHECK:       [[SCAN:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK:       onnx.Return [[SCAN]] : tensor<3xi64>

// -----

// Loop is NOT unrolled: condition is a live tensor<i1> value, not NoneType.
// The LoopUnroll pattern requires NoneType condition.

func.func @test_loop_no_unroll_with_condition(%cond: tensor<i1>) -> tensor<i64> {
  %trip = onnx.Constant dense<3> : tensor<i64>
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %cond, %init) ({
  ^bb0(%iter: tensor<i64>, %body_cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    onnx.Yield %body_cond, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, tensor<i1>, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_with_condition
// CHECK:       onnx.Loop

// -----

// Loop is NOT unrolled: trip count is a runtime value, not a compile-time
// constant.

func.func @test_loop_no_unroll_dynamic_trip(%trip: tensor<i64>) -> tensor<i64> {
  %none = "onnx.NoValue"() {value} : () -> none
  %init = onnx.Constant dense<0> : tensor<i64>
  %result = "onnx.Loop"(%trip, %none, %init) ({
  ^bb0(%iter: tensor<i64>, %cond: tensor<i1>, %carried: tensor<i64>):
    %one = onnx.Constant dense<1> : tensor<i64>
    %next = "onnx.Add"(%carried, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %true = onnx.Constant dense<true> : tensor<i1>
    onnx.Yield %true, %next : tensor<i1>, tensor<i64>
  }) : (tensor<i64>, none, tensor<i64>) -> tensor<i64>
  onnx.Return %result : tensor<i64>
}
// CHECK-LABEL: @test_loop_no_unroll_dynamic_trip
// CHECK:       onnx.Loop

//===----------------------------------------------------------------------===//
// ConstPropConcatFromSequence: when a ConcatFromSequence op's input is a
// compile-time SequenceEmpty → SequenceInsert … chain with all-constant
// elements (NoneType position), fold the whole thing into a single onnx.Constant.
//===----------------------------------------------------------------------===//

// -----

// Concatenate two constant vectors along axis 0.
// Sequence: empty → insert [1,1,1] → insert [2,2,2]
// ConcatFromSequence(axis=0) → [1,1,1,2,2,2]
// Declare the output type as the expected folded shape so types are consistent.

func.func @test_constprop_concatfromseq_concat() -> tensor<6xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e0   = onnx.Constant dense<1> : tensor<3xi32>
  %e1   = onnx.Constant dense<2> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %e0, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<6xi32>
  onnx.Return %result : tensor<6xi32>
}
// CHECK-LABEL: @test_constprop_concatfromseq_concat() -> tensor<6xi32>
// CHECK-NOT:   onnx.ConcatFromSequence
// CHECK-NOT:   onnx.SequenceInsert
// CHECK:       [[CST:%.+]] = onnx.Constant dense<[1, 1, 1, 2, 2, 2]> : tensor<6xi32>
// CHECK:       onnx.Return [[CST]] : tensor<6xi32>

// -----

// Stack two constant row vectors along a new axis, producing a 2×3 matrix.
// Sequence: empty → insert [1,2,3] → insert [4,5,6]
// ConcatFromSequence(axis=0, new_axis=1) → [[1,2,3],[4,5,6]]

func.func @test_constprop_concatfromseq_stack() -> tensor<2x3xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e0   = onnx.Constant dense<[1, 2, 3]> : tensor<3xi32>
  %e1   = onnx.Constant dense<[4, 5, 6]> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %e0, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64, new_axis = 1 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<2x3xi32>
  onnx.Return %result : tensor<2x3xi32>
}
// CHECK-LABEL: @test_constprop_concatfromseq_stack() -> tensor<2x3xi32>
// CHECK-NOT:   onnx.ConcatFromSequence
// CHECK-NOT:   onnx.SequenceInsert
// CHECK:       [[CST:%.+]] = onnx.Constant dense<{{.}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
// CHECK:       onnx.Return [[CST]] : tensor<2x3xi32>

// -----

// NOT folded: one sequence element is a runtime value (function argument).
// The ConstPropConcatFromSequence pattern requires all elements to be
// compile-time constants.

func.func @test_constprop_concatfromseq_no_fold(%arg: tensor<3xi32>) -> tensor<6xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %seq0 = "onnx.SequenceEmpty"() {dtype = 6 : si64} : () -> !onnx.Seq<tensor<*xi32>>
  %e1   = onnx.Constant dense<2> : tensor<3xi32>
  %seq1 = "onnx.SequenceInsert"(%seq0, %arg, %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %seq2 = "onnx.SequenceInsert"(%seq1, %e1,  %none) : (!onnx.Seq<tensor<*xi32>>, tensor<3xi32>, none) -> !onnx.Seq<tensor<*xi32>>
  %result = "onnx.ConcatFromSequence"(%seq2) {axis = 0 : si64} : (!onnx.Seq<tensor<*xi32>>) -> tensor<6xi32>
  onnx.Return %result : tensor<6xi32>
}
// CHECK-LABEL: @test_constprop_concatfromseq_no_fold
// CHECK:       onnx.ConcatFromSequence
