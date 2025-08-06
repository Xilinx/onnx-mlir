git clone -n https://github.com/xilinx/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout c27444ab4976dd9ff131212f87463f9945ab28d7 && cd ..
