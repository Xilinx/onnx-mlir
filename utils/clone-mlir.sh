git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 0531ca15b5cadbb7e78d022a8ab637162db8e519 && cd ..
