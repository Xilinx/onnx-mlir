git clone -n https://github.com/xilinx/llvm-aie.git llvm-project
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 70ac20b17d30b489a8265a10d1c65a5bf8788c15 && cd ..
