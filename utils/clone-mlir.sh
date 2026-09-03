# Check out a specific branch that is known to work with ONNX-MLIR

# Shallow fetch to avoid cloning the full history, shallow clone requires a relative recent git 2.49
git init llvm-project
cd llvm-project
git remote add origin https://github.com/xilinx/llvm-aie.git
git fetch --depth 1 origin c60ed356b69096edbb268a9d2a94dcc9113fee9a
git checkout FETCH_HEAD
cd ..
