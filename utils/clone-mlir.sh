# Check out a specific branch that is known to work with ONNX-MLIR

# Shallow fetch to avoid cloning the full history, shallow clone requires a relative recent git 2.49
git init llvm-project
cd llvm-project
git remote add origin https://github.com/xilinx/llvm-aie.git
git fetch --depth 1 origin ab257aa66ec30eb3a2092d21f27117f63fe92ea1
git checkout FETCH_HEAD
cd ..
