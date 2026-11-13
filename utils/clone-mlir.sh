#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
# affiliates

# Check out a specific branch that is known to work with ONNX-MLIR

# Shallow fetch to avoid cloning the full history, shallow clone requires a relative recent git 2.49
git init llvm-project
cd llvm-project
git remote add origin https://github.com/xilinx/llvm-aie.git
git fetch --depth 1 origin 152bedac13ea8bc47bc1e8e5d1d0734c89fe175f
git checkout FETCH_HEAD
cd ..
