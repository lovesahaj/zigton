build_ptx:
	zig build ptx -Dllc-path=../.local/llvm22/bin/llc -Dgpu-arch=sm_89

run_kernel:
	zig build run -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0

test_kernel:
	zig build test -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0
