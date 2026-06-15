build_ptx:
	zig build ptx -Dllc-path=../.local/llvm22/bin/llc -Dgpu-arch=sm_89

run_kernel:
	zig build run -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0

test_kernel:
	zig build test -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0

test_shared:
	zig build test -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0 -- --test-filter shared_copy

test_reduce:
	zig build test -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0 -- --test-filter reduceSumF32

test_matmul:
	zig build test-matmul -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0

bench_matmul:
	zig build bench-matmul -Dllc-path=../.local/llvm22/bin/llc -Dcuda-prefix=${HOME}/.local/cuda-13.0

test_single_file_kernel:
	zig build single-file-example \
	-Dllc-path=../.local/llvm22/bin/llc \
	-Dcuda-prefix=${HOME}/.local/cuda-13.0 \
	-Dgpu-arch=sm_89
