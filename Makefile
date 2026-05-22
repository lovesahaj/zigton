run_add_kernel:
	zig build-obj kernels/vector_add.zig \
    -target nvptx64-cuda-none \
    -mcpu sm_80 \
    -ofmt=ptx

just_assembly:
	zig build-obj kernels/vector_add.zig \
		-target nvptx64-cuda-none \
		-mcpu sm_89 \
		-fentry=vector_add \
		-fno-emit-bin \
		-femit-asm=zig-out/vector_add.ptx
