run_add_kernel:
  zig build-obj kernels/vector_add.zig \
    -target nvptx64-cuda-none \
    -mcpu sm_80 \
    -ofmt=ptx
