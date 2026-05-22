extern "C" __global__
void vector_add(const float* x, const float* y, float* z, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    z[i] = x[i] + y[i];
  }
}
