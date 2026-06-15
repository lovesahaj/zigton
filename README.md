# zigton

Tile-based GPU kernel experiments in Zig.

`zigton` explores a Triton-like programming model in plain Zig: one GPU block
owns one tile, kernel code performs tile operations, and the device API maps
those operations onto GPU threads.

Status: early prototype. NVIDIA/NVPTX only.

## Current State

The project can compile Zig kernels to PTX, embed that PTX into a Zig host
binary, load it through the CUDA Driver API, and launch kernels from Zig.

Current kernel build path:

```text
kernel .zig file -> LLVM IR -> fix alias -> llc -> PTX -> @embedFile -> host
```

Kernel files are now modular. Each kernel source is compiled into its own PTX
module, avoiding NVPTX object-linking complexity for now.

Phase 2 shared-memory and reduction primitives exist. Phase 3 is focused on
host ergonomics, modular kernel files, same-file host/device experiments, and
broader reduction APIs.

## Implemented Pieces

Device-side API:

- `GlobalPtr(T)` / `ConstGlobalPtr(T)`
- `RegTile(T, ept)` backed by `@Vector(ept, T)`
- `load` / `store` with strided, coalesced indexing
- per-element tail masking against `n`
- multiple elements per thread via `EPT`
- tile + scalar
- tile + tile
- `requireBlock` launch-geometry guard
- `SharedTile(T, n)` backed by static `addrspace(.shared)` storage
- `blockSync()` barrier wrapper lowering to `bar.sync 0`
- device-side `blockReduceSum`
- device-side `blockReduceMax`
- device-side `blockReduce(.Add/.Max/.Min/.Mul, ...)`
- shared target helpers: `is_device`, `kernel_callconv`

Host-side API:

- `Context`
- `DeviceBuffer(T)`
- `Module`
- `Kernel`
- `LaunchConfig`
- `kernelArgs`
- `Reducer` owning reduction kernel lookup
- `Reducer.sumF32` / `Reducer.maxF32` / `Reducer.minF32` / `Reducer.prodF32`

Validated kernels/examples:

- `vector_add`
- `fill`
- `add_scalar`
- `add_const_tile`
- `add_tile`
- `shared_copy`
- `block_sum`
- `block_max`
- `block_min`
- `block_mul`
- `examples/single_file.zig` same-file host/device launch

## Layout

```text
src/root.zig              Public host API entrypoint
src/target.zig            Shared host/device target helpers
src/main.zig              Minimal executable entrypoint
src/host/root.zig         Host CUDA API aggregate
src/host/context.zig      CUDA context wrapper
src/host/buffer.zig       DeviceBuffer(T)
src/host/module.zig       PTX module loading
src/host/kernel.zig       Kernel wrapper
src/host/launch.zig       CUDA launch config
src/host/args.zig         Kernel argument packing
src/host/reduce.zig       Host reduction orchestration
src/host/utils.zig        Host math/CUDA helpers
src/device/root.zig       Device/kernel API aggregate
src/device/regtile.zig    Register tile API
src/device/sharedtile.zig Shared tile API
src/device/reduce.zig     Device block reductions
src/device/config.zig     Shared THREADS / EPT / TILE constants
kernels/base.zig          Base prototype kernels
kernels/reduce.zig        Reduction kernels
bench/matmul.zig          Matmul benchmark executable
examples/single_file.zig  Same-file host/device experiment
tests/base.zig            Base kernel integration tests
tests/reduce.zig          Reduction integration tests
tools/fix_ptx_ir.sh       LLVM IR alias rewrite for NVPTX
```

Host code imports:

```zig
const zt = @import("zigton");
```

Device kernel code imports:

```zig
const zt = @import("zigton_device");
```

## Requirements

- Zig with NVPTX support
- CUDA driver/runtime installed
- NVIDIA GPU
- `llc` from LLVM new enough to parse Zig's emitted LLVM IR

## Build

Build only PTX:

```sh
zig build ptx \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89
```

Run the app:

```sh
zig build run \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

Run tests:

```sh
zig build test \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

Run the matmul benchmark:

```sh
zig build bench-matmul \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

Run the same-file host/device experiment:

```sh
zig build single-file-example \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

## Adding Kernel Files

Kernel sources are registered in `build.zig` with `addKernelFile`:

```zig
const reduce_kernel = addKernelFile(b, .{
    .name = "reduce",
    .source = b.path("kernels/reduce.zig"),
    .gpu_arch = gpu_arch,
    .llc_path = llc_path,
    .optimize = optimize,
});
```

The helper compiles that source to PTX and returns a `KernelFile`:

```zig
.{
    .name = "reduce",
    .import_name = "reduce_ptx",
    .ptx = ...,
}
```

Embed it into a host module with:

```zig
addPtxImport(gpu_tests_mod, reduce_kernel.import_name, reduce_kernel.ptx);
```

Then host code can load it normally:

```zig
const reduce_ptx: [:0]const u8 = @embedFile("reduce_ptx");

var module = try zt.Module.loadData(reduce_ptx);
defer module.deinit();

const block_sum = try module.kernel("block_sum");
```

Current model: one PTX module per kernel file. This keeps kernel files modular
without requiring NVPTX object linking.

## Same-File Host/Device Experiment

`examples/single_file.zig` is compiled twice:

```text
examples/single_file.zig -> nvptx64-cuda -> PTX
examples/single_file.zig -> native host test -> embeds PTX
```

This lets one Zig file contain both a kernel and its host-side launch test.

The pattern uses shared target helpers:

```zig
const builtin = @import("builtin");
const zt = if (builtin.target.cpu.arch == .nvptx64)
    @import("zigton_device")
else
    @import("zigton");

pub export fn single_file_fill(...) callconv(zt.kernel_callconv) void {
    if (comptime !zt.is_device) return;
    // device code
}
```

This is not compiler-level offload. It is build-level double compilation plus
explicit PTX embedding.

## Notes

Zig currently emits NVPTX kernels through an alias pattern that `llc` rejects:

```llvm
@vector_add = alias void (...), ptr @gpu.vector_add
define private ptx_kernel void @gpu.vector_add(...) { ... }
```

`tools/fix_ptx_ir.sh` rewrites the IR so kernel definitions have public bare
names before `llc` lowers them to PTX.

Shared-memory kernels must keep barriers converged. A block barrier must be
reached by every thread in a warp at the same program point; guarding work is
safe, guarding the barrier is not.

## Writeups

- [Building Zigton Phase 1 (Part B) - Real Vector Tiles, Coalescing and Defining Zig-like](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton_part1_b)
- [Building Zigton Phase 1 - First Kernel compilation and run from Zig](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton_part2)
- [Building Zigton Phase 0 - Zig Host, CUDA PTX and the First Runtime Wrapper](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton)
