# zigton

Tile-based GPU kernel experiments in Zig.

`zigton` is exploring a Triton-like programming model in plain Zig: one GPU block
owns one tile, user code performs tile operations, and the device API hides the
thread/lane mapping.

Status: early prototype. NVIDIA/NVPTX only.

## Current Milestone

The project can now compile Zig kernels to PTX and load them from a Zig host via
the CUDA Driver API.

Current build path:

```text
kernels/gpu.zig -> LLVM IR -> fix alias -> llc -> PTX -> @embedFile -> host
```

Phase 2 shared-memory and reduction prototype exists. Register tiles are real
`@Vector` values, and each thread owns `EPT` (elements-per-thread) lanes via a
strided, coalesced mapping:

```zig
const tile = zt.load(f32, zt.EPT, x, n);
const out = tile.addScalar(value);
zt.store(z, out, n);
```

Implemented device-side pieces:

- `GlobalPtr(T)` / `ConstGlobalPtr(T)`
- `RegTile(T, ept)` backed by a real `@Vector(ept, T)`
- `load` / `store` with strided, coalesced indexing
- per-element tail masking against `n`
- multiple elements per thread (`EPT`), tile width decoupled from block size
- tile + scalar
- tile + tile
- `requireBlock` launch-geometry guard
- `SharedTile(T, n)` backed by static `addrspace(.shared)` storage
- `blockSync()` barrier wrapper lowering to `bar.sync 0`
- device-side `blockReduceSum`
- host-side `reduceSumF32`

Validated kernels:

- `vector_add`
- `fill`
- `add_scalar`
- `add_const_tile`
- `add_tile`
- `shared_copy`
- `block_sum`

## Layout

```text
src/root.zig           Public host API entrypoint
src/main.zig           Minimal executable entrypoint
src/host/root.zig      Host CUDA API aggregate
src/host/context.zig   CUDA context wrapper
src/host/buffer.zig    DeviceBuffer(T)
src/host/module.zig    PTX module loading
src/host/kernel.zig    Kernel wrapper
src/host/launch.zig    CUDA launch config
src/host/args.zig      Kernel argument packing
src/host/utils.zig     Host math/CUDA helpers
src/device/root.zig    Device/kernel API aggregate
src/device/regtile.zig Register tile API
src/device/sharedtile.zig Shared tile API
src/device/config.zig  Shared THREADS / EPT / TILE constants
kernels/gpu.zig        Prototype kernels
tests/gpu.zig          GPU integration tests
tools/fix_ptx_ir.sh LLVM IR alias rewrite for NVPTX
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

Run app/tests depending on current entrypoint setup:

```sh
zig build run \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

```sh
zig build test \
  -Dllc-path=/path/to/llc \
  -Dgpu-arch=sm_89 \
  -Dcuda-prefix=/path/to/cuda
```

## Notes

Zig currently emits NVPTX kernels through an alias pattern that `llc` rejects:

```llvm
@vector_add = alias void (...), ptr @gpu.vector_add
define private ptx_kernel void @gpu.vector_add(...) { ... }
```

`tools/fix_ptx_ir.sh` rewrites the IR so kernel definitions have public bare names
before `llc` lowers them to PTX.

## Roadmap

Current next step: Phase 3 host ergonomics and broader reduction APIs.

Planned next pieces:

- cleaner public reduction API that owns kernel lookup
- reusable scratch-buffer reduction path
- broader reduction operations beyond f32 sum
- stronger PTX audit tooling for shared memory/barrier checks

## Writeups

- [Building Zigton Phase 1 (Part B) - Real Vector Tiles, Coalescing and Defining Zig-like](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton_part1_b)
- [Building Zigton Phase 1 - First Kernel compilation and run from Zig](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton_part2)
- [Building Zigton Phase 0 - Zig Host, CUDA PTX and the First Runtime Wrapper](https://portfolio.lovesahaj1225.workers.dev/#writing/building_zigton)
