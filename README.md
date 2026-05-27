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

Phase 1 tile prototype exists:

```zig
const tile = zt.load(f32, .{BLOCK}, x + offset, .{ .valid_len = valid_len });
const out = tile.addScalar(value);
zt.store(z + offset, out, .{ .valid_len = valid_len });
```

Implemented device-side pieces:

- `GlobalPtr(T)` / `ConstGlobalPtr(T)`
- `Tile(T, shape, space)`
- `RegTile(T, shape)`
- `load`
- `store`
- tile + scalar
- tile + tile
- tail-safe `valid_len` handling

Validated kernels:

- `vector_add`
- `fill`
- `add_scalar`
- `add_const_tile`
- `add_tile`

## Layout

```text
src/root.zig       Host API entrypoint
src/context.zig    CUDA context wrapper
src/buffer.zig     DeviceBuffer(T)
src/module.zig     PTX module loading
src/kernel.zig     Kernel wrapper
src/launch.zig     CUDA launch config
src/args.zig       Kernel argument packing
src/device.zig     Device/kernel tile API
kernels/gpu.zig    Prototype kernels
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

Current next step: Phase 2 shared memory and cooperative operations.

Planned next pieces:

- `SharedTile(T, shape)`
- `load_shared`
- `load_reg`
- block barrier wrapper
- `reduce(tile, dim, .sum)`
- block-level sum reduction kernel
