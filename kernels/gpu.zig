const zt = @import("zigton_device");
const std = @import("std");

pub export fn vector_add(
    x: zt.ConstGlobalPtr(f32),
    y: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

pub export fn fill(
    z: zt.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = value;
    }
}

pub export fn add_scalar(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + value;
    }
}

pub export fn add_const_tile(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(.kernel) void {
    zt.requireBlock(zt.THREADS);
    const tile = zt.load(f32, zt.EPT, x, n);

    // we add the value to the tile - this give us a new tile
    const out = tile.addScalar(value);

    // now we have two tiles in reg memory one the og and the second
    // is the one we added the const to

    // we store the tile back to the z pointer which is retured
    zt.store(z, out, n);
}

pub export fn add_tile(
    x: zt.ConstGlobalPtr(f32),
    y: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    zt.requireBlock(zt.THREADS);
    const tile_x = zt.load(f32, zt.EPT, x, n);
    const tile_y = zt.load(f32, zt.EPT, y, n);

    // we add the value to the tile - this give us a new tile
    const out = tile_x.add(tile_y);

    // now we have two tiles in reg memory one the og and the second
    // is the one we added the const to

    // we store the tile back to the z pointer which is retured
    zt.store(z, out, n);
}

var smem_storage: [zt.THREADS]f32 addrspace(.shared) = undefined;

pub export fn shared_copy_raw(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    const tid = zt.threadId(0);
    const i = zt.blockId(0) * zt.THREADS + tid;

    const safe_i = @min(i, n - 1);
    smem_storage[tid] = x[safe_i];

    zt.blockSync();

    if (i < n) z[i] = smem_storage[tid];
}
