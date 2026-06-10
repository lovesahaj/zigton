const zt = @import("zigton_device");

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
    const out = tile.addScalar(value);

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
    const out = tile_x.add(tile_y);

    zt.store(z, out, n);
}

pub export fn shared_copy(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    zt.requireBlock(zt.THREADS);
    if (n == 0) return;

    const tid = zt.utils.threadId(0);
    const i = zt.utils.blockId(0) * zt.THREADS + tid;
    const safe_i = @min(i, n - 1);

    const shared_tile = zt.sharedTile(f32, zt.THREADS, .shared_tile);
    shared_tile.store(tid, x[safe_i]);

    zt.blockSync();

    if (i < n) z[i] = shared_tile.load(tid);
}
