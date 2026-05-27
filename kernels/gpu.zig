const builtin = @import("builtin");
const zt = @import("zigton.zig");

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
    const BLOCK: u32 = 256;
    const offset = BLOCK * zt.blockId(0);
    const active = offset + BLOCK <= n;

    // we load this x into the register - this is the tile
    const tile = zt.load(f32, .{BLOCK}, x + offset, .{ .mask = active });

    // we add the value to the tile - this give us a new tile
    const out = tile.addScalar(value);

    // now we have two tiles in reg memory one the og and the second
    // is the one we added the const to

    // we store the tile back to the z pointer which is retured
    zt.store(z + offset, out, .{ .mask = active });
}
