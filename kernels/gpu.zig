const builtin = @import("builtin");
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
    const BLOCK: u32 = zt.blockSizeHint;
    zt.requireBlock(BLOCK);
    const offset = BLOCK * zt.blockId(0);
    if (offset >= n) return;

    const remaining = n - offset;
    const valid_len = if (remaining < BLOCK) remaining else BLOCK;

    // we load this x into the register - this is the tile
    const tile = zt.load(f32, .{BLOCK}, x + offset, .{ .valid_len = valid_len });

    // we add the value to the tile - this give us a new tile
    const out = tile.addScalar(value);

    // now we have two tiles in reg memory one the og and the second
    // is the one we added the const to

    // we store the tile back to the z pointer which is retured
    zt.store(z + offset, out, .{ .valid_len = valid_len });
}

pub export fn add_tile(
    x: zt.ConstGlobalPtr(f32),
    y: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    const BLOCK: u32 = zt.blockSizeHint;
    zt.requireBlock(BLOCK);
    const offset = BLOCK * zt.blockId(0);
    if (offset >= n) return;

    const remaining = n - offset;
    const valid_len = if (remaining < BLOCK) remaining else BLOCK;

    // we load this x into the register - this is the tile
    const tile_x = zt.load(f32, .{BLOCK}, x + offset, .{ .valid_len = valid_len });
    const tile_y = zt.load(f32, .{BLOCK}, y + offset, .{ .valid_len = valid_len });

    // we add the value to the tile - this give us a new tile
    const out = tile_x.add(tile_y);

    // now we have two tiles in reg memory one the og and the second
    // is the one we added the const to

    // we store the tile back to the z pointer which is retured
    zt.store(z + offset, out, .{ .valid_len = valid_len });
}
