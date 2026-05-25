const std = @import("std");
const cuda = @import("cuda");
const zt = @import("zigton");

const ptx = @embedFile("gpu_ptx");
const n: u32 = 1024;
const block_x: c_uint = 256;

test "vector_add kernel" {
    var x: [n]f32 = undefined;
    var y: [n]f32 = undefined;
    var z: [n]f32 = undefined;

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        y[i] = @floatFromInt(i * 2);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const vector_add: zt.Kernel = try module.kernel("vector_add");

    var dx = try zt.DeviceBuffer(f32).alloc(n);
    defer dx.deinit();

    var dy = try zt.DeviceBuffer(f32).alloc(n);
    defer dy.deinit();

    var dz = try zt.DeviceBuffer(f32).alloc(n);
    defer dz.deinit();

    try dx.copyFromHost(x[0..]);
    try dy.copyFromHost(y[0..]);

    var args = zt.kernelArgs(.{ dx.ptr, dy.ptr, dz.ptr, n });
    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try vector_add.launch(.{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .args = args.ptr(),
    });

    try ctx.sync();
    try dz.copyToHost(z[0..]);

    for (0..n) |i| {
        const expected = x[i] + y[i];
        try std.testing.expectEqual(expected, z[i]);
    }
}

test "fill kernel" {
    var z: [n]f32 = undefined;

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const fill: zt.Kernel = try module.kernel("fill");

    var dz = try zt.DeviceBuffer(f32).alloc(n);
    defer dz.deinit();

    const fill_value: f32 = 1.0;
    var args_fill = zt.kernelArgs(.{
        dz.ptr,
        fill_value,
        n,
    });

    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try fill.launch(.{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .args = args_fill.ptr(),
    });

    try ctx.sync();
    try dz.copyToHost(z[0..]);

    for (0..n) |i| {
        try std.testing.expectEqual(fill_value, z[i]);
    }
}

test "add_scalar kernel" {
    var x: [n]f32 = undefined;
    var z: [n]f32 = undefined;

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const add_scalar: zt.Kernel = try module.kernel("add_scalar");

    var dx = try zt.DeviceBuffer(f32).alloc(n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).alloc(n);
    defer dz.deinit();

    try dx.copyFromHost(x[0..]);

    const scalar: f32 = 3.0;
    var scalar_add_args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        scalar,
        n,
    });

    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try add_scalar.launch(.{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .args = scalar_add_args.ptr(),
    });

    try ctx.sync();
    try dz.copyToHost(z[0..]);

    for (0..n) |i| {
        const expected = x[i] + scalar;
        try std.testing.expectEqual(expected, z[i]);
    }
}


test "add_scalar tile" {
    var x: [n]f32 = undefined;
    var z: [n]f32 = undefined;

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const add_scalar: zt.Kernel = try module.kernel("add_const_tile");

    var dx = try zt.DeviceBuffer(f32).alloc(n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).alloc(n);
    defer dz.deinit();

    try dx.copyFromHost(x[0..]);

    const scalar: f32 = 3.0;
    var scalar_add_args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        scalar,
        n,
    });

    const grid: c_uint = try zt.utils.cdiv(n, 256);
    const block: c_uint = 256;

    try add_scalar.launch(.{
        .grid = .{ .x = grid },
        .block = .{ .x = block },
        .args = scalar_add_args.ptr(),
    });

    try ctx.sync();
    try dz.copyToHost(z[0..]);

    for (0..n) |i| {
        const expected = x[i] + scalar;
        try std.testing.expectEqual(expected, z[i]);
    }
}
