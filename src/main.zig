const std = @import("std");
const cuda = @import("cuda");
const zt = @import("zigton");

const ptx: [:0]const u8 = @embedFile("gpu_ptx");
const n: u32 = 100000000;
const block_x: c_uint = zt.TILE;

test "vector_add kernel" {
    const gpa = std.testing.allocator;

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const y = try gpa.alloc(f32, n);
    defer gpa.free(y);
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);

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

    var dx = try zt.DeviceBuffer(f32).init(n);
    defer dx.deinit();

    var dy = try zt.DeviceBuffer(f32).init(n);
    defer dy.deinit();

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    try dx.copyFromHost(x);
    try dy.copyFromHost(y);

    const args = zt.kernelArgs(.{ dx.ptr, dy.ptr, dz.ptr, n });
    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try vector_add.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = block_x },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..n) |i| {
        const expected = x[i] + y[i];
        try std.testing.expectEqual(expected, z[i]);
    }
}

test "fill kernel" {
    const gpa = std.testing.allocator;

    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const fill: zt.Kernel = try module.kernel("fill");

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    const fill_value: f32 = 1.0;
    const args_fill = zt.kernelArgs(.{
        dz.ptr,
        fill_value,
        n,
    });

    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try fill.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = block_x },
        },
        args_fill,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..n) |i| {
        try std.testing.expectEqual(fill_value, z[i]);
    }
}

test "add_scalar kernel" {
    const gpa = std.testing.allocator;

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const add_scalar: zt.Kernel = try module.kernel("add_scalar");

    var dx = try zt.DeviceBuffer(f32).init(n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const scalar: f32 = 3.0;
    const scalar_add_args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        scalar,
        n,
    });

    const grid_x: c_uint = try zt.utils.cdiv(n, block_x);

    try add_scalar.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = block_x },
        },
        scalar_add_args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..n) |i| {
        const expected = x[i] + scalar;
        try std.testing.expectEqual(expected, z[i]);
    }
}

test "add_scalar tile" {
    const gpa = std.testing.allocator;

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);

    for (0..n) |i| {
        x[i] = @floatFromInt(i);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const add_scalar: zt.Kernel = try module.kernel("add_const_tile");

    var dx = try zt.DeviceBuffer(f32).init(n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const scalar: f32 = 3.0;
    const scalar_add_args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        scalar,
        n,
    });

    const grid: c_uint = try zt.utils.cdiv(n, zt.TILE);
    const block: c_uint = zt.THREADS;

    try add_scalar.launch(
        .{
            .grid = .{ .x = grid },
            .block = .{ .x = block },
        },
        scalar_add_args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..n) |i| {
        const expected = x[i] + scalar;
        try std.testing.expectEqual(expected, z[i]);
    }
}

test "add_tile kernel" {
    const gpa = std.testing.allocator;

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const y = try gpa.alloc(f32, n);
    defer gpa.free(y);
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const vector_add: zt.Kernel = try module.kernel("add_tile");

    var dx = try zt.DeviceBuffer(f32).init(n);
    defer dx.deinit();

    var dy = try zt.DeviceBuffer(f32).init(n);
    defer dy.deinit();

    var dz = try zt.DeviceBuffer(f32).init(n);
    defer dz.deinit();

    try dx.copyFromHost(x);
    try dy.copyFromHost(y);

    const args = zt.kernelArgs(.{ dx.ptr, dy.ptr, dz.ptr, n });
    const grid_x: c_uint = try zt.utils.cdiv(n, zt.TILE);

    try vector_add.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..n) |i| {
        const expected = x[i] + y[i];
        try std.testing.expectEqual(expected, z[i]);
    }
}

test "shared_copy_raw kernel" {
    const gpa = std.testing.allocator;
    const shared_n: u32 = zt.THREADS * 4 + 7;

    const x = try gpa.alloc(f32, shared_n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, shared_n);
    defer gpa.free(z);

    for (0..shared_n) |i| {
        x[i] = @floatFromInt(i);
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(ptx);
    defer module.deinit();

    const shared_copy_raw: zt.Kernel = try module.kernel("shared_copy_raw");

    var dx = try zt.DeviceBuffer(f32).init(shared_n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(shared_n);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        shared_n,
    });

    const grid_x: c_uint = try zt.utils.cdiv(shared_n, zt.THREADS);

    try shared_copy_raw.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..shared_n) |i| {
        try std.testing.expectEqual(x[i], z[i]);
    }
}
