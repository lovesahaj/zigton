const std = @import("std");
const zt = @import("zigton");

const reduce_ptx: [:0]const u8 = @embedFile("reduce_ptx");

test "block_sum kernel" {
    const gpa = std.testing.allocator;
    const shared_n: u32 = zt.THREADS * 4 + 7;

    const grid_x: c_uint = try zt.utils.cdiv(shared_n, zt.TILE);

    const x = try gpa.alloc(f32, shared_n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, grid_x);
    defer gpa.free(z);

    const sum = try gpa.alloc(f32, 1);
    defer gpa.free(sum);

    for (0..shared_n) |i| {
        x[i] = @floatFromInt(i);
    }

    for (0..grid_x) |i| {
        z[i] = 0.0;
    }

    sum[0] = 0.0;

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_sum: zt.Kernel = try module.kernel("block_sum");

    var dx = try zt.DeviceBuffer(f32).init(shared_n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(grid_x);
    defer dz.deinit();

    var dsum = try zt.DeviceBuffer(f32).init(1);
    defer dsum.deinit();

    try dx.copyFromHost(x);

    const args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        shared_n,
    });

    try block_sum.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    const vec: @Vector(shared_n, f32) = x[0..shared_n].*;

    const sum_args = zt.kernelArgs(.{
        dz.ptr,
        dsum.ptr,
        grid_x,
    });

    try block_sum.launch(
        .{
            .grid = .{ .x = 1 },
            .block = .{ .x = zt.THREADS },
        },
        sum_args,
    );

    try ctx.sync();
    try dsum.copyToHost(sum);

    try std.testing.expectEqual(@reduce(.Add, vec), sum[0]);
}

test "block_max kernel" {
    const gpa = std.testing.allocator;
    const shared_n: u32 = zt.TILE * 2 + 7;

    const grid_x: c_uint = try zt.utils.cdiv(shared_n, zt.TILE);

    const x = try gpa.alloc(f32, shared_n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, grid_x);
    defer gpa.free(z);

    for (0..shared_n) |i| {
        x[i] = -@as(f32, @floatFromInt(i + 1));
    }

    for (0..grid_x) |i| {
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_max: zt.Kernel = try module.kernel("block_max");

    var dx = try zt.DeviceBuffer(f32).init(shared_n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(grid_x);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        shared_n,
    });

    try block_max.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..grid_x) |block_i| {
        const first_idx = block_i * @as(usize, @intCast(zt.TILE));
        const expected = -@as(f32, @floatFromInt(first_idx + 1));
        try std.testing.expectEqual(expected, z[block_i]);
    }
}

test "block_min kernel" {
    const gpa = std.testing.allocator;
    const shared_n: u32 = zt.TILE * 2 + 7;

    const grid_x: c_uint = try zt.utils.cdiv(shared_n, zt.TILE);

    const x = try gpa.alloc(f32, shared_n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, grid_x);
    defer gpa.free(z);

    for (0..shared_n) |i| {
        x[i] = @as(f32, @floatFromInt(i + 1));
    }

    for (0..grid_x) |i| {
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_min: zt.Kernel = try module.kernel("block_min");

    var dx = try zt.DeviceBuffer(f32).init(shared_n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(grid_x);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        shared_n,
    });

    try block_min.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..grid_x) |block_i| {
        const first_idx = block_i * @as(usize, @intCast(zt.TILE));
        const expected = @as(f32, @floatFromInt(first_idx + 1));
        try std.testing.expectEqual(expected, z[block_i]);
    }
}

test "block_mul kernel" {
    const gpa = std.testing.allocator;
    const shared_n: u32 = zt.TILE * 2 + 7;

    const grid_x: c_uint = try zt.utils.cdiv(shared_n, zt.TILE);

    const x = try gpa.alloc(f32, shared_n);
    defer gpa.free(x);
    const z = try gpa.alloc(f32, grid_x);
    defer gpa.free(z);

    for (0..shared_n) |i| {
        const step: f32 = @floatFromInt(i % 3);
        x[i] = 1.0 + step * 0.01;
    }

    for (0..grid_x) |i| {
        z[i] = 0.0;
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_mul: zt.Kernel = try module.kernel("block_mul");

    var dx = try zt.DeviceBuffer(f32).init(shared_n);
    defer dx.deinit();

    var dz = try zt.DeviceBuffer(f32).init(grid_x);
    defer dz.deinit();

    try dx.copyFromHost(x);

    const args = zt.kernelArgs(.{
        dx.ptr,
        dz.ptr,
        shared_n,
    });

    try block_mul.launch(
        .{
            .grid = .{ .x = grid_x },
            .block = .{ .x = zt.THREADS },
        },
        args,
    );

    try ctx.sync();
    try dz.copyToHost(z);

    for (0..grid_x) |block_i| {
        const start = block_i * @as(usize, @intCast(zt.TILE));
        const end = @min(start + @as(usize, @intCast(zt.TILE)), @as(usize, @intCast(shared_n)));
        var expected: f32 = 1.0;
        for (x[start..end]) |v| expected *= v;
        try std.testing.expectApproxEqRel(expected, z[block_i], 1e-4);
    }
}

test "sumF32 host helper" {
    const gpa = std.testing.allocator;
    const N: u32 = zt.THREADS * 4 + 7;

    const a = try gpa.alloc(f32, N);
    defer gpa.free(a);

    for (0..N) |i| {
        a[i] = @floatFromInt(i);
    }

    var expected: f32 = 0.0;
    for (a) |v| expected += v;

    try expectReduceF32(.Add, a, expected);
}

test "sumF32 edge sizes" {
    const gpa = std.testing.allocator;

    const sizes = [_]u32{
        0,
        1,
        zt.THREADS - 1,
        zt.THREADS,
        zt.TILE - 1,
        zt.TILE,
        zt.TILE + 1,
        zt.TILE * zt.TILE + 17,
    };

    for (sizes) |size| {
        const alloc_len: u32 = @max(size, 1);
        const input = try gpa.alloc(f32, alloc_len);
        defer gpa.free(input);

        var expected: f32 = 0.0;
        for (0..size) |i| {
            input[i] = @floatFromInt(i);
            expected += input[i];
        }
        if (size == 0) input[0] = 123.0;

        try expectReduceF32(.Add, input[0..@as(usize, @intCast(size))], expected);
    }
}

test "maxF32 host helper" {
    const gpa = std.testing.allocator;
    const N: u32 = zt.THREADS * 4 + 7;

    const a = try gpa.alloc(f32, N);
    defer gpa.free(a);

    for (0..N) |i| {
        a[i] = -@as(f32, @floatFromInt(i + 1));
    }

    var expected: f32 = -std.math.inf(f32);
    for (a) |v| expected = @max(expected, v);

    try expectReduceF32(.Max, a, expected);
}

test "maxF32 edge sizes" {
    const gpa = std.testing.allocator;

    const sizes = [_]u32{
        0,
        1,
        zt.THREADS - 1,
        zt.THREADS,
        zt.TILE - 1,
        zt.TILE,
        zt.TILE + 1,
        zt.TILE * zt.TILE + 17,
    };

    for (sizes) |size| {
        const alloc_len: u32 = @max(size, 1);
        const input = try gpa.alloc(f32, alloc_len);
        defer gpa.free(input);

        var expected: f32 = -std.math.inf(f32);
        for (0..size) |i| {
            input[i] = -@as(f32, @floatFromInt(i + 1));
            expected = @max(expected, input[i]);
        }
        if (size == 0) input[0] = 123.0;

        try expectReduceF32(.Max, input[0..@as(usize, @intCast(size))], expected);
    }
}

test "reduceF32 host helper" {
    const gpa = std.testing.allocator;
    const N: u32 = zt.THREADS * 4 + 7;

    const a = try gpa.alloc(f32, N);
    defer gpa.free(a);

    for (0..N) |i| {
        const step: f32 = @floatFromInt(i % 5);
        a[i] = 1.0 + step * 0.01;
    }

    const ops = [4]std.builtin.ReduceOp{ .Add, .Max, .Min, .Mul };

    inline for (ops) |op| {
        var expected: f32 = identity(op);

        for (a) |v| {
            expected = applyReduceOp(op, v, expected);
        }

        try expectReduceF32(op, a, expected);
    }
}

fn expectReduceF32(
    comptime op: std.builtin.ReduceOp,
    input: []const f32,
    expected: f32,
) !void {
    const alloc_len = @max(input.len, 1);

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const reducer: zt.Reducer = try zt.Reducer.init(module);

    var da = try zt.DeviceBuffer(f32).init(alloc_len);
    defer da.deinit();

    var db = try zt.DeviceBuffer(f32).init(alloc_len);
    defer db.deinit();

    if (input.len > 0) try da.copyFromHost(input);

    const actual = switch (op) {
        .Add => try reducer.sumF32(&ctx, da, db, @intCast(input.len)),
        .Max => try reducer.maxF32(&ctx, da, db, @intCast(input.len)),
        .Min => try reducer.minF32(&ctx, da, db, @intCast(input.len)),
        .Mul => try reducer.prodF32(&ctx, da, db, @intCast(input.len)),
        else => @compileError("unsupported reduction op"),
    };

    if (expected == std.math.inf(f32) or expected == -std.math.inf(f32)) {
        try std.testing.expectEqual(expected, actual);
    } else if (expected == 0.0) {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    } else {
        try std.testing.expectApproxEqRel(expected, actual, 1e-4);
    }
}

inline fn identity(comptime op: std.builtin.ReduceOp) f32 {
    return switch (op) {
        .Add => 0.0,
        .Max => -std.math.inf(f32),
        .Min => std.math.inf(f32),
        .Mul => 1.0,
        else => @compileError("unsupported reduction op"),
    };
}

inline fn applyReduceOp(
    comptime op: std.builtin.ReduceOp,
    a: f32,
    b: f32,
) f32 {
    return switch (op) {
        .Add => a + b,
        .Max => @max(a, b),
        .Min => @min(a, b),
        .Mul => a * b,
        else => @compileError("unsupported block operation"),
    };
}
