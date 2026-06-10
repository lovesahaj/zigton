const std = @import("std");
const zt = @import("zigton");

const base_ptx: [:0]const u8 = @embedFile("base_ptx");
const reduce_ptx: [:0]const u8 = @embedFile("reduce_ptx");
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

    var module = try zt.Module.loadData(base_ptx);
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

    var module = try zt.Module.loadData(base_ptx);
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

    var module = try zt.Module.loadData(base_ptx);
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

    var module = try zt.Module.loadData(base_ptx);
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

    var module = try zt.Module.loadData(base_ptx);
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

test "shared_copy kernel" {
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

    var module = try zt.Module.loadData(base_ptx);
    defer module.deinit();

    const shared_copy: zt.Kernel = try module.kernel("shared_copy");

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

    try shared_copy.launch(
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

test "sum_reduction kernel" {
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

test "reduceSumF32 host helper" {
    const gpa = std.testing.allocator;
    const N: u32 = zt.THREADS * 4 + 7;

    const a = try gpa.alloc(f32, N);
    defer gpa.free(a);

    for (0..N) |i| {
        a[i] = @floatFromInt(i);
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_sum_reducer: zt.Reducer = try zt.Reducer.init(module);

    var da = try zt.DeviceBuffer(f32).init(N);
    defer da.deinit();

    var db = try zt.DeviceBuffer(f32).init(N);
    defer db.deinit();

    try da.copyFromHost(a);

    var expected: f32 = 0.0;
    for (a) |v| expected += v;

    try std.testing.expectApproxEqRel(
        expected,
        try block_sum_reducer.sumF32(&ctx, da, db, N),
        1e-4,
    );
}

test "reduceSumF32 edge sizes" {
    const gpa = std.testing.allocator;

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const block_sum_reducer: zt.Reducer = try zt.Reducer.init(module);
    // const block_sum: zt.Kernel = try module.kernel("block_sum");

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

        var dinput = try zt.DeviceBuffer(f32).init(alloc_len);
        defer dinput.deinit();

        var dscratch = try zt.DeviceBuffer(f32).init(alloc_len);
        defer dscratch.deinit();

        try dinput.copyFromHost(input);

        try std.testing.expectApproxEqRel(
            expected,
            try block_sum_reducer.sumF32(&ctx, dinput, dscratch, size),
            1e-4,
        );
    }
}


test "reduceMaxF32 host helper" {
    const gpa = std.testing.allocator;
    const N: u32 = zt.THREADS * 4 + 7;

    const a = try gpa.alloc(f32, N);
    defer gpa.free(a);

    for (0..N) |i| {
        a[i] = @floatFromInt(i);
    }

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const reducer: zt.Reducer = try zt.Reducer.init(module);

    var da = try zt.DeviceBuffer(f32).init(N);
    defer da.deinit();

    var db = try zt.DeviceBuffer(f32).init(N);
    defer db.deinit();

    try da.copyFromHost(a);

    var expected: f32 = -std.math.inf(f32);
    for (a) |v| expected = @max(expected, v);

    try std.testing.expectApproxEqAbs(
        expected,
        try reducer.maxF32(&ctx, da, db, N),
        1e-4,
    );
}


test "reduceMaxF32 edge sizes" {
    const gpa = std.testing.allocator;

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(reduce_ptx);
    defer module.deinit();

    const reducer: zt.Reducer = try zt.Reducer.init(module);
    // const block_sum: zt.Kernel = try module.kernel("block_sum");

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
            input[i] = @floatFromInt(i);
            expected = @max(expected, input[i]);
        }


        if (size == 0) input[0] = 123.0;

        var dinput = try zt.DeviceBuffer(f32).init(alloc_len);
        defer dinput.deinit();

        var dscratch = try zt.DeviceBuffer(f32).init(alloc_len);
        defer dscratch.deinit();

        try dinput.copyFromHost(input);

        try std.testing.expectApproxEqAbs(
            expected,
            try reducer.maxF32(&ctx, dinput, dscratch, size),
            1e-4,
        );
    }
}
