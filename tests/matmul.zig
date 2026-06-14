const std = @import("std");
const zt = @import("zigton");

const matmul: [:0]const u8 = @embedFile("matmul_ptx");

const MatmulCase = struct {
    m: usize,
    k: usize,
    n: usize,
};

fn matmulCPU(
    A: []const f32,
    B: []const f32,
    C: []f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    for (0..M) |r| {
        for (0..N) |c| {
            var tmp: f32 = 0.0;
            for (0..K) |i| {
                tmp += A[r * K + i] * B[i * N + c];
            }
            C[r * N + c] = tmp;
        }
    }
}

test "matmul kernel" {
    const cases = [_]MatmulCase{
        .{ .m = 4, .k = 5, .n = 3 },
        .{ .m = 33, .k = 7, .n = 35 },
        // .{ .m = 4096, .k = 4096, .n = 4096}
    };

    for (cases) |case| {
        try performanceMatmul(case);
    }
}

test "matmul performance kernel" {
    const cases = [_]MatmulCase{
        .{ .m = 4, .k = 5, .n = 3 },
        .{ .m = 33, .k = 7, .n = 35 },
        // .{ .m = 4096, .k = 4096, .n = 4096}
    };

    for (cases) |case| {
        try expectMatmul(case);
    }
}

fn performanceMatmul(case: MatmulCase) !void {
    const gpa = std.testing.allocator;

    const A = try gpa.alloc(f32, case.m * case.k);
    defer gpa.free(A);
    const B = try gpa.alloc(f32, case.k * case.n);
    defer gpa.free(B);
    const C = try gpa.alloc(f32, case.m * case.n);
    defer gpa.free(C);

    for (A, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i % 11) + 1));
    }

    for (B, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i % 7) + 1));
    }

    @memset(C, 0.0);

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(matmul);
    defer module.deinit();

    const matmul_naive: zt.Kernel = try module.kernel("matmul_naive");

    var dA = try zt.DeviceBuffer(f32).init(A.len);
    defer dA.deinit();

    var dB = try zt.DeviceBuffer(f32).init(B.len);
    defer dB.deinit();

    var dC = try zt.DeviceBuffer(f32).init(C.len);
    defer dC.deinit();

    try dA.copyFromHost(A);
    try dB.copyFromHost(B);
    try dC.copyFromHost(C);

    const args = zt.kernelArgs(.{
        dA.ptr,
        dB.ptr,
        dC.ptr,
        @as(u32, @intCast(case.m)),
        @as(u32, @intCast(case.k)),
        @as(u32, @intCast(case.n)),
    });

    const grid_x: c_uint = try zt.utils.cdiv(case.m, 32);
    const grid_y: c_uint = try zt.utils.cdiv(case.n, 32);

    var timer = try std.time.Timer.start();

    try matmul_naive.launch(
        .{
            .grid = .{ .x = grid_x, .y = grid_y },
            .block = .{ .x = 32, .y = 32 },
        },
        args,
    );

    try ctx.sync();
    const elapsed_ns = timer.read();

    try dC.copyToHost(C);

    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    std.debug.print(
        "matmul_naive {d}x{d}x{d}: {d} ns ({d:.3} ms)\n",
        .{ case.m, case.k, case.n, elapsed_ns, elapsed_ms },
    );
}

fn expectMatmul(case: MatmulCase) !void {
    const gpa = std.testing.allocator;

    const A = try gpa.alloc(f32, case.m * case.k);
    defer gpa.free(A);
    const B = try gpa.alloc(f32, case.k * case.n);
    defer gpa.free(B);
    const C = try gpa.alloc(f32, case.m * case.n);
    defer gpa.free(C);

    for (A, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i % 11) + 1));
    }

    for (B, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i % 7) + 1));
    }

    @memset(C, 0.0);

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(matmul);
    defer module.deinit();

    const matmul_naive: zt.Kernel = try module.kernel("matmul_naive");

    var dA = try zt.DeviceBuffer(f32).init(A.len);
    defer dA.deinit();

    var dB = try zt.DeviceBuffer(f32).init(B.len);
    defer dB.deinit();

    var dC = try zt.DeviceBuffer(f32).init(C.len);
    defer dC.deinit();

    try dA.copyFromHost(A);
    try dB.copyFromHost(B);
    try dC.copyFromHost(C);

    const args = zt.kernelArgs(.{
        dA.ptr,
        dB.ptr,
        dC.ptr,
        @as(u32, @intCast(case.m)),
        @as(u32, @intCast(case.k)),
        @as(u32, @intCast(case.n)),
    });

    const grid_x: c_uint = try zt.utils.cdiv(case.m, 32);
    const grid_y: c_uint = try zt.utils.cdiv(case.n, 32);

    try matmul_naive.launch(
        .{
            .grid = .{ .x = grid_x, .y = grid_y },
            .block = .{ .x = 32, .y = 32 },
        },
        args,
    );

    try ctx.sync();
    try dC.copyToHost(C);

    const expected = try gpa.alloc(f32, C.len);
    defer gpa.free(expected);

    matmulCPU(A, B, expected, case.m, case.k, case.n);

    for (0..C.len) |i| {
        try std.testing.expectApproxEqAbs(expected[i], C[i], 1e-4);
    }
}
