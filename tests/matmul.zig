const std = @import("std");
const zt = @import("zigton");

const matmul: [:0]const u8 = @embedFile("matmul_ptx");

const MatmulCase = struct {
    m: usize,
    k: usize,
    n: usize,
};

const MatmulLaunch = struct {
    grid_x: c_uint,
    grid_y: c_uint,
    block_x: c_uint,
    block_y: c_uint,
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
    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(matmul);
    defer module.deinit();

    const matmul_naive: zt.Kernel = try module.kernel("matmul_naive");
    const matmul_coalsce: zt.Kernel = try module.kernel("matmul_coalsce");
    const matmul_tile: zt.Kernel = try module.kernel("matmul_tiled");
    const matmul_tiled_tm: zt.Kernel = try module.kernel("matmul_tiled_tm");
    const matmul_tiled_2d: zt.Kernel = try module.kernel("matmul_tiled_2d");

    const cases = [_]MatmulCase{
        .{ .m = 4, .k = 5, .n = 3 },
        .{ .m = 33, .k = 7, .n = 35 },
        .{ .m = 17, .k = 31, .n = 19 },
        .{ .m = 32, .k = 32, .n = 32 },
    };

    for (cases) |case| {
        try expectMatmul(&ctx, &matmul_naive, "matmul_naive", case, try naiveLaunch(case));
        try expectMatmul(&ctx, &matmul_coalsce, "matmul_coalsce", case, try coalsceLaunch(case));
        try expectMatmul(&ctx, &matmul_tile, "matmul_tiled", case, try tileLaunch(case));
        try expectMatmul(&ctx, &matmul_tiled_tm, "matmul_tiled_tm", case, try tileTMLaunch(case));
        try expectMatmul(&ctx, &matmul_tiled_2d, "matmul_tiled_2d", case, try tileLaunch(case));
    }
}

fn expectMatmul(
    ctx: *zt.Context,
    kernel: *const zt.Kernel,
    name: []const u8,
    case: MatmulCase,
    launch: MatmulLaunch,
) !void {
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

    try kernel.launch(
        .{
            .grid = .{ .x = launch.grid_x, .y = launch.grid_y },
            .block = .{ .x = launch.block_x, .y = launch.block_y },
        },
        args,
    );

    try ctx.sync();
    try dC.copyToHost(C);

    const expected = try gpa.alloc(f32, C.len);
    defer gpa.free(expected);

    matmulCPU(A, B, expected, case.m, case.k, case.n);

    for (0..C.len) |i| {
        const ok = if (expected[i] == 0.0)
            std.math.approxEqAbs(f32, expected[i], C[i], 1e-4)
        else
            std.math.approxEqRel(f32, expected[i], C[i], 1e-4);

        if (!ok) {
            std.debug.print(
                "{s} mismatch case={d}x{d}x{d} index={d} expected={d} actual={d}\n",
                .{ name, case.m, case.k, case.n, i, expected[i], C[i] },
            );
            return error.MatmulMismatch;
        }
    }
}

fn naiveLaunch(case: MatmulCase) !MatmulLaunch {
    // matmul_naive maps x -> row and y -> col.
    return .{
        .grid_x = try zt.utils.cdiv(case.m, 32),
        .grid_y = try zt.utils.cdiv(case.n, 32),
        .block_x = 32,
        .block_y = 32,
    };
}

fn coalsceLaunch(case: MatmulCase) !MatmulLaunch {
    // matmul_coalsce maps x -> col and y -> row, so warp lanes advance across
    // columns for coalesced B loads and C stores.
    return .{
        .grid_x = try zt.utils.cdiv(case.n, 32),
        .grid_y = try zt.utils.cdiv(case.m, 32),
        .block_x = 32,
        .block_y = 32,
    };
}

fn tileLaunch(case: MatmulCase) !MatmulLaunch {
    // matmul_coalsce maps x -> col and y -> row, so warp lanes advance across
    // columns for coalesced B loads and C stores.
    return .{
        .grid_x = try zt.utils.cdiv(case.n, 16),
        .grid_y = try zt.utils.cdiv(case.m, 16),
        .block_x = 16,
        .block_y = 16,
    };
}

fn tileTMLaunch(case: MatmulCase) !MatmulLaunch {
    // matmul_coalsce maps x -> col and y -> row, so warp lanes advance across
    // columns for coalesced B loads and C stores.
    const BM = 64; // block M dim
    const BN = 16; // block N dim
    const TM = 4; // number of elements in a thread
    return .{
        .grid_x = try zt.utils.cdiv(case.n, BN),
        .grid_y = try zt.utils.cdiv(case.m, BM),
        .block_x = BN,
        .block_y = BM / TM,
    };
}

fn tile2DLaunch(case: MatmulCase) !MatmulLaunch {
    // matmul_coalsce maps x -> col and y -> row, so warp lanes advance across
    // columns for coalesced B loads and C stores.
    const BM = 64; // block M dim
    const BN = 16; // block N dim
    const TM = 4; // number of elements in a thread
    const TN = 4; // number of elements in a thread
    return .{
        .grid_x = try zt.utils.cdiv(case.n, BN),
        .grid_y = try zt.utils.cdiv(case.m, BM),
        .block_x = BN / TN,
        .block_y = BM / TM,
    };
}
