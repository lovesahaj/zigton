const std = @import("std");
const Io = std.Io;
const zt = @import("zigton");

const matmul_ptx: [:0]const u8 = @embedFile("matmul_ptx");

const warmup_iters = 5;
const timed_iters = 20;

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

const BenchStats = struct {
    min_ns: i96,
    p50_ns: i96,
    p80_ns: i96,
    max_ns: i96,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(matmul_ptx);
    defer module.deinit();

    const matmul_naive = try module.kernel("matmul_naive");
    const matmul_coalsce = try module.kernel("matmul_coalsce");
    const matmul_tiled = try module.kernel("matmul_tiled");
    const matmul_tiled_tm = try module.kernel("matmul_tiled_tm");
    const matmul_tiled_2d = try module.kernel("matmul_tiled_2d");

    const cases = [_]MatmulCase{
        .{ .m = 512, .k = 512, .n = 512 },
        .{ .m = 1024, .k = 1024, .n = 1024 },
        .{ .m = 4096, .k = 4096, .n = 4096 },
    };

    std.debug.print("bench-matmul warmup={d} timed={d}\n", .{ warmup_iters, timed_iters });

    for (cases) |case| {
        try benchKernel(
            io,
            allocator,
            &ctx,
            &matmul_naive,
            "matmul_naive",
            case,
            try naiveLaunch(case),
        );

        try benchKernel(
            io,
            allocator,
            &ctx,
            &matmul_coalsce,
            "matmul_coalsce_compat",
            case,
            try coalsceCompatibleLaunch(case),
        );

        try benchKernel(
            io,
            allocator,
            &ctx,
            &matmul_tiled,
            "matmul_tiled",
            case,
            try tiledLaunch(case),
        );

        try benchKernel(
            io,
            allocator,
            &ctx,
            &matmul_tiled_tm,
            "matmul_tiled_tm",
            case,
            try tileTMLaunch(case),
        );
        
        try benchKernel(
            io,
            allocator,
            &ctx,
            &matmul_tiled_2d,
            "matmul_tiled_2d",
            case,
            try tile2DLaunch(case),
        );
    }
}

fn benchKernel(
    io: Io,
    allocator: std.mem.Allocator,
    ctx: *zt.Context,
    kernel: *const zt.Kernel,
    name: []const u8,
    case: MatmulCase,
    launch: MatmulLaunch,
) !void {
    const A = try allocator.alloc(f32, case.m * case.k);
    const B = try allocator.alloc(f32, case.k * case.n);
    const C = try allocator.alloc(f32, case.m * case.n);

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

    const stats = try doBench(io, ctx, kernel, args, launch);
    const p50_ms = nsToMs(stats.p50_ns);

    const flops = 2.0 * @as(f64, @floatFromInt(case.m)) * @as(f64, @floatFromInt(case.n)) * @as(f64, @floatFromInt(case.k));
    const p50_gflops = flops / nsToSeconds(stats.p50_ns) / 1e9;

    std.debug.print(
        "{s} {d}x{d}x{d} grid=({d},{d}) block=({d},{d}) min={d:.3}ms p50={d:.3}ms p80={d:.3}ms max={d:.3}ms p50={d:.2}GF/s\n",
        .{
            name,
            case.m,
            case.k,
            case.n,
            launch.grid_x,
            launch.grid_y,
            launch.block_x,
            launch.block_y,
            nsToMs(stats.min_ns),
            p50_ms,
            nsToMs(stats.p80_ns),
            nsToMs(stats.max_ns),
            p50_gflops,
        },
    );
}

fn doBench(
    io: Io,
    ctx: *zt.Context,
    kernel: *const zt.Kernel,
    args: anytype,
    launch: MatmulLaunch,
) !BenchStats {
    for (0..warmup_iters) |_| {
        try launchKernel(kernel, args, launch);
    }
    try ctx.sync();

    var timings: [timed_iters]i96 = undefined;

    for (&timings) |*timing| {
        const started = Io.Clock.Timestamp.now(io, .awake);
        try launchKernel(kernel, args, launch);
        try ctx.sync();
        timing.* = started.untilNow(io).raw.toNanoseconds();
    }

    std.mem.sort(i96, timings[0..], {}, lessThanI96);

    return .{
        .min_ns = timings[0],
        .p50_ns = percentile(&timings, 50),
        .p80_ns = percentile(&timings, 80),
        .max_ns = timings[timings.len - 1],
    };
}

fn launchKernel(kernel: *const zt.Kernel, args: anytype, launch: MatmulLaunch) !void {
    try kernel.launch(
        .{
            .grid = .{ .x = launch.grid_x, .y = launch.grid_y },
            .block = .{ .x = launch.block_x, .y = launch.block_y },
        },
        args,
    );
}

fn percentile(timings: *const [timed_iters]i96, comptime p: u32) i96 {
    const index = ((timings.len - 1) * p) / 100;
    return timings[index];
}

fn lessThanI96(_: void, lhs: i96, rhs: i96) bool {
    return lhs < rhs;
}

fn nsToMs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms));
}

fn nsToSeconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
}

const TILE = 32;
const TILED_MATMUL_TILE = 16;

fn naiveLaunch(case: MatmulCase) !MatmulLaunch {
    return .{
        .grid_x = try zt.utils.cdiv(case.m, TILE),
        .grid_y = try zt.utils.cdiv(case.n, TILE),
        .block_x = TILE,
        .block_y = TILE,
    };
}

fn coalsceCompatibleLaunch(case: MatmulCase) !MatmulLaunch {
    return .{
        .grid_x = try zt.utils.cdiv(case.n, TILE),
        .grid_y = try zt.utils.cdiv(case.m, TILE),
        .block_x = TILE,
        .block_y = TILE,
    };
}

fn tiledLaunch(case: MatmulCase) !MatmulLaunch {
    return .{
        .grid_x = try zt.utils.cdiv(case.n, TILED_MATMUL_TILE),
        .grid_y = try zt.utils.cdiv(case.m, TILED_MATMUL_TILE),
        .block_x = TILED_MATMUL_TILE,
        .block_y = TILED_MATMUL_TILE,
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
    const BN = 64; // block N dim
    const TM = 4; // number of elements in a thread
    const TN = 4; // number of elements in a thread
    return .{
        .grid_x = try zt.utils.cdiv(case.n, BN),
        .grid_y = try zt.utils.cdiv(case.m, BM),
        .block_x = BN / TN,
        .block_y = BM / TM,
    };
}
