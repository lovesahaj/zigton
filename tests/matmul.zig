const std = @import("std");
const zt = @import("zigton");

const matmul: [:0]const u8 = @embedFile("matmul_ptx");
const M_dim = 4096;
const K_dim = 4096;
const N_dim = 4096;
const n: u32 = M_dim * N_dim;

fn matmulCPU(
    A: []f32,
    B: []f32,
    C: []f32,
    M: u32,
    K: u32,
    N: u32,
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
    const gpa = std.testing.allocator;

    const A = try gpa.alloc(f32, M_dim * K_dim);
    defer gpa.free(A);
    const B = try gpa.alloc(f32, K_dim * N_dim);
    defer gpa.free(B);
    const C = try gpa.alloc(f32, M_dim * N_dim);
    defer gpa.free(C);

    for (0..(M_dim * K_dim)) |i| {
        A[i] = @as(f32, @floatFromInt(i));
    }

    for (0..(N_dim * K_dim)) |i| {
        B[i] = @as(f32, @floatFromInt(i));
    }

    @memset(C, 0.0);


    var ctx = try zt.Context.init(.{ .device_index = 0 });
    defer ctx.deinit();

    var module = try zt.Module.loadData(matmul);
    defer module.deinit();

    const vector_add: zt.Kernel = try module.kernel("matmul_naive");

    var dA = try zt.DeviceBuffer(f32).init(M_dim * K_dim);
    defer dA.deinit();

    var dB = try zt.DeviceBuffer(f32).init(N_dim * K_dim);
    defer dB.deinit();

    var dC = try zt.DeviceBuffer(f32).init(M_dim * N_dim);
    defer dC.deinit();

    try dA.copyFromHost(A);
    try dB.copyFromHost(B);

    const args = zt.kernelArgs(.{ dA.ptr, dB.ptr, dC.ptr, M_dim, K_dim, N_dim });

    const grid_x: c_uint = try zt.utils.cdiv(M_dim, 32);
    const grid_y: c_uint = try zt.utils.cdiv(N_dim, 32);

    try vector_add.launch(
        .{
            .grid = .{ .x = grid_x, .y = grid_y },
            .block = .{ .x = 32, .y = 32 },
        },
        args,
    );

    try ctx.sync();
    try dC.copyToHost(C);

    const expected = try gpa.alloc(f32, M_dim * N_dim);
    defer gpa.free(expected);

    matmulCPU(A, B, expected, M_dim, K_dim, N_dim);

    for (0..n) |i| {
        try std.testing.expectEqual(expected[i], C[i]);
    }
}
