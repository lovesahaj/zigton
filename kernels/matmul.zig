const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;
const kernel_callconv: std.builtin.CallingConvention = if (is_device) .kernel else .c;

const zt =
    @import("zigton_device");

pub export fn matmul_naive(
    A: zt.ConstGlobalPtr(f32),
    B: zt.ConstGlobalPtr(f32),
    C: zt.GlobalPtr(f32),
    M: u32,
    K: u32,
    N: u32,
) callconv(.kernel) void {
    // r and c give me the row and col
    // of the matrix C that this thread
    // is trying to calculate
    const r = zt.utils.blockId(0) * zt.utils.blockSize(0) + zt.utils.threadId(0);
    const c = zt.utils.blockId(1) * zt.utils.blockSize(1) + zt.utils.threadId(1);

    if (r < M and c < N) {
        var tmp: f32 = 0.0;
        for (0..K) |i| {
            tmp += A[r * K + i] * B[i * N + c];
        }
        C[r * N + c] = tmp;
    }
}

pub export fn matmul_coalsce(
    A: zt.ConstGlobalPtr(f32),
    B: zt.ConstGlobalPtr(f32),
    C: zt.GlobalPtr(f32),
    M: u32,
    K: u32,
    N: u32,
) callconv(.kernel) void {
    // r and c give me the row and col
    // of the matrix C that this thread
    // is trying to calculate
    const r = zt.utils.blockId(1) * zt.utils.blockSize(1) + zt.utils.threadId(1);
    const c = zt.utils.blockId(0) * zt.utils.blockSize(0) + zt.utils.threadId(0);

    if (r < M and c < N) {
        var tmp: f32 = 0.0;
        for (0..K) |i| {
            tmp += A[r * K + i] * B[i * N + c];
        }
        C[r * N + c] = tmp;
    }
}


// pub export fn matmul_shared_memory_cache_blocking(
//     A: zt.ConstGlobalPtr(f32),
//     B: zt.ConstGlobalPtr(f32),
//     C: zt.GlobalPtr(f32),
//     M: u32,
//     K: u32,
//     N: u32,
// ) callconv(.kernel) void {
//     // r and c give me the row and col
//     // of the matrix C that this thread
//     // is trying to calculate
//     const BLOCK_SIZE = 32;
//
//     const cRow = zt.utils.blockId(0) * BLOCK_SIZE;
//     const cCol = zt.utils.blockId(1) * BLOCK_SIZE;
//
//     const r = cRow + zt.utils.threadId(0) / BLOCK_SIZE;
//     const c = cCol + zt.utils.threadId(0) % BLOCK_SIZE;
//
//     // const local_tile = zt.utils.blockId(0) * zt.utils.blockSize(0) + zt.utils.threadId(1);
//
//     if (r < M and c < N) {
//         var tmp: f32 = 0.0;
//         for (0..K) |i| {
//             tmp += A[r * K + i] * B[i * N + c];
//         }
//         C[r * N + c] = tmp;
//     }
// }
