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

pub export fn matmul_tiled(
    A: zt.ConstGlobalPtr(f32),
    B: zt.ConstGlobalPtr(f32),
    C: zt.GlobalPtr(f32),
    M: u32,
    K: u32,
    N: u32,
) callconv(.kernel) void {
    const BLOCK_SIZE = 16;

    const As = zt.sharedTile(f32, BLOCK_SIZE * BLOCK_SIZE, .As);
    const Bs = zt.sharedTile(f32, BLOCK_SIZE * BLOCK_SIZE, .Bs);

    // i am going to swap the row for cols during launch
    const outer_row = zt.utils.blockId(1);
    const outer_col = zt.utils.blockId(0);

    // i call these inner because these are the thread indexes
    // inside each block. Also they corresponding exactly to
    // what each thread is going to store in the shared memory
    // this is coalasced as well
    const inner_row = zt.utils.threadId(1);
    const inner_col = zt.utils.threadId(0);
    const smem_idx = inner_row * BLOCK_SIZE + inner_col;

    // global coords of the C elements this thread owns
    const row = outer_row * BLOCK_SIZE + inner_row;
    const col = outer_col * BLOCK_SIZE + inner_col;

    // now we need to navigate to the, start of this till
    // we update the A, B, and C pointers
    var inner_A = A + (outer_row * BLOCK_SIZE * K);
    var inner_B = B + outer_col * BLOCK_SIZE;
    var inner_C = C + (outer_row * BLOCK_SIZE * N) + (outer_col * BLOCK_SIZE);

    var tmp: f32 = 0.0;

    var bIdx: u32 = 0;
    while (bIdx < K) : (bIdx += BLOCK_SIZE) {
        const a_k = bIdx + inner_col; // k index this thread reads from A
        const b_k = bIdx + inner_row; // k index this thread reads from B

        // guarded, zero-filled loads -> every thread would reach the sync
        As.store(smem_idx, if (row < M and a_k < K) inner_A[inner_row * K + inner_col] else 0.0);
        Bs.store(smem_idx, if (b_k < K and col < N) inner_B[inner_row * N + inner_col] else 0.0);
        zt.blockSync();

        inner_A += BLOCK_SIZE;
        inner_B += BLOCK_SIZE * N;

        for (0..BLOCK_SIZE) |dotidx| {
            tmp += As.load(@intCast(inner_row * BLOCK_SIZE + dotidx)) *
                Bs.load(@intCast(dotidx * BLOCK_SIZE + inner_col));
        }

        // no predicates, therefore no branching
        zt.blockSync();
    }

    if (row < M and col < N) {
        inner_C[inner_row * N + inner_col] = tmp;
    }
}
