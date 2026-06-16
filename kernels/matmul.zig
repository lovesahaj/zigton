const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;
const kernel_callconv: std.builtin.CallingConvention = if (is_device) .kernel else .c;

// const zt = if (is_device)
//     @import("zigton_device")
// else
//     @import("zigton");
const zt = @import("zigton_device");

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

pub export fn matmul_tiled_tm(
    A: zt.ConstGlobalPtr(f32),
    B: zt.ConstGlobalPtr(f32),
    C: zt.GlobalPtr(f32),
    M: u32,
    K: u32,
    N: u32,
) callconv(.kernel) void {
    // each thread computes TM elements
    // each thread keep TM accumlators in register -> this would be a tile as well
    // shared B value is reused across TM row results

    const BM = 64; // block M dim
    const BN = 16; // block N dim
    const BK = 16; // block K dim
    const TM = 4; // number of elements in a thread

    const As = zt.sharedTile(f32, BM * BK, .matmul_tm_a);
    const Bs = zt.sharedTile(f32, BK * BN, .matmul_tm_b);

    const thread_col = zt.utils.threadId(0);
    const thread_row = zt.utils.threadId(1);

    // we flatten the thread ids because it is BN != BK
    // we won't be able to multiply the A and B
    const local_tid = thread_row * BN + thread_col;

    const block_row = zt.utils.blockId(1) * BM;
    const block_col = zt.utils.blockId(0) * BN;

    var acc: [TM]f32 = @splat(0.0);

    var k0: u32 = 0;
    while (k0 < K) : (k0 += BK) {
        inline for (0..TM) |load_i| {
            const a_linear: u32 = @as(u32, @intCast(local_tid + load_i * (BN * (BM / TM))));
            const a_row = a_linear / BK;
            const a_col = a_linear % BK;

            const global_row = a_row + block_row;
            const global_col = k0 + a_col;
            As.store(
                a_linear,
                if (global_row < M and global_col < K) A[global_row * K + global_col] else 0.0,
            );
        }

        const b_row = local_tid / BN;
        const b_col = local_tid % BN;

        const global_row_b = b_row + k0;
        const global_col_b = block_col + b_col;

        Bs.store(
            local_tid,
            if (global_row_b < K and global_col_b < N) B[global_row_b * N + global_col_b] else 0.0,
        );

        zt.blockSync();

        var dot: u32 = 0;
        while (dot < BK) : (dot += 1) {
            const b_tmp = Bs.load(dot * BN + thread_col);
            inline for (0..TM) |res_i| {
                const a_row = thread_row * TM + res_i;
                acc[res_i] += As.load(@as(u32, @intCast(a_row * BK + dot))) * b_tmp;
            }
        }

        zt.blockSync();
    }

    // this would point to the exact col and row start, that
    // this thread owns
    const col = block_col + thread_col;
    const row_base = block_row + thread_row * TM;

    inline for (0..TM) |res_i| {
        const row = row_base + res_i;
        if (row < M and col < N) {
            C[row * N + col] = acc[res_i];
        }
    }
}

inline fn accIdx(comptime TN: u32, mi: u32, ni: u32) u32 {
    return mi * TN + ni;
}

pub export fn matmul_tiled_2d(
    A: zt.ConstGlobalPtr(f32),
    B: zt.ConstGlobalPtr(f32),
    C: zt.GlobalPtr(f32),
    M: u32,
    K: u32,
    N: u32,
) callconv(.kernel) void {
    const BM = 64;
    const BN = 64;
    const BK = 32;
    const TM = 4;
    const TN = 4;

    const THREADS_M = BM / TM; // 16
    const THREADS_N = BN / TN; // 4

    const As = zt.sharedTile(f32, BM * BK, .matmul_2d_a);
    const Bs = zt.sharedTile(f32, BK * BN, .matmul_2d_b);

    const thread_m = zt.utils.threadId(1);
    const thread_n = zt.utils.threadId(0);

    const block_row = zt.utils.blockId(1) * BM;
    const block_col = zt.utils.blockId(0) * BN;

    var acc: [TM * TN]f32 = @splat(0.0);

    const local_tid = thread_m * THREADS_N + thread_n;
    const num_threads = THREADS_M * THREADS_N; // 64 threads

    // Calculate how many elements each thread needs to load to fill
    // the shared tiles
    const a_loads_per_thread = (BM * BK) / num_threads; // 1024 / 64 = 16
    const b_loads_per_thread = (BK * BN) / num_threads; // 256 / 64 = 4

    var k0: u32 = 0;
    while (k0 < K) : (k0 += BK) {

        // 1. Load Matrix A into Shared Memory
        inline for (0..a_loads_per_thread) |load_i| {
            const a_linear: u32 = @intCast(local_tid + load_i * num_threads);
            const a_row = a_linear / BK;
            const a_col = a_linear % BK;

            const global_row = block_row + a_row;
            const global_col = k0 + a_col;

            As.store(
                a_linear,
                if (global_row < M and global_col < K) A[global_row * K + global_col] else 0.0,
            );
        }

        // 2. Load Matrix B into Shared Memory
        inline for (0..b_loads_per_thread) |load_i| {
            const b_linear: u32 = @intCast(local_tid + load_i * num_threads);
            const b_row = b_linear / BN;
            const b_col = b_linear % BN;

            const global_row = k0 + b_row;
            const global_col = block_col + b_col;

            Bs.store(
                b_linear, // Fixed: changed from local_tid
                if (global_row < K and global_col < N) B[global_row * N + global_col] else 0.0,
            );
        }

        zt.blockSync();

        // 3. Compute Thread-Local Outer Product
        var dot: u32 = 0;
        while (dot < BK) : (dot += 1) {
            var a_frag: [TM]f32 = undefined;
            var b_frag: [TN]f32 = undefined;

            inline for (0..TM) |mi| {
                const row = thread_m * TM + mi;
                a_frag[mi] = As.load(@intCast(row * BK + dot));
            }

            inline for (0..TN) |ni| {
                const col = thread_n * TN + ni;
                b_frag[ni] = Bs.load(@intCast(dot * BN + col));
            }

            inline for (0..TM) |mi| {
                inline for (0..TN) |ni| {
                    acc[accIdx(TN, mi, ni)] += a_frag[mi] * b_frag[ni];
                }
            }
        }

        zt.blockSync();
    }

    // 4. Write Results back to Global Memory (with boundary checks)
    const col_base = block_col + thread_n * TN;
    const row_base = block_row + thread_m * TM;

    inline for (0..TM) |mi| {
        inline for (0..TN) |ni| {
            const row = row_base + mi;
            const col = col_base + ni;
            if (row < M and col < N) {
                C[row * N + col] = acc[accIdx(TN, mi, ni)];
            }
        }
    }
}
