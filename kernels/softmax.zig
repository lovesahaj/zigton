//! Numerically stable softmax over rows of a 2-D matrix.
//!
//! One CUDA block handles one row.  The implementation mirrors the
//! three-pass CUDA reference (softmax_smem_primitives) translated
//! into zigton idioms:
//!
//!   Pass 1  – stream the row into a shared-memory tile and find the
//!             per-block maximum  (blockReduceMax).
//!   Pass 2  – compute sum(exp(x – m)) using the cached shared row.
//!   Pass 3  – write exp(x – m) / sum back to global memory.
//!
//! Launch configuration expected by the host:
//!   grid  = (batch_size, 1, 1)   – one block per row
//!   block = (THREADS,   1, 1)    – THREADS from config (default 256)
//!
//! Shared memory required per block:
//!   sizeof(f32) * seq_len        – the row cache
//!   sizeof(f32) * THREADS        – scratch for blockReduceMax/Sum
//!
//! The kernel accepts seq_len as a runtime parameter so it is not
//! limited to a compile-time constant tile width.

const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;

const zt = if (is_device)
    @import("zigton_device")
else
    @import("zigton");

/// Numerically-stable row-wise softmax.
///
/// Parameters
/// ----------
/// in       – input  matrix  [batch_size × seq_len], row-major
/// out      – output matrix  [batch_size × seq_len], row-major
/// seq_len  – number of columns (runtime value)
pub export fn softmax(
    in: zt.ConstGlobalPtr(f32),
    out: zt.GlobalPtr(f32),
    seq_len: u32,
) callconv(.kernel) void {
    // One block per row.  The block id directly selects the row.
    const row_ptr_in  = in  + @as(u32, zt.utils.blockId(0)) * seq_len;
    const row_ptr_out = out + @as(u32, zt.utils.blockId(0)) * seq_len;

    const tid = zt.utils.threadId(0);
    const bsz = zt.utils.blockSize(0);

    // ------------------------------------------------------------------
    // Shared row cache  –  holds the entire row so we only read global
    // memory once and reuse values across Pass 2 and Pass 3.
    // The tag `.softmax_row_cache` makes the static shared buffer unique
    // within the translation unit (same mechanism as matmul tiles).
    // ------------------------------------------------------------------
    // We need a runtime-length shared buffer.  zigton's sharedTile helper
    // is compile-time sized, so we declare a fixed upper-bound tile and
    // guard every access with `< seq_len`.
    //
    // Max supported seq_len == MAX_SEQ (adjust if you need longer rows).
    const MAX_SEQ = 4096;
    const row_cache = zt.sharedTile(f32, MAX_SEQ, .softmax_row_cache);

    // ------------------------------------------------------------------
    // Pass 1 – load row into shared memory, find block-wide maximum.
    // ------------------------------------------------------------------
    var local_max: f32 = -std.math.inf(f32);

    var i: u32 = tid;
    while (i < seq_len) : (i += bsz) {
        const x = row_ptr_in[i];
        row_cache.store(i, x);
        local_max = @max(local_max, x);
    }

    const m = zt.blockReduceMax(f32, local_max);

    // ------------------------------------------------------------------
    // Pass 2 – compute denominator  l = sum(exp(x - m))  from the cache.
    // ------------------------------------------------------------------
    var local_sum: f32 = 0.0;

    i = tid;
    while (i < seq_len) : (i += bsz) {
        local_sum += @exp(row_cache.load(i) - m);
    }

    const l     = zt.blockReduceSum(f32, local_sum);
    const inv_l = 1.0 / l;

    // ------------------------------------------------------------------
    // Pass 3 – write normalised values to global output.
    // blockReduceSum/Max end with a blockSync, so the cache is still
    // coherent here; no additional sync is needed before reading it.
    // ------------------------------------------------------------------
    i = tid;
    while (i < seq_len) : (i += bsz) {
        row_ptr_out[i] = @exp(row_cache.load(i) - m) * inv_l;
    }
}
