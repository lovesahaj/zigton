const std = @import("std");
const THREADS = @import("config.zig").THREADS;
const utils = @import("utils.zig");
const shared = @import("sharedtile.zig");

inline fn applyReduceOp(
    comptime op: std.builtin.ReduceOp,
    a: anytype,
    b: @TypeOf(a),
) @TypeOf(a) {
    return switch (op) {
        .Add => a + b,
        .Max => @max(a, b),
        .Min => @min(a, b),
        .Mul => a * b,
        else => @compileError("unsupported block operation"),
    };
}

pub fn blockReduceSum(
    comptime T: type,
    value: T,
) T {
    const tid = utils.threadId(0);
    const partials = shared.sharedTile(T, THREADS, .block_reduce_sum);

    partials.store(tid, value);
    utils.blockSync();

    var stride: u32 = THREADS / 2;

    while (stride > 0) : (stride /= 2) {
        if (tid < stride) {
            const a = partials.load(tid);
            const b = partials.load(tid + stride);
            partials.store(tid, a + b);
        }
        utils.blockSync();
    }

    return partials.load(0);
}

pub fn blockReduceMax(
    comptime T: type,
    value: T,
) T {
    const tid = utils.threadId(0);
    const partials = shared.sharedTile(T, THREADS, .block_reduce_max);

    partials.store(tid, value);
    utils.blockSync();

    var stride: u32 = THREADS / 2;

    while (stride > 0) : (stride /= 2) {
        if (tid < stride) {
            const a = partials.load(tid);
            const b = partials.load(tid + stride);
            partials.store(tid, @max(a, b));
        }

        utils.blockSync();
    }

    return partials.load(0);
}

pub inline fn blockReduce(
    comptime op: std.builtin.ReduceOp,
    comptime T: type,
    value: T,
) T {
    const tid = utils.threadId(0);
    const partials = shared.sharedTile(T, THREADS, .block_reduce_sum);

    partials.store(tid, value);
    utils.blockSync();

    var stride: u32 = THREADS / 2;

    while (stride > 0) : (stride /= 2) {
        if (tid < stride) {
            const a = partials.load(tid);
            const b = partials.load(tid + stride);
            partials.store(tid, applyReduceOp(op, a, b));
        }
        utils.blockSync();
    }

    return partials.load(0);
}
