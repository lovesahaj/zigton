const std = @import("std");
const builtin = @import("builtin");

const is_device = builtin.target.cpu.arch == .nvptx64;
const kernel_callconv: std.builtin.CallingConvention = if (is_device) .kernel else .c;

const zt = if (is_device)
    @import("zigton_device")
else
    @import("zigton");

pub export fn block_sum(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    blockReduceKernel(.Add, x, z, n);
}

pub export fn block_max(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(kernel_callconv) void {
    blockReduceKernel(.Max, x, z, n);
}

inline fn blockReduceKernel(
    comptime op: std.builtin.ReduceOp,
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) void {
    zt.requireBlock(zt.THREADS);
    if (n == 0) return;

    const fill = switch (op) {
        .Add => 0.0,
        .Max => -std.math.inf(f32),
        .Min => std.math.inf(f32),
        .Mul => 1.0,
        else => @compileError("unsupported reduction op"),
    };

    const tile = zt.loadFill(f32, zt.EPT, x, n, fill);

    const T = @TypeOf(tile).Element;

    const partial_max: T = @reduce(op, tile.data);
    const block_max_value = zt.blockReduce(op, T, partial_max);

    if (zt.utils.threadId(0) == 0) z[zt.utils.blockId(0)] = block_max_value;
}
