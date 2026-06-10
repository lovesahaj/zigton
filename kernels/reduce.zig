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
    zt.requireBlock(zt.THREADS);
    if (n == 0) return;
    const tile = zt.load(f32, zt.EPT, x, n);

    const T = @TypeOf(tile).Element;

    const partial_sum: T = @reduce(.Add, tile.data);
    const block_sum_value = zt.blockReduceSum(T, partial_sum);

    if (zt.utils.threadId(0) == 0) z[zt.utils.blockId(0)] = block_sum_value;
}

pub export fn block_max(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(kernel_callconv) void {
    zt.requireBlock(zt.THREADS);
    if (n == 0) return;

    const tile = zt.loadFill(f32, zt.EPT, x, n, -std.math.inf(f32));

    const T = @TypeOf(tile).Element;

    const partial_max: T = @reduce(.Max, tile.data);
    const block_max_value = zt.blockReduceSum(T, partial_max);

    if (zt.utils.threadId(0) == 0) z[zt.utils.blockId(0)] = block_max_value;
}
