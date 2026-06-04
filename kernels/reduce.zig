const zt = @import("zigton_device");

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
