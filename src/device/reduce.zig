const THREADS = @import("config.zig").THREADS;
const utils = @import("utils.zig");
const shared = @import("sharedtile.zig");

pub fn blockReduceSum(
    comptime T: type,
    value: T,
) T {
    const tid = utils.threadId(0);
    const partials = shared.sharedTile(T, THREADS, .block_reduce_sum,);

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
