const utils = @import("utils.zig");

// The tile is a real vector now
pub fn RegTile(comptime T: type, comptime ept: u32) type {
    return struct {
        data: @Vector(ept, T),
        const Self = @This();

        pub const Element = T;
        pub const element_per_thread = ept;

        pub fn addScalar(self: Self, v: T) Self {
            return .{ .data = self.data + @as(@Vector(ept, T), @splat(v)) };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .data = self.data + other.data };
        }
    };
}

pub fn load(
    comptime T: type,
    comptime ept: u32,
    ptr: utils.ConstGlobalPtr(T),
    n: u32,
) RegTile(T, ept) {
    const stride = utils.blockSize(0);
    const block_base = utils.blockId(0) * stride * ept; // this is owning the block
    var lanes: [ept]T = @splat(0);

    inline for (0..ept) |k| {
        const kk: u32 = @intCast(k);
        const idx = block_base + utils.threadId(0) + stride * kk;
        // thread 0 -> (0, 8, 16, 24) if THREAD = 8 and EPT = 4
        // thread 1 -> (1, 9, 17, 25) if THREAD = 8 and EPT = 4
        // this looks weird but when we look at the timesteps
        // at t = 0, [0, 1, 2, 3, 4, 5, 6, 7] -> this would be loaded
        // into all the threads. Continguous!!!
        if (idx < n) lanes[k] = ptr[idx];
    }

    return .{ .data = lanes };
}

pub fn store(
    ptr: anytype,
    tile: anytype,
    n: u32,
) void {
    const ept = @TypeOf(tile).element_per_thread; // infered from the tile type comptime
    const stride = utils.blockSize(0);
    const block_base = utils.blockId(0) * stride * ept;

    inline for (0..ept) |k| {
        const kk: u32 = @intCast(k);
        const idx = block_base + utils.threadId(0) + stride * kk;
        if (idx < n) ptr[idx] = tile.data[k];
    }
}

pub fn loadFill(
    comptime T: type,
    comptime ept: u32,
    ptr: utils.ConstGlobalPtr(T),
    n: u32,
    fill: T,
) RegTile(T, ept) {
    const stride = utils.blockSize(0);
    const block_base = utils.blockId(0) * stride * ept; // this is owning the block
    var lanes: [ept]T = @splat(fill);

    inline for (0..ept) |k| {
        const kk: u32 = @intCast(k);
        const idx = block_base + utils.threadId(0) + stride * kk;
        if (idx < n) lanes[k] = ptr[idx];
    }

    return .{ .data = lanes };
}
