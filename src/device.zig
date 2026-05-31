pub const THREADS = @import("config.zig").THREADS;
pub const EPT = @import("config.zig").EPT;
pub const TILE = @import("config.zig").TILE;

pub const AddressSpace = enum {
    reg,
    shared,
    global,
};

pub inline fn blockSync() void {
    asm volatile ("bar.sync 0;" ::: "memory");
}

pub fn GlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) T;
}

pub fn ConstGlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) const T;
}

pub fn SharedPtr(comptime T: type) type {
    return [*]addrspace(.shared) T;
}

// @workGroupId -> ctadi.x (block id)
pub fn blockId(comptime dim: u32) u32 {
    return @workGroupId(dim);
}

// @workGroupSize -> ntid.x (block size)
pub fn blockSize(comptime dim: u32) u32 {
    return @workGroupSize(dim);
}

// @workItemId -> tid. (thread idx)
pub fn threadId(comptime dim: u32) u32 {
    return @workItemId(dim);
}

pub fn linearIndex() u32 {
    return blockId(0) * blockSize(0) + threadId(0);
}

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

pub fn requireBlock(comptime threads: u32) void {
    if (blockSize(0) != threads) @trap();
}

pub fn load(
    comptime T: type,
    comptime ept: u32,
    ptr: ConstGlobalPtr(T),
    n: u32,
) RegTile(T, ept) {
    const stride = blockSize(0);
    const block_base = blockId(0) * stride * ept; // this is owning the block
    var lanes: [ept]T = @splat(0);

    inline for (0..ept) |k| {
        const kk: u32 = @intCast(k);
        const idx = block_base + threadId(0) + stride * kk;
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
    const stride = blockSize(0);
    const block_base = blockId(0) * stride * ept;

    inline for (0..ept) |k| {
        const kk: u32 = @intCast(k);
        const idx = block_base + threadId(0) + stride * kk;
        if (idx < n) ptr[idx] = tile.data[k];
    }
}
