pub const BLOCK = @import("config.zig").BLOCK;

pub const AddressSpace = enum {
    reg,
    shared,
    global,
};

pub fn GlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) T;
}

pub fn ConstGlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) const T;
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

pub fn Tile(comptime T: type, comptime shape: anytype, comptime space: AddressSpace) type {
    if (shape.len != 1) {
        @compileError("Phase 1 only support 1D tiles");
    }

    const N = shape[0];

    return struct {
        value: T,

        pub const Element = T;
        pub const Shape = shape;
        pub const Space = space;
        pub const len = N;

        const Self = @This();

        pub fn addScalar(self: Self, value: T) Self {
            return .{ .value = self.value + value };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .value =  self.value + other.value};
        }
    };
}

pub fn RegTile(comptime T: type, comptime shape: anytype) type {
    return Tile(T, shape, .reg);
}

pub fn requireBlock(comptime block_size: u32) void {
    if (blockSize(0) != block_size) @trap();
}

pub const LoadOptions = struct {
    valid_len: u32,
};

pub fn load(
    comptime T: type,
    comptime block_size: u32,
    ptr: ConstGlobalPtr(T),
    opts: LoadOptions,
) RegTile(T, .{block_size}) {
    // per lane model: this thread own element 'lane' of a block_size-wide tile
    // PRECONDITION: blockDim.x == block_size. Enforced at runtime by requireBlock()
    const lane = threadId(0);
    return .{
        .value = if (lane < opts.valid_len) ptr[lane] else @as(T, 0),
    };
}

pub const StoreOptions = struct {
    valid_len: u32,
};

pub fn store(
    ptr: anytype,
    tile: anytype,
    opts: StoreOptions,
) void {
    // per lane model: this thread writes element 'lane' of a BLOCK-wide tile
    // PRECONDITION: blockDim.x == BLOCK. Enforced at runtime by requireBlock()
    const lane = threadId(0);
    if (lane < opts.valid_len) {
        ptr[lane] = tile.value;
    }
}
