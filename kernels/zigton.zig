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
        data: [N]T,

        pub const Element = T;
        pub const Shape = shape;
        pub const Space = space;
        pub const len = N;

        const Self = @This();

        pub fn addScalar(self: Self, value: T) Self {
            var out: Self = undefined;

            inline for (0..N) |i| {
                out.data[i] = self.data[i] + value;
            }

            return out;
        }
    };
}

pub fn RegTile(comptime T: type, comptime shape: anytype) type {
    return Tile(T, shape, .reg);
}

pub const LoadOptions = struct {
    mask: bool = true,
};

pub fn load(
    comptime T: type,
    comptime shape: anytype,
    ptr: ConstGlobalPtr(T),
    opts: LoadOptions,
) RegTile(T, shape) {
    // creating the struct tile of register memory
    const TileT = RegTile(T, shape);

    // this needs be the same tile as the global memory
    var out: TileT = undefined;

    // adding the values to the tiles
    // mask is for when we need to a calculation
    // TODO: mask only can be 0 right now. fix for mask = inf for softmax
    // right now we are just using mask to know where to perform an
    // operation and where to not.
    inline for (0..TileT.len) |i| {
        out.data[i] = if (opts.mask) ptr[i] else @as(T, 0);
    }

    return out;
}

pub const StoreOptions = struct {
    mask: bool = true,
};

pub fn store(
    ptr: anytype,
    tile: anytype,
    opts: StoreOptions,
) void {
    const TileT = @TypeOf(tile);

    inline for (0..TileT.len) |i| {
        if (opts.mask) {
            ptr[i] = tile.data[i];
        }
    }
}
