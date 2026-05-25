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
