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

pub fn Tile1(comptime T: type, comptime N: u32) type {
    return struct {
        data: [N]T,

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
