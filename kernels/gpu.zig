const builtin = @import("builtin");

const zt = struct {
    // @workGroupId -> ctadi.x (block id)
    // @workGroupSize -> ntid.x (block size)
    // @workItemId -> tid. (thread idx)

    pub fn blockId(comptime dim: u32) u32 {
        return @workGroupId(dim);
    }

    pub fn blockSize(comptime dim: u32) u32 {
        return @workGroupSize(dim);
    }

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
};

pub export fn vector_add(
    x: [*]addrspace(.global) const f32,
    y: [*]addrspace(.global) const f32,
    z: [*]addrspace(.global) f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

pub export fn fill(
    z: [*]addrspace(.global) f32,
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = value;
    }
}

pub export fn add_scalar(
    x: [*] addrspace(.global) const f32,
    z: [*] addrspace(.global) f32,
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + value;
    }
}
