pub const AddressSpace = enum {
    reg,
    shared,
    global,
};

pub inline fn blockSync() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

pub fn GlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) T;
}

pub fn ConstGlobalPtr(comptime T: type) type {
    return [*]addrspace(.global) const T;
}

pub inline fn SharedPtr(comptime T: type) type {
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


pub fn requireBlock(comptime threads: u32) void {
    if (blockSize(0) != threads) @trap();
}


