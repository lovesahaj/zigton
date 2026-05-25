const builtin = @import("builtin");

pub export fn vector_add(
    x: [*]addrspace(.global) const f32,
    y: [*]addrspace(.global) const f32,
    z: [*]addrspace(.global) f32,
    n: u32,
) callconv(.kernel) void {
    const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);

    // @workGroupId -> ctadi.x (block id)
    // @workGroupSize -> ntid.x (block size)
    // @workItemId -> tid. (thread idx)

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

comptime {
    if (builtin.cpu.arch != .nvptx64) {
        @compileError("vector_add.zig must be compiled for nvptx64-cuda-none");
    }
}
