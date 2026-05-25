const builtin = @import("builtin");
const zt = @import("zigton.zig");


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
