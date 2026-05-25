const builtin = @import("builtin");
const zt = @import("zigton.zig");


pub export fn vector_add(
    x: zt.ConstGlobalPtr(f32),
    y: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

pub export fn fill(
    z: zt.GlobalPtr(f32), 
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = value;
    }
}

pub export fn add_scalar(
    x: zt.ConstGlobalPtr(f32),
    z: zt.GlobalPtr(f32),
    value: f32,
    n: u32,
) callconv(.kernel) void {
    const i = zt.linearIndex();

    if (i < n) {
        z[i] = x[i] + value;
    }
}
