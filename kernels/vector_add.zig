const builtin = @import("builtin");

pub fn vector_add(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]f32,
    n: u32,
) callconv(.kernel) void {
    const i = blockIdxX() * blockDimX() + threadIdxX();

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

fn threadIdxX() u32 { 
    // Geting the GPU Thread 
    return asm volatile ("mov.u32 $0, %tid.x;"
        : [ret] "=r" (-> u32),
    );
}
fn blockIdxX() u32 {
    // Geting the block ID
    return asm volatile ("mov.u32 $0, %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
}
fn blockDimX() u32 {
    // How many thread are there per block
    return asm volatile ("mov.u32 $0, %ntid.x;"
        : [ret] "=r" (-> u32),
    );
}

comptime {
    if (builtin.cpu.arch != .nvptx64) {
        @compileError("vector_add.zig must be compiled for nvptx64-cuda-none");
    }

    @export(&vector_add, .{ .name = "vector_add", .linkage = .strong });
}
