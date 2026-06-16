const Kernel = @import("kernel.zig").Kernel;
const kernelArgs = @import("args.zig").kernelArgs;
const Module = @import("module.zig").Module;
const Context = @import("context.zig").Context;
const DeviceBuffer = @import("buffer.zig").DeviceBuffer;
const cdiv = @import("utils.zig").cdiv;

pub const Matmul = struct {
    tiled_2d: Kernel,

    pub fn init(module: Module) !Matmul {
        return .{
            .tiled_2d = try module.kernel("matmul_tiled_2d"),
        };
    }

    pub fn runF32(
        self: @This(),
        ctx: *Context,
        a: DeviceBuffer(f32),
        b: DeviceBuffer(f32),
        c: DeviceBuffer(f32),
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        const BM = 64;
        const BN = 64;
        const TM = 4;
        const TN = 4;

        const args = kernelArgs(.{ a.ptr, b.ptr, c.ptr, m, k, n });

        try self.tiled_2d.launch(.{
            .grid = .{
                .x = try cdiv(n, BN),
                .y = try cdiv(m, BM),
            },
            .block = .{
                .x = BN / TN,
                .y = BM / TM,
            },
        }, args);

        try ctx.sync();
    }
};
