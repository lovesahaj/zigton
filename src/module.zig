const cuda = @import("cuda");
const check = @import("utils.zig").check;
const kernel_mod = @import("kernel.zig");

pub const Module = struct {
    handle: cuda.CUmodule,

    pub fn loadData(ptx: []const u8) !Module {
        var handle: cuda.CUmodule = undefined;
        try check(cuda.cuModuleLoadData(&handle, ptx.ptr));

        return .{
            .handle = handle,
        };
    }

    pub fn deinit(self: *Module) void {
        _ = cuda.cuModuleUnload(self.handle);
        self.* = undefined;
    }

    pub fn getFunction(self: *Module, comptime name: [:0]const u8) !cuda.CUfunction {
        var func: cuda.CUfunction = undefined;
        try check(cuda.cuModuleGetFunction(
            &func,
            self.handle,
            name,
        ));
        return func;
    }

    pub fn kernel(self: Module, comptime name: [:0]const u8) !kernel_mod.Kernel {
        var func: cuda.CUfunction = undefined;
        try check(cuda.cuModuleGetFunction(&func, self.handle, name));
        return .{ .func = func };
    }
};
