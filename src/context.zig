// --- context.zig
const std = @import("std");
const cuda = @import("cuda");
const utils = @import("utils.zig");

pub const Options = struct {
    device_index: c_int = 0,
};

pub const Context = struct {
    device: cuda.CUdevice,
    handle: cuda.CUcontext,

    pub fn init(opts: Options) utils.CudaError!Context {
        try utils.check(cuda.cuInit(0));

        var device: cuda.CUdevice = undefined;
        try utils.check(cuda.cuDeviceGet(
            &device,
            opts.device_index,
        ));

        var handle: cuda.CUcontext = undefined;
        try utils.check(cuda.cuCtxCreate_v4(
            &handle,
            null,
            0,
            device,
        ));

        return .{
            .device = device,
            .handle = handle,
        };
    }

    pub fn deinit(self: *Context) void {
        _ = cuda.cuCtxDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn sync(self: *Context) utils.CudaError!void {
        const result = cuda.cuCtxSynchronize_v2(self.handle);

        utils.check(result) catch |err| {
            std.debug.print("cuCtxSynchronize failed: {s}\n", .{utils.errorString(result)});
            return err;
        };
    }
};
