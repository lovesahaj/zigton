// --- context.zig
const std = @import("std");
const cuda = @import("cuda");

pub fn check(result: cuda.CUresult) !void {
    if (result != cuda.CUDA_SUCCESS) {
        std.debug.print("CUDA Error encountered: {}\n", .{result});
        return error.CudaError;
    }
}

pub const Error = error{
    CudaError,
};

pub const Options = struct {
    device_index: c_int = 0,
};

pub const Context = struct {
    device: cuda.CUdevice,
    handle: cuda.CUcontext,

    pub fn init(opts: Options) !Context {
        try check(cuda.cuInit(0));

        var device: cuda.CUdevice = undefined;
        try check(cuda.cuDeviceGet(&device, opts.device_index));

        var handle: cuda.CUcontext = undefined;
        try check(cuda.cuCtxCreate_v4(&handle, null, 0, device));

        return .{
            .device = device,
            .handle = handle,
        };
    }

    pub fn deinit(self: *Context) void {
        _ = cuda.cuCtxDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn sync(self: *Context) !void {
        try check(cuda.cuCtxSynchronize_v2(self.handle));
    }
};
