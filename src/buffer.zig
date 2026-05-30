const cuda = @import("cuda");
const context = @import("context.zig");

pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        ptr: cuda.CUdeviceptr_v2,
        len: usize,

        const Self = @This();

        pub fn init(len: usize) !Self {
            var ptr: cuda.CUdeviceptr_v2 = undefined;

            try context.check(cuda.cuMemAlloc_v2(
                &ptr,
                len * @sizeOf(T),
            ));

            return .{
                .ptr = ptr,
                .len = len,
            };
        }

        pub fn deinit(self: *Self) void {
            _ = cuda.cuMemFree_v2(self.ptr);
            self.* = undefined;
        }

        pub fn copyFromHost(
            self: Self,
            src: []const T,
        ) !void {
            if (src.len > self.len) return error.DeviceBufferTooSmall;
            try context.check(cuda.cuMemcpyHtoD_v2(
                self.ptr,
                src.ptr,
                src.len * @sizeOf(T),
            ));
        }

        pub fn copyToHost(
            self: Self,
            dst: []T,
        ) !void {
            if (dst.len > self.len) return error.HostBufferTooSmall;
            try context.check(cuda.cuMemcpyDtoH_v2(
                dst.ptr,
                self.ptr,
                self.len * @sizeOf(T),
            ));
        }
    };
}
