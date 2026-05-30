const cuda = @import("cuda");
const utils = @import("utils.zig");

pub const Dim3 = struct {
    x: c_uint,
    y: c_uint = 1,
    z: c_uint = 1,
};

pub const LaunchConfig = struct {
    grid: Dim3,
    block: Dim3,
    shared_bytes: c_uint = 0,
    stream: cuda.CUstream = null,
};

pub fn launch(
    func: cuda.CUfunction,
    config: LaunchConfig,
    kernel_args: [*]?*anyopaque,
) !void {
    try utils.check(cuda.cuLaunchKernel(
        func,
        config.grid.x,
        config.grid.y,
        config.grid.z,
        config.block.x,
        config.block.y,
        config.block.z,
        config.shared_bytes,
        config.stream,
        kernel_args,
        null,
    ));
}
