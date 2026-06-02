const cuda = @import("cuda");
const launch_mod = @import("launch.zig");

pub const Kernel = struct {
    func: cuda.CUfunction,

    pub fn launch(
        self: Kernel,
        config: launch_mod.LaunchConfig,
        args: anytype,
    ) !void {
        var local = args;
        try launch_mod.launch(self.func, config, local.ptr());
    }
};
