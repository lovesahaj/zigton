pub const THREADS = @import("config.zig").THREADS;
pub const EPT = @import("config.zig").EPT;
pub const TILE = @import("config.zig").TILE;

pub const blockSync = @import("utils.zig").blockSync;
pub const GlobalPtr = @import("utils.zig").GlobalPtr;
pub const ConstGlobalPtr = @import("utils.zig").ConstGlobalPtr;
pub const requireBlock = @import("utils.zig").requireBlock;

pub const linearIndex = @import("utils.zig").linearIndex;
pub const utils = @import("utils.zig");


pub const RegTile = @import("regtile.zig").RegTile;
pub const load = @import("regtile.zig").load;
pub const store = @import("regtile.zig").store;

pub const sharedTile = @import("shatile.zig").sharedTile;

