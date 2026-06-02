pub const THREADS: u32 = 64; // Thread per block
pub const EPT: u32 = 8; // Elements per thread
pub const TILE: u32 = THREADS * EPT; // element processed per block

comptime {
    if (THREADS == 0 or (THREADS & (THREADS - 1)) != 0)
        @compileError("THREADS must be power of two");
}
