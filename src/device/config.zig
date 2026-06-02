pub const THREADS: u32 = 16; // Thread per block
pub const EPT: u32 = 8;     // Elements per thread 
pub const TILE: u32 = THREADS * EPT; // element processed per block
