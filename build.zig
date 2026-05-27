const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda_prefix = b.option([]const u8, "cuda-prefix", "CUDA install prefix") orelse "/afs/inf.ed.ac.uk/user/s28/s2881386/.local/cuda-13.0";
    // better would be `zig build run -Dcuda-prefix=path_to_cuda`

    // Path to an `llc` from LLVM >= 21 (must understand `captures(none)` IR).
    // Defaults to `llc` on PATH; override with -Dllc-path=/path/to/llc.
    const llc_path = b.option([]const u8, "llc-path", "Path to llc (LLVM >= 21)") orelse "llc";

    // GPU architecture to target, e.g. sm_89 for an RTX 4060.
    const gpu_arch = b.option([]const u8, "gpu-arch", "Target GPU SM arch") orelse "sm_89";

    // ------------------------------------------------------------------
    // GPU kernel pipeline:  gpu.zig -> LLVM IR -> (rewrite) -> PTX
    // ------------------------------------------------------------------
    const ptx = buildPtx(b, .{
        .kernel_source = b.path("kernels/gpu.zig"),
        .gpu_arch = gpu_arch,
        .llc_path = llc_path,
        .optimize = optimize,
    });

    // ------------------------------------------------------------------
    // CUDA header translation (host side) — unchanged from before.
    // ------------------------------------------------------------------
    const c_translation = b.addTranslateC(.{
        .root_source_file = b.path("src/cuda_includes.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    c_translation.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_prefix}) });

    const cuda_module = b.createModule(.{
        .root_source_file = c_translation.getOutput(),
    });

    const mod = b.addModule("zigton", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cuda", .module = cuda_module },
        },
    });

    const exe = b.addExecutable(.{
        .name = "zigton",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigton", .module = mod },
                .{ .name = "cuda", .module = cuda_module },
            },
        }),
    });

    // Make the generated PTX embeddable from the host source as
    //     const ptx = @embedFile("gpu_ptx");
    // This replaces the fragile "copy the .ptx into src/" dance: the PTX is now
    // a tracked build artifact and the exe depends on it through the graph.
    exe.root_module.addAnonymousImport("gpu_ptx", .{
        .root_source_file = ptx,
    });

    exe.root_module.linkSystemLibrary("cuda", .{});
    exe.root_module.link_libc = true;
    exe.root_module.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_prefix}) });

    b.installArtifact(exe);

    // Convenience: `zig build ptx` to produce just the PTX without building the
    // host exe — handy while iterating on the kernel.
    const ptx_step = b.step("ptx", "Build the GPU kernel PTX only");
    const install_ptx = b.addInstallFile(ptx, "gpu.ptx");
    ptx_step.dependOn(&install_ptx.step);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}

const PtxOptions = struct {
    kernel_source: std.Build.LazyPath,
    gpu_arch: []const u8,
    llc_path: []const u8,
    optimize: std.builtin.OptimizeMode,
};

// Compile a Zig kernel source to PTX and return the PTX as a LazyPath that
// other steps can consume. Three stages:
//   1. `zig build-obj -femit-llvm-ir` for the nvptx64-cuda target -> raw .ll
//   2. tools/fix_ptx_ir.sh to collapse the NVPTX kernel alias            -> fixed .ll
//   3. llc -mcpu=<arch> to lower the fixed IR                            -> .ptx
fn buildPtx(b: *std.Build, opts: PtxOptions) std.Build.LazyPath {
    const nvptx_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
    });

    const device_mod = b.createModule(.{
        .root_source_file = b.path("src/device.zig"),
        .target = nvptx_target,
        .optimize = .ReleaseSmall,
    });

    // The kernel is its own module/object, cross-compiled to nvptx64.
    const kernel_obj = b.addObject(.{
        .name = "gpu",
        .root_module = b.createModule(.{
            .root_source_file = opts.kernel_source,
            .target = nvptx_target,
            // ReleaseSmall keeps the kernel lean; the alias problem lives in
            // emission, not optimization, so the opt level is free to choose.
            .optimize = .ReleaseSmall,
            .imports = &.{
                .{ .name = "zigton_device", .module = device_mod },
            },
        }),
    });
    // The device target has no UBSan runtime to link against.
    kernel_obj.bundle_ubsan_rt = false;

    // Stage 1: emitted LLVM IR (the .ll). getEmittedLlvmIr() gives us the
    // LazyPath to the IR file the object step produces.
    const raw_ll = kernel_obj.getEmittedLlvmIr();

    // Stage 2: rewrite the IR to collapse the kernel alias.
    const fix = b.addSystemCommand(&.{"bash"});
    fix.addFileArg(b.path("tools/fix_ptx_ir.sh"));
    fix.addFileArg(raw_ll);
    const fixed_ll = fix.addOutputFileArg("gpu.fixed.ll");

    // Stage 3: lower the fixed IR to PTX with a modern llc.
    const llc = b.addSystemCommand(&.{opts.llc_path});
    llc.addArg(b.fmt("-mcpu={s}", .{opts.gpu_arch}));
    llc.addFileArg(fixed_ll);
    llc.addArg("-o");
    const ptx = llc.addOutputFileArg("gpu.ptx");

    return ptx;
}
