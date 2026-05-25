#!/usr/bin/env bash
#
# fix_ptx_ir.sh — collapse Zig's NVPTX kernel alias into a real definition.
#
# Zig emits a kernel as a `private` definition under a module-qualified name
# (e.g. `@gpu.vector_add`) plus a public `alias` (`@vector_add`) pointing at it.
# The NVPTX backend rejects alias-to-kernel:
#
#     LLVM ERROR: NVPTX aliasee must be a non-kernel function definition
#
# This script rewrites the IR so the public symbol IS the kernel definition:
#   1. delete the `= alias` line
#   2. strip `private` from the ptx_kernel definition (make it externally visible)
#   3. rename the module-qualified symbol to its bare name
#
# Usage: fix_ptx_ir.sh <input.ll> <output.ll>
#
# The bare kernel name is derived automatically: the qualified name is the part
# after the last '.' in the `define ... ptx_kernel ... @<qualified>` line, so
# `@gpu.vector_add` -> `vector_add`. This survives file renames and multiple
# kernels without hardcoding "gpu".

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <input.ll> <output.ll>" >&2
    exit 2
fi

in="$1"
out="$2"

if [[ ! -f "$in" ]]; then
    echo "error: input IR not found: $in" >&2
    exit 1
fi

# Find every qualified ptx_kernel symbol, e.g. "gpu.vector_add".
# Match the define line, pull out the @"..."-or-@bareword symbol token.
mapfile -t qualified < <(
    grep -oE 'define [^@]*ptx_kernel[^@]*@("[^"]+"|[A-Za-z0-9_.$]+)' "$in" \
        | grep -oE '@("[^"]+"|[A-Za-z0-9_.$]+)$' \
        | sed -E 's/^@//; s/^"//; s/"$//'
)

if [[ ${#qualified[@]} -eq 0 ]]; then
    echo "error: no ptx_kernel definition found in $in" >&2
    exit 1
fi

cp "$in" "$out"

# 1. Remove all alias lines (the public-symbol-to-kernel indirection).
sed -i -E '/^@[A-Za-z0-9_.$"]+ = .*alias /d' "$out"

# 2. Make ptx_kernel definitions externally visible.
sed -i -E 's/define private ptx_kernel/define ptx_kernel/g' "$out"

# 3. For each qualified kernel name, rename to its bare (post-last-dot) form
#    everywhere it appears (both the @"qualified" and @qualified spellings).
for q in "${qualified[@]}"; do
    bare="${q##*.}"   # strip everything up to and including the last '.'
    if [[ "$bare" == "$q" ]]; then
        # No dot: already bare, nothing to rename.
        continue
    fi
    # Escape regex-significant chars in the qualified name ('.' and '$').
    q_esc="$(printf '%s' "$q" | sed -E 's/[.$]/\\&/g')"
    # Replace both @"gpu.vector_add" and @gpu.vector_add with @vector_add.
    sed -i -E "s/@\"${q_esc}\"/@${bare}/g; s/@${q_esc}/@${bare}/g" "$out"
    echo "fix_ptx_ir: $q -> $bare" >&2
done

echo "fix_ptx_ir: wrote $out" >&2
