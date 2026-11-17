#!/usr/bin/env bash
# LLM maintained.
set -euo pipefail

cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
build_dir="$root/tests/build/batch_bench"
result_dir="$root/tests/perf/results"
mkdir -p "$build_dir" "$result_dir"

cc=${CC:-cc}
"$cc" -O2 -DNDEBUG -std=c11 \
    -Wall -Wextra -Werror -Wsign-conversion -Wstrict-prototypes -Wshadow \
    -I"$root" -I"$root/util" \
    "$root/tests/perf/sokol_gfx_batch_bench.c" \
    -o "$build_dir/sokol_gfx_batch_bench" \
    -lEGL -lGL -lm -ldl -lpthread

frames=${BENCH_FRAMES:-120}
warmup=${BENCH_WARMUP:-16}
"$build_dir/sokol_gfx_batch_bench" "$frames" "$warmup" "$result_dir/batch_bench.csv" \
    | tee "$result_dir/batch_bench.txt"
