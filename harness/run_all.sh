#!/usr/bin/env bash
# Collect the paper's dataset: 4 shards x 50 keys x 2000 signatures per
# parameter set, ordinary and bounded-weight signing. Twelve processes.
#
#   bash run_all.sh <output directory>
set -euo pipefail

out="${1:?usage: run_all.sh <output directory>}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"

run() { # <mode> <name> <lo> <hi> <shard>
  "$here/hint_collect_$1" 50 2000 "$3" "$4" "$5" \
    > "$out/shard_$2_$5.csv" 2> "$out/shard_$2_$5.err"
}

pids=()
for shard in 0 1 2 3; do
  run 2 44 60 66 "$shard" & pids+=("$!")
  run 3 65 35 41 "$shard" & pids+=("$!")
  run 5 87 54 60 "$shard" & pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
echo "finished, failed shards: $failed"
[ "$failed" -eq 0 ]

for name in 44 65 87; do
  cat "$out"/shard_"$name"_{0,1,2,3}.csv > "$out/mldsa$name.csv"
done
