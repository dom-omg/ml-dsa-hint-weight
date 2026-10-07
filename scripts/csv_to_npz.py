"""
Convert the output of harness/hint_collect_<mode> to a compact .npz file.

Usage: python3 scripts/csv_to_npz.py <harness output> <results/c_ref/mldsaXX.npz>
"""

import os
import sys

import numpy as np


def convert(src: str, dst: str) -> None:
    norms, std, bw, runs = {}, {}, {}, {}
    with open(src) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                raise ValueError("short record: " + line)
            tag, key = parts[0], int(parts[1])
            if tag == "N":
                norms[key] = [float(v) for v in parts[2:]]
            elif tag == "S":
                std.setdefault(key, []).append([int(v) for v in parts[2:]])
            elif tag == "B":
                bw.setdefault(key, []).append([int(v) for v in parts[2:]])
            elif tag == "C":
                runs[key] = int(parts[2])
            else:
                raise ValueError("unknown record: " + line)
    keys = sorted(norms)
    if not keys or keys != sorted(std) or keys != sorted(bw) or keys != sorted(runs):
        raise ValueError("incomplete run: the four record types do not cover the same keys")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".tmp.npz"
    np.savez_compressed(
        tmp,
        norms=np.array([norms[k] for k in keys], dtype=np.float64),
        std=np.array([std[k] for k in keys], dtype=np.uint8),
        bw=np.array([bw[k] for k in keys], dtype=np.uint8),
        runs=np.array([runs[k] for k in keys], dtype=np.int64),
    )
    os.replace(tmp, dst)
    print(dst, "keys:", len(keys), "signatures per key:", len(std[keys[0]]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2])
