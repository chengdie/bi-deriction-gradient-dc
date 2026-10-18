"""Build a reproducible subpool from the same processed instruction sources as LESS.

Never downloads or silently substitutes data. The manifest maps EVERY output row
back to the original source row and hashes all inputs and the output.
"""

import argparse
from pathlib import Path
import random

from less.train.online_proxy_io import inspect_inputs, sha256_file, write_json


def prepare(files, output, size=40000, seed=2026, allow_small=False):
    sources = inspect_inputs(files)
    if not allow_small and any(source["rows"] <= 10 for source in sources):
        raise ValueError("Placeholder sources detected. Supply the complete processed LESS data.")
    total = sum(source["rows"] for source in sources)
    if not 0 < size <= total:
        raise ValueError(f"Pool size {size} is outside [1, {total}]")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be empty: {output}")
    chosen = set(random.Random(seed).sample(range(total), size))
    output.mkdir(parents=True, exist_ok=True)
    mapping, index = [], 0
    with (output / "pool.jsonl").open("x", encoding="utf-8") as destination:
        for source_index, source in enumerate(sources):
            with Path(source["path"]).open(encoding="utf-8") as stream:
                for row, line in enumerate(stream):
                    if not line.strip():
                        continue
                    if index in chosen:
                        destination.write(line.rstrip("\r\n") + "\n")
                        mapping.append(dict(pool_index=len(mapping), source_index=source_index,
                                            source_row=row, global_source_index=index))
                    index += 1
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("Source changed while building the pool")
    manifest = dict(completed=True, seed=seed, source_count=total, pool_size=size, sources=sources,
                    rows=mapping, order="original source order after uniform sampling without replacement",
                    pool_sha256=sha256_file(output / "pool.jsonl"))
    write_json(output / "pool_manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--train-files", nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--size", type=int, default=40000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--allow-small", action="store_true", help="Only for synthetic diagnostics")
    args = p.parse_args()
    files = args.train_files or [args.data_dir / "train" / "processed" / name / f"{name}_data.jsonl"
                                for name in ("flan_v2", "cot", "dolly", "oasst1")]
    manifest = prepare(files, args.output_dir, args.size, args.seed, args.allow_small)
    print(f"Saved {manifest['pool_size']} rows; sha256={manifest['pool_sha256']}")


if __name__ == "__main__":
    main()
