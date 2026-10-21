#!/usr/bin/env python3
"""Fold identity-level duplicate links into an identity map, then renumber.

merge_duplicate_boxes.py --identity-map emits pairs of *identities*, and its one-to-one
no-chain rule already guarantees each identity is absorbed at most once and never both
absorbs and is absorbed. So this is a single relabel pass, not a union-find: feeding
these pairs back through union-find over raw tracks is what previously chained one
identity across 42 animals in a frame.

With ``--tracks`` it also writes the final per-frame count CSV (``active_identities``,
``cumulative_identities``), which is the pipeline's distinct-animal curve.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--dup-links", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, default=None,
                        help="Raw tracks CSV; when given, write <out stem>_counts.csv next to --out")
    args = parser.parse_args()

    idmap = {int(k): int(v) for k, v in json.loads(args.identity_map.read_text()).items()}
    links = json.loads(args.dup_links.read_text())["links"]
    absorb = {int(l["succ"]): int(l["pred"]) for l in links}
    assert not (set(absorb) & set(absorb.values())), "links are not one-to-one"

    merged = {t: absorb.get(i, i) for t, i in idmap.items()}
    renumber = {old: new for new, old in enumerate(sorted(set(merged.values())), 1)}
    final = {str(t): renumber[i] for t, i in merged.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, indent=1) + "\n")
    summary = {"identities_before": len(set(idmap.values())),
               "duplicate_pairs_applied": len(absorb),
               "identities_after": len(renumber)}

    if args.tracks:
        per_frame: dict[int, set[int]] = defaultdict(set)
        with args.tracks.open() as handle:
            for row in csv.DictReader(handle):
                ident = final.get(row["track_id"])
                if ident is not None:
                    per_frame[int(row["frame"])].add(ident)
        counts_path = args.out.with_name(f"{args.out.stem}_counts.csv")
        seen: set[int] = set()
        with counts_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "active_identities", "cumulative_identities"])
            for frame in range(max(per_frame) + 1 if per_frame else 0):
                seen |= per_frame.get(frame, set())
                writer.writerow([frame, len(per_frame.get(frame, set())), len(seen)])
        summary["max_active_identities"] = max((len(v) for v in per_frame.values()), default=0)
        summary["counts"] = str(counts_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
