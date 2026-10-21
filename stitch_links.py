#!/usr/bin/env python3
"""Turn stitch_tracks.py output into the links file merge_identities_world.py reads.

`stitch_tracks.py` writes a CSV that has already renumbered tracks into stitched
identities. `merge_identities_world.py` and `render_tracked_video.py` both work on the
raw tracker CSV plus a links file, so the stitching result has to be expressed as
pairs of raw track ids rather than as a renumbered table. Each stitched identity
becomes a star of links from its first raw track to the rest; union-find on the other
side rebuilds the same grouping.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    members: dict[int, set[int]] = defaultdict(set)
    first_frame: dict[tuple[int, int], int] = {}
    with args.stitched.open() as handle:
        for row in csv.DictReader(handle):
            ident, orig, frame = int(row["identity"]), int(row["orig_track_id"]), int(row["frame"])
            members[ident].add(orig)
            key = (ident, orig)
            if frame < first_frame.get(key, 1 << 30):
                first_frame[key] = frame

    links = []
    for ident, orig_ids in members.items():
        ordered = sorted(orig_ids, key=lambda o: first_frame[(ident, o)])
        for succ in ordered[1:]:
            links.append({"pred": ordered[0], "succ": succ})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": str(args.stitched),
        "stitched_identities": len(members),
        "raw_tracks": sum(len(v) for v in members.values()),
        "links": links,
    }, indent=1) + "\n")
    print(json.dumps({"stitched_identities": len(members),
                      "raw_tracks": sum(len(v) for v in members.values()),
                      "links": len(links)}, indent=2))


if __name__ == "__main__":
    main()
