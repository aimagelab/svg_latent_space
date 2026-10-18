"""Convert a directory of SVGs to path-level JSONL for SLS training."""

import argparse
import json
from pathlib import Path

from sls.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).rglob("*.svg"))
    if not files:
        parser.error("No SVG files found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as stream:
        for filename in files:
            for path in preprocess(str(filename)).splitlines():
                if path.strip():
                    stream.write(json.dumps({"text": path}) + "\n")
                    count += 1
    print(f"Wrote {count} paths from {len(files)} SVG files to {output}")


if __name__ == "__main__":
    main()
