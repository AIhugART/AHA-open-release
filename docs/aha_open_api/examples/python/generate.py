"""Text to image with the standard library only.

    python generate.py "a red fox in the snow" --out fox.png
    python generate.py "a red fox" --model qwen-image-2.1 --width 1024 --height 1024 --seed 7 --steps 30

Settings: AHA_OPEN_URL (default http://127.0.0.1:8890), AHA_OPEN_TOKEN (default: the local token file). See aha_open_client.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aha_open_client import AhaOpenError, Client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--model")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    params = {k: v for k, v in (("seed", args.seed), ("steps", args.steps)) if v is not None}

    def progress(job: dict) -> None:
        p = job["progress"]
        if job["status"] == "queued" and job.get("queue_position") is not None:
            print(f"\rqueued (position {job['queue_position']})      ", end="", file=sys.stderr)
        elif p["total_steps"]:
            print(f"\r{job['status']}: step {p['step']}/{p['total_steps']}      ", end="", file=sys.stderr)

    try:
        client = Client()
        model = client.pick_model("image.generate", args.model)
        lic = model["license"]
        print(f"model {model['id']} - license: {lic['name']} (commercial use: {lic['commercial_use']})", file=sys.stderr)
        job, png = client.generate(args.prompt, model=model["id"], width=args.width, height=args.height, on_progress=progress, **params)
    except AhaOpenError as err:
        print(f"\nerror: {err}", file=sys.stderr)
        if err.code == "unauthorized":
            print("The server did not accept the token. Set AHA_OPEN_TOKEN to a valid client token "
                  "(see docs/aha_open_api/01_quickstart.md).", file=sys.stderr)
        return 1
    Path(args.out).write_bytes(png)
    r = job["results"][0]
    print(f"\nsaved {args.out}: {r['width']}x{r['height']}, {r['bytes']} bytes, seed {r['seed']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
