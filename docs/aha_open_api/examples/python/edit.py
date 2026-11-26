"""Edit an image: a source picture, optional reference pictures, and (only if the model supports it) a mask.

    python edit.py "make the jacket red" --source me.png --out me_red.png
    python edit.py "put this style on it" --source me.png --reference style.png

Settings: AHA_OPEN_URL, AHA_OPEN_TOKEN (see aha_open_client.py). The media type comes from the file extension (png, jpg, webp).
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

from aha_open_client import AhaOpenError, Client


def media_type(path: str) -> str:
    kind = mimetypes.guess_type(path)[0] or ""
    if not kind.startswith("image/"):
        raise SystemExit(f"error: {path} does not look like an image (png, jpg or webp)")
    return kind


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", action="append", default=[])
    ap.add_argument("--mask")
    ap.add_argument("--model")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="edited.png")
    args = ap.parse_args()
    try:
        client = Client()
        model = client.pick_model("image.edit", args.model)
        caps = model["capabilities"]
        if args.mask and not caps.get("mask"):
            print(f"error: {model['id']} does not support a mask (capabilities.mask is false). Paint the area on the source "
                  "image or describe it in the prompt instead.", file=sys.stderr)
            return 1
        if len(args.reference) > caps.get("max_reference_images", 0):
            print(f"error: {model['id']} accepts at most {caps.get('max_reference_images', 0)} reference images", file=sys.stderr)
            return 1
        inputs = [{"role": "source", "upload_id": client.upload(args.source, media_type(args.source))}]
        inputs += [{"role": "reference", "upload_id": client.upload(p, media_type(p))} for p in args.reference]
        if args.mask:
            inputs.append({"role": "mask", "upload_id": client.upload(args.mask, media_type(args.mask))})
        if args.width is None:
            first = caps["sizes"][0]
            width, height = first["width"], first["height"]
        else:
            width, height = args.width, args.height
        params = {"width": width, "height": height, **({"seed": args.seed} if args.seed is not None else {})}
        job = client.create_job({"task": "image.edit", "model": model["id"], "prompt": args.prompt, "inputs": inputs, "params": params})
        job = client.wait(job["id"])
        if job["status"] != "succeeded":
            err = job.get("error") or {}
            print(f"error: {err.get('code', job['status'])}: {err.get('message', '')}", file=sys.stderr)
            return 1
        Path(args.out).write_bytes(client.result(job["id"]))
    except AhaOpenError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    r = job["results"][0]
    print(f"saved {args.out}: {r['width']}x{r['height']}, seed {r['seed']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
