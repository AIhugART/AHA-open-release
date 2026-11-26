"""Recipes beyond plain text to image. Each one asks the manifest first and says in a sentence why it cannot go on.

    python recipes.py transparent "a red apple" --out apple.png      # a PNG with a see-through background
    python recipes.py loras                                          # which LoRA files this server offers
    python recipes.py lora "a portrait of a fox" --lora ink --scale 0.8 --out fox.png
    python recipes.py video "waves on a beach" --duration 4 --out waves.mp4      # a model with free durations
    python recipes.py video "a lighthouse at sunset" --tier preview --out l.mp4  # MiniMax H3: 5 s, preview (608x352) or native (1344x768)
    python recipes.py cancel "a very detailed city" --after 3        # start a job, change your mind, cancel it

Settings: AHA_OPEN_URL, AHA_OPEN_TOKEN (see aha_open_client.py).
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
import time
from pathlib import Path

from aha_open_client import AhaOpenError, Client


def first_size(model: dict) -> dict:
    return model["capabilities"]["sizes"][0]  # an exact size the model declares; never invent one


def fail(text: str) -> int:
    print(f"error: {text}", file=sys.stderr)
    return 1


def finish(client: Client, job: dict, out: str, timeout: float = 900.0) -> int:
    job = client.wait(job["id"], timeout=timeout)
    if job["status"] != "succeeded":
        err = job.get("error") or {}
        return fail(f"the job {job['status']}: {err.get('code', '')}: {err.get('message', '')}")
    Path(out).write_bytes(client.result(job["id"]))
    r = job["results"][0]
    if r["kind"] == "image":
        print(f"saved {out}: {r['width']}x{r['height']}, seed {r['seed']}", file=sys.stderr)
    else:
        size = f"{r['width']}x{r['height']}, " if r.get("width") else ""
        print(f"saved {out}: {size}{r.get('duration_s')} s, {r.get('fps')} fps, audio: {r.get('has_audio')}", file=sys.stderr)
    return 0


def transparent(client: Client, args) -> int:
    models = [m for m in client.manifest()["models"] if "image.generate" in m["tasks"] and m["state"] == "ready"]
    capable = [m for m in models if m["capabilities"].get("rgba")]
    if not capable:
        return fail("no ready model can produce transparency (capabilities.rgba is false for: "
                    f"{', '.join(m['id'] for m in models) or 'no ready image model'})")
    model = capable[0]
    size = first_size(model)
    job = client.create_job({"task": "image.generate", "model": model["id"], "prompt": args.prompt,
                             "params": {"width": size["width"], "height": size["height"], "transparent": True}})
    return finish(client, job, args.out)


def list_loras(client: Client, args) -> int:
    found = False
    for model in client.manifest()["models"]:
        caps = (model["capabilities"] or {}).get("loras")
        if not caps:
            continue
        for lora in caps["available"]:
            found = True
            print(f"{model['id']}: {lora['id']}  (default scale {lora.get('default_scale', caps['scale']['default'])}, "
                  f"trigger word: {lora.get('trigger') or 'none'})")
        print(f"{model['id']}: up to {caps['max']} LoRAs per job, scale {caps['scale']['min']} to {caps['scale']['max']}", file=sys.stderr)
    if not found:
        print("this server offers no LoRA files (the operator adds them: `aha-open loras add`)", file=sys.stderr)
    return 0


def lora(client: Client, args) -> int:
    for model in client.manifest()["models"]:
        caps = (model["capabilities"] or {}).get("loras")
        if model["state"] == "ready" and "image.generate" in model["tasks"] and caps:
            ids = [item["id"] for item in caps["available"]]
            if args.lora not in ids:
                return fail(f"{model['id']} does not offer a LoRA called {args.lora!r}; it offers: {', '.join(ids) or 'none'}")
            size = first_size(model)
            job = client.create_job({"task": "image.generate", "model": model["id"], "prompt": args.prompt, "params": {
                "width": size["width"], "height": size["height"], "seed": args.seed, "loras": [{"id": args.lora, "scale": args.scale}]}})
            return finish(client, job, args.out)
    return fail("no ready image model accepts LoRAs")


def video(client: Client, args) -> int:
    models = [m for m in client.manifest()["models"] if "video.generate" in m["tasks"] and m["state"] == "ready"]
    wanted_refs = list(args.reference or [])
    if wanted_refs:  # references: a model that takes them; without references: one that does not need them
        models = [m for m in models if (m["capabilities"].get("max_reference") or {}).get("total", 0) >= len(wanted_refs)]
    else:
        models = [m for m in models if not m["capabilities"].get("reference_required")]
    if not models:
        return fail("no ready video model takes " + (f"{len(wanted_refs)} reference(s)" if wanted_refs else "a text-only request")
                    + " (see GET /v1/manifest: capabilities.max_reference and reference_required)")
    model = models[0]
    caps = model["capabilities"]
    low, high = caps["durations_s"]["min"], caps["durations_s"]["max"]
    duration = low if args.duration is None else args.duration  # the model's own shortest clip unless you ask for another length
    if not low <= duration <= high:
        wanted = f"exactly {low}" if low == high else f"{low} to {high}"
        return fail(f"{model['id']} makes videos of {wanted} seconds; you asked for {duration}")
    params = {"aspect_ratio": caps["aspect_ratios"][0], "duration_s": duration, "short_edge": caps["short_edges"][0]}
    sizes = caps.get("sizes")  # a model that declares exact sizes (MiniMax H3): pick one of them by name, never invent a size
    if sizes:
        tiers = {s["tier"]: s for s in sizes}
        offered = ", ".join(f"{name} ({s['width']}x{s['height']})" for name, s in tiers.items())
        tier = args.tier or min(sizes, key=lambda s: s["width"] * s["height"])["tier"]  # smallest by default: a first try takes minutes
        if tier not in tiers:
            return fail(f"{model['id']} offers these sizes: {offered}; you asked for {tier!r}")
        params.update(aspect_ratio=tiers[tier]["aspect_ratio"], short_edge=tiers[tier]["short_edge"])
        print(f"size: {tier} ({tiers[tier]['width']}x{tiers[tier]['height']}); the sizes on offer are {offered}", file=sys.stderr)
    licence = model.get("license") or {}
    if licence.get("required_ui_label"):  # a licence duty, so a real client shows it where the model is chosen
        print(f"licence: show the name {licence['required_ui_label']!r} in your app"
              + (f"; not licensed in {', '.join(licence['excluded_territories'])}" if licence.get("excluded_territories") else ""), file=sys.stderr)
    inputs = []
    for path in wanted_refs:  # the order given is the order the model reads them in
        kind = (mimetypes.guess_type(path)[0] or "image/png")
        role = "reference_video" if kind.startswith("video/") else "reference_audio" if kind.startswith("audio/") else "reference"
        inputs.append({"role": role, "upload_id": client.upload(Path(path), kind)})
    job = client.create_job({"task": "video.generate", "model": model["id"], "prompt": args.prompt, "params": params, "inputs": inputs})
    print(f"job {job['id']} created; a video takes minutes, waiting up to {args.timeout / 60:.0f} (the job keeps running if this stops)", file=sys.stderr)
    return finish(client, job, args.out, timeout=args.timeout)


def cancel(client: Client, args) -> int:
    model = client.pick_model("image.generate")
    size = first_size(model)
    job = client.create_job({"task": "image.generate", "model": model["id"], "prompt": args.prompt,
                             "params": {"width": size["width"], "height": size["height"]}})
    print(f"job {job['id']} started; cancelling in {args.after} s", file=sys.stderr)
    time.sleep(args.after)
    job = client.cancel(job["id"])
    if job["status"] not in ("cancelled", "succeeded", "failed"):
        job = client.wait(job["id"], timeout=60)  # the server answered before the job stopped: keep polling until it is final
    print(f"final status: {job['status']}; results: {len(job['results'])}", file=sys.stderr)
    try:
        client.result(job["id"])
    except AhaOpenError as err:
        print(f"asking for the result now gives: {err.code}", file=sys.stderr)  # job_not_ready: a cancelled job never has a result
    return 0 if job["status"] == "cancelled" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="recipe", required=True)
    t = sub.add_parser("transparent")
    t.add_argument("prompt")
    t.add_argument("--out", default="transparent.png")
    sub.add_parser("loras")
    li = sub.add_parser("lora")
    li.add_argument("prompt")
    li.add_argument("--lora", required=True)
    li.add_argument("--scale", type=float, default=1.0)
    li.add_argument("--seed", type=int, default=1)
    li.add_argument("--out", default="lora.png")
    v = sub.add_parser("video")
    v.add_argument("prompt")
    v.add_argument("--duration", type=float, default=None, help="seconds; default: the model's shortest clip")
    v.add_argument("--tier", help="a size the model declares (for MiniMax H3: preview or native); default: the smallest")
    v.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait for the clip (MiniMax H3 at native size: ~25 min)")
    v.add_argument("--reference", action="append", help="a reference image/video/audio file; repeat for several (their order matters)")
    v.add_argument("--out", default="out.mp4")
    c = sub.add_parser("cancel")
    c.add_argument("prompt")
    c.add_argument("--after", type=float, default=3.0)
    args = ap.parse_args()
    handlers = {"transparent": transparent, "loras": list_loras, "lora": lora, "video": video, "cancel": cancel}
    try:
        return handlers[args.recipe](Client(), args)
    except AhaOpenError as err:
        return fail(str(err))


if __name__ == "__main__":
    sys.exit(main())
