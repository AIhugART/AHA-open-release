"""`aha-open` command line (docs/product/04_architecture.md section 6)."""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__
from . import loras as lora_store
from .auth import Auth
from .catalog import CATALOG, describe_gpu, detect_gpus, model_state, variant_installed
from .config import Config, ConfigError, ModelConfig, default_data_dir, load, render, validate_for_serve
from .hardware import fit_for, gpu_budget_mib
from .store import Store
from .util import iso


def _load(args: argparse.Namespace) -> Config:
    try:
        return load(Path(args.data_dir) if args.data_dir else None)
    except ConfigError as err:
        sys.exit(f"error: {err}")


def cmd_init(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    path = data_dir / "config.toml"
    if path.exists() and not args.force:
        print(f"{path} already exists; keeping it (use --force to overwrite).")
        return 0
    cfg = Config(data_dir=data_dir, profile=args.profile)
    if args.profile == "lan":
        cfg.listen = "127.0.0.1:8890"  # the operator must name the real interface; never a wildcard
    for spec in args.model or []:
        model_id, _, variant = spec.partition(":")
        if model_id not in CATALOG:
            sys.exit(f"error: unknown model {model_id!r} (known: {', '.join(i for i, s in CATALOG.items() if not s.internal)})")
        cfg.models.append(ModelConfig(id=model_id, worker=CATALOG[model_id].family, variant=variant))
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cfg), encoding="utf-8")
    print(f"Wrote {path}")
    print("Next: `aha-open models install qwen-image-2.1 --accept-license qwen-research`, then `aha-open serve`.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    cfg = _load(args)
    try:
        validate_for_serve(cfg)
    except ConfigError as err:
        sys.exit(f"error: {err}")
    print(f"AHA open {__version__} listening on {cfg.listen} (profile {cfg.profile})")
    serve(cfg)
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    """Open the bundled web page. On the local profile the token goes in the URL fragment: the browser never sends a
    fragment to the server and it is not logged, so there is nothing to paste."""
    import webbrowser

    cfg = _load(args)
    host = cfg.listen_host or "127.0.0.1"
    scheme = "https" if cfg.profile == "lan" else "http"
    url = f"{scheme}://{host}:{cfg.listen_port}/ui/"
    if cfg.profile == "local" and cfg.local_token_path.exists():
        url += "#token=" + cfg.local_token_path.read_text(encoding="utf-8").strip()
    elif cfg.profile != "local":
        print("Paste a client token on the page (create one with `aha-open clients create --name ...`).")
    print(f"Opening {url.split('#')[0]}")
    if not args.print_only:
        webbrowser.open(url)
    return 0


def _tell(text: str, gui: bool, ok: bool = True) -> None:
    """Print; with --gui (the Start-menu shortcuts run without a console) also show a box, because a failure nobody sees is no answer."""
    print(text)
    if gui and sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "AHA open", 0x40 if ok else 0x10)


def cmd_start(args: argparse.Namespace) -> int:
    from . import launcher

    cfg = _load(args)
    ok, text = launcher.start(cfg)
    if not ok:
        _tell(text, args.gui, ok=False)
        return 1
    print(text)
    if args.no_open:
        return 0
    return cmd_open(argparse.Namespace(data_dir=args.data_dir, print_only=False))


def cmd_stop(args: argparse.Namespace) -> int:
    from . import launcher

    ok, text = launcher.stop(_load(args))
    if args.quiet:
        return 0 if ok else 1
    _tell(text, args.gui, ok=ok)
    return 0 if ok else 1


def cmd_models(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.action == "list":
        print(f"{'MODEL':<22}{'VARIANT':<14}{'STATE':<22}LICENSE")
        for mc in cfg.models:
            spec = CATALOG.get(mc.id)
            if spec is None:
                print(f"{mc.id:<22}{'-':<14}{'unknown model':<22}")
                continue
            state, detail = model_state(cfg, mc, spec)
            print(f"{mc.id:<22}{mc.variant or '-':<14}{state + (' (' + detail + ')' if detail else ''):<22}{spec.license['name']} / commercial: {spec.license['commercial_use']}")
            if spec.variants:  # every variant the model offers: which are on disk, which one is in use
                gpus = detect_gpus()
                for name, info in spec.variants.items():
                    mark = "in use" if name == mc.variant else ("installed" if variant_installed(cfg, spec, name) else "not installed")
                    fit = fit_for(info, gpu_budget_mib(cfg, gpus[0] if gpus else None))
                    print(f"  - {name:<14}{mark:<14}{info['disk_gib']:>3} GB disk, GPU fit: {fit}")
        return 0
    from .install import install_model, remove_variant, use_variant

    if args.action == "use":
        done, message = use_variant(cfg, args.model, args.variant)
        print(message if done else f"error: {message}")
        return 0 if done else 2
    if args.action == "remove":
        done, message = remove_variant(cfg, args.model, args.variant, include_shared=args.include_shared, dry_run=not args.yes)
        if done and not args.yes:
            print(f"{message}\nNothing was deleted. Run again with --yes to delete.")
            return 0
        print(message if done else f"error: {message}")
        return 0 if done else 2

    return install_model(cfg, args)


def cmd_loras(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.action == "list":
        found, problems = lora_store.scan(cfg)
        print(f"{'ID':<28}{'SIZE':>10}  {'SCALE':<6}NAME")
        for lora in found.values():
            print(f"{lora.id:<28}{lora.bytes / 2**20:>7.0f} MB  {lora.default_scale:<6}{lora.display_name}")
        if not found:
            print(f"(none yet - put a .safetensors LoRA in {lora_store.lora_dir(cfg)} or run `aha-open loras add FILE`)")
        for problem in problems:
            print(f"skipped: {problem}")
        return 0
    try:
        if args.action == "add":
            lora = lora_store.add(
                cfg, Path(args.file), args.id, display_name=args.name, default_scale=args.scale, trigger=args.trigger,
                license=args.license, notes=args.notes,
            )
            print(f"added LoRA {lora.id} ({lora.bytes / 2**20:.0f} MB). Clients see it in the manifest immediately; no restart needed.")
        else:
            print("removed" if lora_store.remove(cfg, args.id) else "no LoRA with that id")
    except lora_store.LoraError as err:
        print(f"error: {err}")
        return 2
    return 0


def cmd_clients(args: argparse.Namespace) -> int:
    cfg = _load(args)
    store = Store(cfg.data_dir)
    auth = Auth(cfg, store)
    try:
        if args.action == "create":
            client_id, token = auth.create_client(args.name, args.kind or "")
            print(f"client {client_id} ({args.name})\ntoken (shown once, keep it safe):\n{token}")
        elif args.action == "list":
            for c in store.list_clients():
                state = "revoked " + iso(c["revoked_at"]) if c["revoked_at"] else "active"
                print(f"{c['id']}  {c['name']:<24}{c['kind']:<12}{state}")
        elif args.action == "revoke":
            n = store.revoke_client(args.target, __import__("time").time())
            print(f"revoked {n} client(s)" if n else "no active client with that id or name")
            return 0 if n else 1
        elif args.action == "approve":
            made = auth.approve_enroll(args.target)
            print(f"approved {made['name']} ({made['client_id']}); the client receives its token on its next poll")
        elif args.action == "deny":
            print("denied" if auth.deny_enroll(args.target) else "no pending request with that code")
    finally:
        store.close()
    return 0


def cmd_tls(args: argparse.Namespace) -> int:
    from .tlsutil import fingerprint, init_tls

    cfg = _load(args)
    cert = init_tls(cfg, force=args.action == "init" and args.force)
    print(f"certificate: {cert}\nSHA-256 fingerprint: {fingerprint(cert)}")
    return 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # a redirect to a login page is an answer, not something to follow
        return None


def _get(url: str, headers: dict, cafile: str | None, timeout: float = 15.0) -> tuple[int, dict, bytes]:
    context = ssl.create_default_context(cafile=cafile) if url.startswith("https") else None
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=context))
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read(1_000_000)
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read(1_000_000)


def _explain_connection_error(err: Exception, url: str) -> str:
    """One sentence per distinct cause, so the operator knows which of the three things to fix."""
    reason = getattr(err, "reason", err)
    host = urllib.parse.urlparse(url).netloc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "the server's certificate is not trusted here; for a lan server pass --cacert <data_dir>/tls/server.crt (and compare `aha-open tls fingerprint`)"
    if isinstance(reason, ssl.SSLError):
        return "the TLS handshake failed; is this an https address of the right server?"
    if isinstance(reason, ConnectionRefusedError):
        return f"nothing is listening at {host}: start it with `aha-open serve`, or check the address and port"
    if isinstance(reason, socket.gaierror):
        return f"the name {host!r} does not resolve: check the spelling and your DNS"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return f"{host} did not answer in time: wrong address, firewall, or the machine is off"
    return f"could not connect: {reason}"


def cmd_check(args: argparse.Namespace) -> int:
    """Client-side connectivity check for any profile. Prints which layer failed and what to do about it."""
    base = args.url.rstrip("/")
    problems = 0

    def line(ok: bool, text: str) -> None:
        nonlocal problems
        problems += 0 if ok else 1
        print(("ok    " if ok else "FAIL  ") + text)

    extra = {}
    if args.cf_id or args.cf_secret:
        extra = {"CF-Access-Client-Id": args.cf_id or "", "CF-Access-Client-Secret": args.cf_secret or ""}
    try:
        status, headers, body = _get(base + "/v1/health", extra, args.cacert)
    except (urllib.error.URLError, OSError) as err:
        line(False, _explain_connection_error(err, base))
        return 1
    via_cloudflare = any(k.lower().startswith("cf-") or k.lower() == "server" and "cloudflare" in v.lower() for k, v in headers.items())
    if status in (301, 302, 303, 307, 308) or (status in (401, 403) and via_cloudflare):
        line(False, "Cloudflare Access stopped the request before it reached AHA open: check CF-Access-Client-Id/Secret (--cf-id, --cf-secret) "
                    "and that the Access policy is Service Auth for this hostname")
        return 1
    if status == 403:
        line(False, "AHA open refused this Host/Origin: the hostname must be listed in `hosts` in config.toml (internet/lan profiles)")
        return 1
    if status != 200:
        line(False, f"/v1/health answered {status}: this is not an AHA open /v1 server, or a proxy is in the way")
        return 1
    try:
        health = json.loads(body)
        line(True, f"server is up (status {health['status']}, API {health['api_version']})")
    except (ValueError, KeyError):
        line(False, "the address answered 200 but not with a /v1 health document")
        return 1
    if not args.token:
        print("      (give --token to check the token and list the models)")
        return 0
    status, _, body = _get(base + "/v1/manifest", {**extra, "Authorization": f"Bearer {args.token}"}, args.cacert)
    if status == 401:
        line(False, "the server is reachable but does not accept this token: it is wrong or was revoked (ask the operator for a new one)")
        return 1
    if status != 200:
        line(False, f"/v1/manifest answered {status}")
        return 1
    manifest = json.loads(body)
    line(True, f"token accepted (profile {manifest['profile']})")
    for model in manifest["models"]:
        ok = model["state"] == "ready"
        detail = {
            "not_installed": "not installed on the server", "unsupported_hardware": "the server's hardware cannot run it",
            "loading": "loading, try again shortly", "error": "disabled by the operator",
        }.get(model["state"], "")
        line(ok, f"model {model['id']}: {model['state']}" + (f" - {detail}" if detail else ""))
    return 1 if problems else 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print (or save) the support report. Works with the server stopped, which is when the web page cannot be opened."""
    from .report import build_report

    text = build_report(_load(args), note=args.note or "")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out} ({len(text.encode('utf-8')) // 1024} KiB). Read it, then send it to whoever supports your installation.")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))  # UTF-8 whatever the console's code page is: the text is meant to be copied
    return 0


def cmd_tunnel(args: argparse.Namespace) -> int:
    """Print what the operator has to do for the internet profile. Nothing here touches a Cloudflare account."""
    try:  # this command only prints advice, so it must work before `init` too
        port = load(Path(args.data_dir) if args.data_dir else None).listen_port
    except ConfigError:
        port = 8890
    host = args.hostname
    print(f"""# 1. ~/.cloudflared/config.yml  (after `cloudflared tunnel create aha-open` and `cloudflared tunnel route dns aha-open {host}`)
tunnel: aha-open
credentials-file: <path printed by `cloudflared tunnel create`>
ingress:
  - hostname: {host}
    service: http://127.0.0.1:{port}
  - service: http_status:404

# 2. AHA open config.toml
profile = "internet"
listen = "127.0.0.1:{port}"
hosts = ["{host}"]
access_confirmed = true     # set only after step 3 is done

# 3. Cloudflare Zero Trust > Access > Applications: add {host}, policy "Service Auth", one service token per person/machine.
#    Cloudflare then sees the images and prompts in clear text at its edge (TLS ends there); use `lan` or a VPN if that is not acceptable.

# 4. Check from another network:
#    aha-open check https://{host} --cf-id <id> --cf-secret <secret> --token <aha-open token>""")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _load(args)
    problems = 0

    def line(ok: bool, text: str) -> None:
        nonlocal problems
        problems += 0 if ok else 1
        print(("ok    " if ok else "FAIL  ") + text)

    print(f"AHA open {__version__}  data_dir={cfg.data_dir}  profile={cfg.profile}  listen={cfg.listen}")
    gpus = detect_gpus(ttl_s=0)
    line(bool(gpus), "GPU: " + (", ".join(describe_gpu(g) for g in gpus) or "no NVIDIA GPU found (nvidia-smi)"))
    if cfg.gpu_memory_limit_gib:
        print(f"      [gpu] memory_limit_gib = {cfg.gpu_memory_limit_gib}: the server uses at most this much GPU memory")
    try:
        validate_for_serve(cfg)
        line(True, "configuration is allowed to start")
    except ConfigError as err:
        line(False, f"configuration: {err}")
    for mc in cfg.models:
        spec = CATALOG.get(mc.id)
        if spec is None:
            line(False, f"{mc.id}: not a known model")
            continue
        state, detail = model_state(cfg, mc, spec)
        line(state == "ready", f"{mc.id} [{mc.variant or 'default'}]: {state}" + (f" - {detail}" if detail else ""))
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aha-open", description="AHA open: open image/video models behind the neutral /v1 API")
    p.add_argument("--data-dir", help="data directory (default: AHA_OPEN_DATA_DIR or the per-user data folder)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create the data directory and a default config.toml")
    s.add_argument("--profile", choices=["local", "lan", "internet"], default="local")
    s.add_argument("--model", action="append", help="model id (repeatable)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("serve", help="run the gateway")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("start", help="start the gateway in the background if needed, then open the web page (local profile)")
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    s.add_argument("--gui", action="store_true", help="show a message box on failure (used by the Start-menu shortcut)")
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("stop", help="stop the gateway that `aha-open start` started")
    s.add_argument("--gui", action="store_true", help="show a message box (used by the Start-menu shortcut)")
    s.add_argument("--quiet", action="store_true", help="print and show nothing (used by the uninstaller)")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("open", help="open the bundled web page in your browser")
    s.add_argument("--print-only", action="store_true", help="do not start a browser")
    s.set_defaults(func=cmd_open)

    s = sub.add_parser("models", help="list or install models")
    ms = s.add_subparsers(dest="action", required=True)
    ms.add_parser("list", help="the configured models and, for each, which variants are installed and which one is in use")
    u = ms.add_parser("use", help="select an installed variant (restart the server to load it)")
    u.add_argument("model")
    u.add_argument("variant")
    r = ms.add_parser("remove", help="delete the files of one variant (shows what it would delete unless --yes is given)")
    r.add_argument("model")
    r.add_argument("variant")
    r.add_argument("--include-shared", action="store_true", help="also delete the files shared with other variants, if none of them is installed")
    r.add_argument("--yes", action="store_true", help="really delete")
    i = ms.add_parser("install", help="install a model (downloads weights, prepares its worker environment)")
    i.add_argument("model")
    i.add_argument("--variant", help="which variant to install (default: the best one that fits this machine; see `models list`)")
    i.add_argument("--accept-license", help="license key you accept for this model (see the message printed without it)")
    i.add_argument("--confirm-territory", action="store_true", help="confirm you are not in a territory the license excludes")
    i.add_argument("--from-dir", help="use weights already on disk instead of downloading")
    i.add_argument("--skip-env", action="store_true", help="do not create the worker virtual environment")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("loras", help="LoRA files this server offers to clients (Qwen)")
    ls = s.add_subparsers(dest="action", required=True)
    ls.add_parser("list")
    a = ls.add_parser("add", help="validate a .safetensors LoRA and copy it into the server's folder")
    a.add_argument("file")
    a.add_argument("--id", help="lowercase id clients use (default: the file name)")
    a.add_argument("--name", help="display name")
    a.add_argument("--scale", type=float, help="default strength, above 0 and at most 2 (default 1.0)")
    a.add_argument("--trigger", help="trigger words to put in the prompt, shown to clients")
    a.add_argument("--license", help="the LoRA's licence, shown to clients")
    a.add_argument("--notes")
    r = ls.add_parser("remove")
    r.add_argument("id")
    s.set_defaults(func=cmd_loras)

    s = sub.add_parser("clients", help="manage client tokens and pairing")
    cs = s.add_subparsers(dest="action", required=True)
    c = cs.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--kind")
    cs.add_parser("list")
    for name in ("revoke", "approve", "deny"):
        c = cs.add_parser(name)
        c.add_argument("target", help="client id/name (revoke) or pairing code (approve/deny)")
    s.set_defaults(func=cmd_clients)

    s = sub.add_parser("tls", help="certificate for the lan profile")
    ts = s.add_subparsers(dest="action", required=True)
    t = ts.add_parser("init")
    t.add_argument("--force", action="store_true")
    ts.add_parser("fingerprint")
    s.set_defaults(func=cmd_tls)

    s = sub.add_parser("check", help="check that a server is reachable and says which layer fails (any profile)")
    s.add_argument("url")
    s.add_argument("--token", help="client token: also checks it and lists the models")
    s.add_argument("--cacert", help="trust this certificate (the lan server's tls/server.crt)")
    s.add_argument("--cf-id", help="Cloudflare Access service token id (internet profile)")
    s.add_argument("--cf-secret", help="Cloudflare Access service token secret")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("tunnel", help="print the Cloudflare Tunnel + Access steps for the internet profile")
    s.add_argument("--hostname", required=True, help="public hostname, e.g. aha-open.example.com")
    s.set_defaults(func=cmd_tunnel)

    s = sub.add_parser("doctor", help="check GPU, configuration and models")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("report", help="print a support report (machine, models, end of the logs; no token, no prompt) to send to whoever helps you")
    s.add_argument("--out", help="write it to this file instead of printing it")
    s.add_argument("--note", help="what happened, in your own words (added at the top)")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
