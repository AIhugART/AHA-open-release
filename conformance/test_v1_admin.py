"""The LoRA admin page (`/admin/loras`): for the person at this machine, in the `local` profile, and nobody else.

Real HTTP against a real gateway. Every security rule has a negative test.
"""

from __future__ import annotations

import json
import struct

from conftest import Api, new_client_token

from aha_open import loras


def lora_bytes(names=("transformer.blocks.0.attn.to_q.lora_A.weight", "transformer.blocks.0.attn.to_q.lora_B.weight")) -> bytes:
    header = {n: {"dtype": "F32", "shape": [2, 2], "data_offsets": [i * 16, (i + 1) * 16]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 32


def _manifest_ids(api) -> list[str]:
    model = next(m for m in api.get("/v1/manifest").json()["models"] if m["id"] == "example-image")
    return [x["id"] for x in model["capabilities"]["loras"]["available"]]


def test_a_lora_put_on_the_page_shows_up_in_the_manifest_without_a_restart(gw, api):
    assert _manifest_ids(api) == []
    done = api.put("/admin/loras/ink", params={"name": "Ink", "trigger": "inkpaint", "scale": "0.8"}, content=lora_bytes())
    assert done.status_code == 201, done.text
    body = done.json()
    assert body["id"] == "ink" and body["display_name"] == "Ink" and body["trigger"] == "inkpaint" and body["default_scale"] == 0.8
    assert _manifest_ids(api) == ["ink"]
    assert (loras.lora_dir(gw.cfg) / "ink.safetensors").read_bytes() == lora_bytes()
    assert json.loads((loras.lora_dir(gw.cfg) / "ink.json").read_text(encoding="utf-8"))["trigger"] == "inkpaint"
    assert not list((gw.cfg.data_dir / "tmp").glob("*"))  # the upload was moved, nothing is left behind
    assert str(gw.cfg.data_dir) not in done.text  # the answer never carries a server path
    assert api.get("/admin/loras").json()["loras"][0]["id"] == "ink"


def test_the_page_refuses_what_is_not_a_lora_and_leaves_no_trace(gw, api):
    for name, data in (("junk", b"not a safetensors file at all"), ("plain", lora_bytes(("model.layers.0.weight",)))):
        refused = api.put(f"/admin/loras/{name}", content=data)
        assert refused.status_code == 400, (name, refused.text)
        assert f"{name}.safetensors" in refused.text and "lora-" not in refused.text  # speaks of the user's file, not an internal name
    assert api.put("/admin/loras/Bad_ID", content=lora_bytes()).status_code == 400  # uppercase is not an allowed id
    assert api.put("/admin/loras/ok", params={"name": "x" * 501}, content=lora_bytes()).status_code == 400
    assert api.put("/admin/loras/ok", params={"scale": "3"}, content=lora_bytes()).status_code == 400
    assert api.put("/admin/loras/ok", params={"scale": "abc"}, content=lora_bytes()).status_code == 400
    assert not list(loras.lora_dir(gw.cfg).glob("*")) and not list((gw.cfg.data_dir / "tmp").glob("*"))
    assert _manifest_ids(api) == []


def test_an_existing_id_is_not_overwritten(gw, api):
    assert api.put("/admin/loras/ink", content=lora_bytes()).status_code == 201
    again = api.put("/admin/loras/ink", content=lora_bytes() + b"\0")
    assert again.status_code == 400 and "already exists" in again.text
    assert (loras.lora_dir(gw.cfg) / "ink.safetensors").read_bytes() == lora_bytes()


def test_delete_removes_the_files_and_unknown_ids_are_404(gw, api):
    api.put("/admin/loras/ink", params={"name": "Ink"}, content=lora_bytes())
    assert api.delete("/admin/loras/ink").status_code == 204
    assert _manifest_ids(api) == [] and not list(loras.lora_dir(gw.cfg).glob("*"))
    assert api.delete("/admin/loras/ink").status_code == 404
    assert api.delete("/admin/loras/..%2Fconfig").status_code in (400, 404)


def test_the_list_says_which_files_were_skipped_and_why(gw, api):
    folder = loras.lora_dir(gw.cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "old.ckpt").write_bytes(b"pickle")
    problems = api.get("/admin/loras").json()["problems"]
    assert any("old.ckpt" in p and "safetensors" in p for p in problems)


# ---------------------------------------------------------------------------------------------- negative tests


def test_no_token_or_a_wrong_token_is_401(gw):
    for token in (None, "aho_not-a-real-token"):
        anon = Api(gw.base, token)
        assert anon.get("/admin/loras").status_code == 401
        assert anon.put("/admin/loras/ink", content=lora_bytes()).status_code == 401
        assert anon.delete("/admin/loras/ink").status_code == 401
    assert not list(loras.lora_dir(gw.cfg).glob("*"))


def test_a_foreign_host_or_origin_is_403_before_anything_else(gw):
    token = gw.tokens["local"]
    assert Api(gw.base, token, host="evil.example").get("/admin/loras").status_code == 403  # DNS rebinding
    page = Api(gw.base, token)
    refused = page.put("/admin/loras/ink", headers={"Origin": "https://evil.example"}, content=lora_bytes())
    assert refused.status_code == 403  # a web page on another site cannot drive the admin page
    assert not list(loras.lora_dir(gw.cfg).glob("*"))


def test_the_internet_profile_has_no_admin_page_even_through_the_tunnel_path(make_gateway):
    """cloudflared reaches the gateway as 127.0.0.1 with the loopback Host; that must NOT open /admin."""
    gw = make_gateway(profile="internet", access_confirmed=True, hosts=["aha.example.com"])
    token = new_client_token(gw, "remote")
    for host in ("aha.example.com", "127.0.0.1"):
        tunnel = Api(gw.base, token, host=host)
        for method, path in (("GET", "/admin/loras"), ("PUT", "/admin/loras/ink"), ("DELETE", "/admin/loras/ink")):
            sent = tunnel.request(method, path, content=lora_bytes() if method == "PUT" else None)
            assert sent.status_code == 404, (host, method)
    assert not list(loras.lora_dir(gw.cfg).glob("*"))
    assert Api(gw.base, token, host="127.0.0.1").get("/v1/manifest").status_code == 200  # /v1 is unaffected
