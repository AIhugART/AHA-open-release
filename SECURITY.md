# Security policy

AHA open runs AI models on your GPU and exposes them over HTTP, so a vulnerability can mean someone else uses your GPU,
reads your prompts and images, or runs code on your machine. We want to hear about it before anyone else does.

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private vulnerability reporting instead:
on the repository page choose **Security → Report a vulnerability**. Please include:

- the version or commit (`aha-open --version`, or `git rev-parse HEAD`) and the profile you ran (`local`, `lan`, `internet`);
- what you did, what you expected, what happened, and the smallest request or file that reproduces it;
- what an attacker gains (read another client's job, use the GPU without a token, run code, ...).

Please do not include real prompts, images, tokens or model weights in a report; synthetic ones are enough.

We aim to acknowledge a report within 7 days and to tell you what we plan to do within 30 days. These are targets for a
small open-source project, not a contract. If you want public credit once a fix is released, say so.

## Supported versions

Only the latest release and the `main` branch receive security fixes. There is no stable release yet (`0.1.0.dev`).

## What is in scope

The rules AHA open promises are written in [docs/product/07_security.md](docs/product/07_security.md), and each one has a negative test in
`conformance/`. A report that shows one of these rules can be broken is in scope, for example:

- reading, cancelling or downloading another client's job (must be `404`);
- using the API without a valid token, or with a revoked one (must be `401`);
- a web page in a browser reaching the server through a foreign `Origin` or a rebound `Host` (must be `403`);
- the server listening beyond what the profile allows (`0.0.0.0` on `lan`, plain HTTP on `lan`);
- a token, prompt or image appearing in a log, a database or any file other than the job's own folder;
- a crafted upload, LoRA file or request that crashes the gateway, reads or writes outside its folders, or runs code
  (LoRA files must be `.safetensors`; nothing is unpickled);
- the server sending data to a third party (AHA open has no telemetry).

## What is out of scope

- Anything that needs the attacker to already control the operator's account or `data_dir` (the operator's user can read it by design).
- Cloudflare, Hugging Face, or the model publishers' services themselves; report those to their owners.
- Model behaviour (what a model generates) and model licenses; see [docs/product/05_models.md](docs/product/05_models.md).
- Setups that go against the documentation, for example opening the port to the internet without the tunnel and Access.
- Denial of service by a client with a valid token within the documented limits (queue, upload size, rate). A limit that
  is missing or can be bypassed is in scope.

## Operator checklist

- Keep the default `listen = 127.0.0.1:8890` unless you need another machine; `lan` needs HTTPS and one token per machine.
- `internet` means Cloudflare Tunnel with Cloudflare Access in front **and** the AHA open token; Cloudflare can see the traffic at its edge.
- Revoke a token with `aha-open clients revoke <id>`; it stops working on the next request.
- Download weights only from the publisher, at the revision AHA open pins, and read the license you accept.
