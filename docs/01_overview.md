# AHA open — overview

## 1. The problem

Open models such as Qwen-Image-2.1 and MiniMax H3 give good quality but need PyTorch, CUDA, tens of GB of weights and
library versions pinned separately for each model. People who use a creative application should not have to care about
that, and the creative application should not contain it.

## 2. The solution

AHA open is a **separate service**:

- it runs on a machine with a GPU (your own machine, a company server, or a machine at home reached over the internet);
- it loads open models, accepts requests over the `/v1` HTTP API, and returns images/video;
- it serves several clients at once (a queue, fair between clients).

## 3. Boundaries

| Part of AHA open | Not part of AHA open |
|---|---|
| Running models, VRAM management, the queue | Accounts, payments and usage rights of any client application |
| One token per client, pairing a machine | The user's image/video library (the client keeps it) |
| A manifest of each model's capabilities and license | Deciding a license on the operator's behalf |
| Deployment guides for LAN/internet | Model weights (the operator downloads them) |

**A client does not depend on AHA open.** A client only knows the `/v1` standard; any service implementing `/v1` correctly works.
Conversely AHA open does not depend on any client: every client that speaks `/v1` works.

**AHA open is an independent open-source project** (Apache-2.0) with its own repository, versions and release schedule.

**AHA open has its own simple web page** (`/ui/`) for generating images without any other client: choose a model, type a
prompt, choose a size, attach reference images, follow progress, cancel, download the result. It is **not** an image library or an
editor — it only keeps recent results for the job retention period. The page is a `/v1` client like any other and has no private path to a model.

## 4. Principles

1. **Data stays where the user chooses.** Images and prompts travel from the client to the machine running AHA open and nowhere else.
   AHA open sends no telemetry. The one exception is the operator's own choice: Cloudflare Tunnel (see
   [06_deployment.md](06_deployment.md)).
2. **Capabilities must be measurable.** The manifest declares only what has actually run on that hardware.
3. **Licenses are per model and shown clearly.** Weights are never bundled.
4. **Widening network exposure is an explicit act.** The default is to listen on `127.0.0.1` only.
5. **One environment per model.** Upgrading one model never breaks another.
6. **Cancel is real.** Cancelling stops at the next pipeline step; no result is returned after a cancel.

## 5. Who uses it

| User | Scenario |
|---|---|
| An individual with a GPU | The client application and AHA open on the same machine |
| A studio/company | One GPU server serving employees' machines on the LAN |
| An individual/small team | A GPU machine at home, used from a laptop elsewhere through a Cloudflare Tunnel |
| A developer | Uses `/v1` for their own application |
