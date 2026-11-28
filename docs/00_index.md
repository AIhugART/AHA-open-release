# AHA open — documentation map

| File | Role | Read it when |
|---|---|---|
| [01_overview.md](product/01_overview.md) | Goals, boundaries, principles | First look |
| [03_api_v1.md](product/03_api_v1.md) | The `/v1` API specification (OpenAPI is in [`../schema/`](../schema/)) | Writing a client or another `/v1` provider |
| [aha_open_api/README.md](aha_open_api/README.md) | **How-to guides for using the API**: quickstart, concepts, building a client, recipes, errors, developer hand-off, runnable examples (Python, PowerShell, curl, Node) | Using AHA open from a script, or building an app on it |
| [04_architecture.md](product/04_architecture.md) | How AHA open works inside | Developing AHA open |
| [05_models.md](product/05_models.md) | Models, hardware, licenses, measurements | Choosing a model, deploying |
| [06_deployment.md](product/06_deployment.md) | The three deployment profiles | Installing |
| [07_security.md](product/07_security.md) | Threat model and security rules | Opening AHA open to a LAN or the internet |
| [08_auto_test_and_debug.md](product/08_auto_test_and_debug.md) | `aha-open-mcp`: agent-driven end-to-end tests and diagnosis | Testing a real server, debugging a failed job |
| [10_web_ui.md](product/10_web_ui.md) | The web page: the menu (Generate, Settings, Report, About), the support report | Using or changing `/ui/` |
| [../SECURITY.md](../SECURITY.md) | How to report a vulnerability | You found a security problem |
| [../THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) | Licenses of the packages AHA open installs | Checking compliance |

(The file numbers keep the project's historical numbering; `02` is not part of the public documentation.)

## Conventions

- **Measured** and **inferred** are always kept apart; a number that has not been measured is written "to be measured".
- Evidence levels: ✅ run for real (real model, real GPU, real network) · 🧪 code and tests only · ⬜ not checked.
- Changing the API ⇒ update `03_api_v1.md` (and the OpenAPI file) in the same commit.
- Adding a model ⇒ add a section to `05_models.md` with hardware measurements and the license quoted verbatim.
- English is the canonical language. Vietnamese translations are in [`vi/`](vi/) and may lag behind.

## Versions

| Component | Version |
|---|---|
| API specification | `v1` (draft) |
| AHA open | not released yet (`0.1.0.dev`) |
