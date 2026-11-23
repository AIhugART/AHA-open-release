# AHA open security

Report a vulnerability privately: see [../SECURITY.md](../SECURITY.md).

## 1. Assets to protect

| Asset | Why |
|---|---|
| Users' images, prompts, videos | Private data, possibly a company's customer assets |
| The GPU | An expensive resource; if it is taken over the service is lost and power is wasted |
| Client tokens | Whoever holds a token can use the GPU and read the token owner's jobs |
| Names/addresses of internal servers | They reveal a company's network structure |

## 2. Threats by profile

| Threat | `local` | `lan` | `internet` |
|---|---|---|---|
| A web page in a browser calling `127.0.0.1`/an internal IP | yes | yes | yes |
| Another machine on the network calling in | — | yes | — |
| Eavesdropping on the wire | — | yes | (Cloudflare sees the data in clear at the edge) |
| Bots scanning the whole internet | — | — | yes |
| A leaked token | low | medium | high |

## 3. Mandatory rules

1. **Listen on `127.0.0.1` only by default.** The `lan` profile must name an interface explicitly; listening on `0.0.0.0` by itself is forbidden.
2. **`lan` requires HTTPS.** A self-signed certificate + fingerprint, or a certificate from an internal CA.
3. **`internet` only through a tunnel:** the gateway still listens on loopback; Cloudflare Access (a service token) **and** the
   AHA open token are both required — two independent layers. Pairing is off by default.
4. **One token per client**, stored only as a hash, revoked individually, shown exactly once.
5. **Job ownership:** only the token that created a job may read/cancel/download it; any other token gets `404`.
6. **Block browsers:** a request with a foreign `Origin` ⇒ `403`; the `Host` is checked against the list of configured server names
   (against DNS rebinding).
7. **Limits:** per-client queue, upload size, number of parts, job-creation rate; `429` beyond them.
8. **No telemetry.** AHA open sends nothing to anyone.
9. **Logs contain no prompt and no token.** Only the job id, model, size, time and error code are recorded.
10. **The code run in a worker is pinned code.** No `trust_remote_code` with unreviewed code; weights are loaded as `safetensors`,
    never an arbitrary pickle.
11. **Pairing needs a human to approve** on the server machine; a pairing code expires after a few minutes and is single-use.
12. **`local_client.token`** (the `local` profile) is written only when the profile is `local`, readable only by the current user, and removed
    when the profile changes.
13. **A model with "hosted service" obligations** (MiniMax H3) is off by default on the `internet` profile.

## 4. Mandatory negative tests (the conformance suite)

| Test | Expectation |
|---|---|
| No token / a wrong token | `401` |
| Token A reads/cancels/downloads token B's job | `404` |
| Header `Origin: https://evil.example` | `403` |
| A foreign `Host` | `403` |
| The `lan` profile starting with `listen = 0.0.0.0` | refuses to start |
| The `lan` profile without TLS | refuses to start |
| An upload with a wrong sha256 | `400 upload_incomplete` |
| Exceeding the queue | `429 busy` |
| A used/expired pairing code | no token is issued |
| A revoked token | `401` on the very next call |
| The `lan`/`internet` profiles | no `local_client.token` on disk |
| The `internet` profile, MiniMax obligations not accepted | the model's `state` is not `ready`, jobs are refused |
| Negative search: a sample prompt/token in the logs and state files | 0 matches |

## 5. The operator's responsibilities

- Comply with each model's license (see [05_models.md](05_models.md)).
- Decide whether to accept that Cloudflare sees the data (the `internet` profile).
- Protect the machine running AHA open: another process of the same user on the same machine can read `data_dir`.

## 6. Reporting security issues

See [../SECURITY.md](../SECURITY.md): a private channel, not a public issue.
