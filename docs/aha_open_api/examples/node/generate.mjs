// Text to image with Node 18 or newer: built-in fetch, no packages, one file.
//
//   node generate.mjs "a red fox in the snow" fox.png
//   node generate.mjs "a red fox" fox.png --model qwen-image-2.1 --width 1024 --height 1024 --seed 7
//
// Settings: AHA_OPEN_URL (default http://127.0.0.1:8890), AHA_OPEN_TOKEN (default: the local token file).
// Run it from Node, not from a web page: a browser adds an Origin header, and AHA open answers 403 to that (docs/03_api_v1.md section 1).
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { model: { type: "string" }, width: { type: "string" }, height: { type: "string" }, seed: { type: "string" }, steps: { type: "string" } },
});
const [prompt, out = "out.png"] = positionals;
if (!prompt) { console.error('usage: node generate.mjs "prompt" [out.png] [--model id] [--width n --height n] [--seed n] [--steps n]'); process.exit(2); }

const base = (process.env.AHA_OPEN_URL || "http://127.0.0.1:8890").replace(/\/+$/, "");

function findToken() {
  if (process.env.AHA_OPEN_TOKEN) return process.env.AHA_OPEN_TOKEN.trim();
  const dir = process.env.AHA_OPEN_DATA_DIR
    || (process.platform === "win32" ? join(process.env.LOCALAPPDATA || homedir(), "AHA-open")
      : join(process.env.XDG_DATA_HOME || join(homedir(), ".local", "share"), "aha-open"));
  try { return readFileSync(join(dir, "local_client.token"), "utf8").trim(); } catch { return ""; }
}
const token = findToken();
if (!token) { console.error("error: no token. Set AHA_OPEN_TOKEN (docs/aha_open_api/01_quickstart.md)."); process.exit(1); }

class AhaOpenError extends Error {
  constructor(status, code, message, retryAfterS = null) { super(`${code}: ${message}`); Object.assign(this, { status, code, retryAfterS }); }
}

// The token travels only in this header; nothing in this file prints it.
async function api(method, path, body) {
  let response;
  try {
    response = await fetch(base + path, {
      method,
      headers: { Authorization: `Bearer ${token}`, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new AhaOpenError(0, "unreachable", `cannot reach ${base} (${err.cause?.code || err.message})`);
  }
  if (!response.ok) {
    let info = {};
    try { info = (await response.json()).error ?? {}; } catch { /* not JSON */ }
    throw new AhaOpenError(response.status, info.code ?? "error", info.message ?? `HTTP ${response.status}`, info.retry_after_s ?? null);
  }
  return path.includes("/result") ? Buffer.from(await response.arrayBuffer()) : response.json();
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

try {
  const { models } = await api("GET", "/v1/manifest");
  const candidates = models.filter((m) => m.tasks.includes("image.generate") && (!values.model || m.id === values.model));
  const model = candidates.find((m) => m.state === "ready");
  if (!model) {
    const why = candidates.map((m) => `${m.id} is ${m.state}`).join(", ") || "no model offers image.generate";
    throw new AhaOpenError(0, "no_ready_model", `no model is ready (${why}). Install the model or wait for it to load.`);
  }
  console.error(`model ${model.id} - license: ${model.license.name} (commercial use: ${model.license.commercial_use})`);

  const size = values.width ? { width: Number(values.width), height: Number(values.height) } : model.capabilities.sizes[0]; // sizes are never stretched
  const params = { width: size.width, height: size.height };
  if (values.seed !== undefined) params.seed = Number(values.seed);
  if (values.steps !== undefined) params.steps = Number(values.steps);
  const request = { task: "image.generate", model: model.id, prompt, params, client_request_id: crypto.randomUUID() };

  let job;
  for (;;) {                       // wait out `busy`; the same client_request_id can never create two jobs
    try { job = await api("POST", "/v1/jobs", request); break; }
    catch (err) { if (err.code !== "busy") throw err; await sleep(Math.max(1, err.retryAfterS ?? 1) * 1000); }
  }
  for (let pause = 250; !["succeeded", "failed", "cancelled"].includes(job.status); pause = Math.min(pause * 1.5, 2000)) {
    await sleep(pause);
    job = await api("GET", `/v1/jobs/${job.id}`);
  }
  if (job.status !== "succeeded") throw new AhaOpenError(0, job.error?.code ?? job.status, job.error?.message ?? `the job ${job.status}`);
  writeFileSync(out, await api("GET", `/v1/jobs/${job.id}/result?index=0`));
  const r = job.results[0];
  console.error(`saved ${out}: ${r.width}x${r.height}, ${r.bytes} bytes, seed ${r.seed}`);
} catch (err) {
  console.error(`error: ${err.message}`);
  if (err.code === "unauthorized") console.error("The server did not accept the token. Set AHA_OPEN_TOKEN to a valid client token (see docs/aha_open_api/01_quickstart.md).");
  process.exit(1);
}
