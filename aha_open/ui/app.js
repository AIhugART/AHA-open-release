"use strict";
const $ = id => document.getElementById(id);
let token = "", manifest = null, jobId = null, polling = false, lastJob = null;
// What went wrong on this page, kept in memory for the Report section: a code and the server's own sentence, never a prompt or a token.
const pageErrors = [];
function noteError(kind, text) {
  pageErrors.push(`${new Date().toISOString()} ${kind}: ${String(text).slice(0, 300)}`);
  if (pageErrors.length > 40) pageErrors.shift();
}
window.addEventListener("error", e => noteError("script error", `${e.message} (${String(e.filename || "").split("/").pop()}:${e.lineno})`));
window.addEventListener("unhandledrejection", e => noteError("unhandled", e.reason && e.reason.message ? e.reason.message : e.reason));
try { token = sessionStorage.getItem("aho_token") || ""; } catch (e) { /* storage may be blocked; the page still works */ }
// `aha-open open` puts the token in the URL fragment: it is never sent to the server and never reaches a log.
const m = location.hash.match(/token=([^&]+)/);
if (m) { token = decodeURIComponent(m[1]); try { sessionStorage.setItem("aho_token", token); } catch (e) {} history.replaceState(null, "", location.pathname); }
$("token").value = token;

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: "Bearer " + token } });
  if (!res.ok) {
    let e = {}; try { e = (await res.json()).error || {}; } catch (_) {}
    const err = new Error(e.message || res.statusText); err.code = e.code || String(res.status); err.status = res.status; err.retry = e.retry_after_s;
    noteError("request failed", `${opts.method || "GET"} ${path.split("?")[0]} -> ${res.status} ${err.code}: ${err.message}`); // the path without its query, the server's sentence
    throw err;
  }
  return res;
}
const say = (el, text, cls = "") => { el.textContent = text; el.className = "note " + cls; };

// ---- Where the person is: one page, one section per menu entry, chosen by the address (#/generate, #/settings/models, ...).
// A job that is polling or a model that is downloading keeps going while another section is shown: it is the same page.
const ROUTES = {
  "generate": { page: "page-generate", nav: "generate", title: "Generate" },
  "settings/connect": { page: "page-connect", nav: "settings", title: "Connect Your Server" },
  "settings/download": { page: "page-download", nav: "settings", title: "Download a Model" },
  "settings/models": { page: "page-models", nav: "settings", title: "Manage Models" },
  "report": { page: "page-report", nav: "report", title: "Report" },
  "about": { page: "page-about", nav: "about", title: "About AHA open" },
};
const onEnter = {}; // route -> function called each time the section is shown (set by the file that owns the section)
let route = null, landingPending = false, isConnected = false;

function routeOf(hash) {
  const h = hash.replace(/^#\/?/, "").replace(/\/+$/, "");
  return h === "settings" ? "settings/connect" : Object.hasOwn(ROUTES, h) ? h : null;
}

function render(userMoved) {
  route = routeOf(location.hash);
  const shown = route || "settings/connect", r = ROUTES[shown]; // no address yet (still connecting): the Connect section says what is going on
  for (const p of document.querySelectorAll(".page")) p.hidden = p.id !== r.page;
  const mark = (a, on) => { a.classList.toggle("current", on); if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); };
  for (const a of document.querySelectorAll("[data-nav]")) mark(a, a.dataset.nav === r.nav);
  for (const a of document.querySelectorAll("[data-sub]")) mark(a, a.dataset.sub === shown);
  $("subnav").hidden = r.nav !== "settings";
  document.title = r.title + " - AHA open";
  if (userMoved) {
    const h = document.querySelector("#" + r.page + " h1, #" + r.page + " h2");
    if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); }
    window.scrollTo(0, 0);
  }
  if (onEnter[shown]) onEnter[shown]();
}

function navigate(to, replace) {
  if (replace) { history.replaceState(null, "", "#/" + to); render(false); } else location.hash = "#/" + to; // the hashchange event draws it
}

// A menu entry shows a dot while something started from its section is still running.
function busy(navKey, on) {
  for (const a of document.querySelectorAll(`[data-nav="${navKey}"]`)) { a.classList.toggle("busy", on); a.title = on ? "Working..." : ""; }
}

const hasReadyModel = () => !!manifest && manifest.models.some(x => x.state === "ready");

// What the header says, and where a first visit starts: a model that works -> Generate; none yet -> Download; no server -> Connect.
function connected(ok) {
  isConnected = ok;
  const pill = $("connpill");
  pill.textContent = ok ? `Connected - ${manifest.provider.name} ${manifest.provider.version}` : "Not connected";
  pill.className = "conn " + (ok ? "ok" : "");
  const first = landingPending; landingPending = false;
  if (first || (ok && route === "settings/connect")) navigate(ok ? (hasReadyModel() ? "generate" : "settings/download") : "settings/connect", first);
  else if (onEnter[route]) onEnter[route]();
}

async function connect() {
  const button = $("save"), field = $("token"), status = $("status");
  if (button.disabled) return;
  const accessToken = field.value.trim();
  if (!accessToken) {
    field.setAttribute("aria-invalid", "true");
    say(status, "Enter your access token to connect.", "err");
    field.focus();
    return;
  }
  token = accessToken;
  button.disabled = true;
  field.removeAttribute("aria-invalid");
  say(status, "Checking your connection…");
  try { sessionStorage.setItem("aho_token", token); } catch (e) {}
  try {
    manifest = await (await api("/v1/manifest")).json();
  } catch (e) {
    const why = { unauthorized: "That token is not accepted. Check it, or ask the operator for a new one.", forbidden: "The server refused this page (Host/Origin not allowed)." }[e.code];
    say(status, why || ("Could not reach the server. Check that it is running and try again. " + e.message), "err");
    if (e.code === "unauthorized") {
      field.setAttribute("aria-invalid", "true");
      if (route === "settings/connect") field.focus();
    }
    $("make").hidden = true;
    connected(false);
    return;
  } finally {
    button.disabled = false;
  }
  say(status, `Connected to ${manifest.provider.name} ${manifest.provider.version} (profile ${manifest.profile}).`, "ok");
  $("make").hidden = manifest.models.length === 0; // a fresh install has no model yet: Generate says so and Download a Model comes first
  fillModels();
  onModel();
  loadAdmin();
  loadSetup();
  loadModels();
  connected(true);
}

function fillModels() {
  $("model").innerHTML = "";
  for (const mod of manifest.models) $("model").add(new Option(mod.display_name + (mod.state === "ready" ? "" : " - " + mod.state), mod.id));
}

// Read the manifest again (a LoRA may have been added or removed) and keep what the user had chosen.
async function refreshManifest() {
  const model = $("model").value, task = $("task").value, picks = {};
  for (const r of document.querySelectorAll("#loralist .lora")) {
    const cb = r.querySelector("input[type=checkbox]"); picks[cb.dataset.id] = { on: cb.checked, scale: r.querySelector("input[type=text]").value };
  }
  manifest = await (await api("/v1/manifest")).json();
  fillModels();
  if (manifest.models.some(x => x.id === model)) $("model").value = model;
  onModel();
  if ([...$("task").options].some(o => o.value === task)) { $("task").value = task; onTask(); }
  for (const r of document.querySelectorAll("#loralist .lora")) {
    const cb = r.querySelector("input[type=checkbox]"), p = picks[cb.dataset.id];
    if (p) { cb.checked = p.on; r.querySelector("input[type=text]").value = p.scale; }
  }
}
