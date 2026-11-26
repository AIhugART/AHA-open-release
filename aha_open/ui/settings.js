"use strict";
// ---- LoRA management: the server answers 404 unless this page runs on the machine itself (local profile), then it stays hidden.
async function loadAdmin() {
  let res;
  try { res = await (await api("/admin/loras")).json(); } catch (e) { $("loraadmin").hidden = true; return; }
  $("loraadmin").hidden = false;
  const list = $("loraadmlist"); list.replaceChildren();
  for (const l of res.loras) {
    const row = document.createElement("div"); row.className = "lora";
    const label = document.createElement("span"); label.textContent = `${l.display_name} (${l.id}, ${(l.bytes / 2 ** 20).toFixed(0)} MB)`;
    const rm = document.createElement("button"); rm.type = "button"; rm.className = "secondary"; rm.textContent = "Remove";
    rm.addEventListener("click", () => removeLora(l));
    row.append(label, rm); list.append(row);
  }
  for (const p of res.problems) { const d = document.createElement("div"); d.className = "note err"; d.textContent = p; list.append(d); }
}

async function removeLora(l) {
  if (!confirm(`Remove "${l.display_name}" from this server? A job that is already waiting and uses it will fail.`)) return;
  try { await api("/admin/loras/" + encodeURIComponent(l.id), { method: "DELETE" }); } catch (e) { say($("loraadmsg"), e.message, "err"); return; }
  say($("loraadmsg"), `Removed ${l.id}.`);
  await refreshManifest(); await loadAdmin();
}

async function addLora() {
  const file = $("lorafile").files[0], msg = $("loraadmsg"), id = $("loraid").value.trim();
  if (!file) { say(msg, "Choose a .safetensors file first.", "err"); return; }
  if (!id) { say(msg, "Give the LoRA an id (lowercase letters, digits, - or _).", "err"); return; }
  const q = new URLSearchParams();
  for (const [key, el] of [["name", "loraname"], ["trigger", "loratrigger"], ["scale", "lorascale"]]) if ($(el).value.trim()) q.set(key, $(el).value.trim());
  $("loraadd").disabled = true; say(msg, `Sending ${(file.size / 2 ** 20).toFixed(0)} MB and checking it...`);
  try {
    await api(`/admin/loras/${encodeURIComponent(id)}?${q}`, { method: "PUT", body: file });
    say(msg, `Added ${id}. It is in the list above.`);
    $("lorafile").value = ""; for (const el of ["loraid", "loraname", "loratrigger"]) $(el).value = "";
    await refreshManifest(); await loadAdmin();
  } catch (e) { say(msg, e.message, "err"); } finally { $("loraadd").disabled = false; }
}

$("lorafile").addEventListener("change", () => {
  const f = $("lorafile").files[0]; if (!f) return;
  const stem = f.name.replace(/\.safetensors$/i, "").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  if (!$("loraid").value) $("loraid").value = stem.slice(0, 64);
  if (!$("loraname").value) $("loraname").value = f.name.replace(/\.safetensors$/i, "");
});
$("loraadd").addEventListener("click", addLora);
$("lorarefresh").addEventListener("click", async () => {
  try { await refreshManifest(); await loadAdmin(); } catch (e) { say($("loranote"), "Could not read the list: " + e.message, "err"); }
});

// ---- Download a model: the server answers 404 unless this page runs on the machine itself (local profile), then the panel stays hidden.
// What the Download and Manage sections say instead of an empty panel: it is not the machine's own page, or there is nothing to list yet.
let adminOk = null; // null = not asked yet, false = the server answered 404 to /admin/*
function adminNotes() {
  const remote = adminOk === false;
  $("remote-download").hidden = $("remote-models").hidden = $("reportremote").hidden = !remote; // the Report says the same thing about the logs
  $("setup-none").hidden = !(adminOk && $("setup").hidden);
  $("models-none").hidden = !(adminOk && $("models").hidden);
}
const PHASE_TEXT = {
  preparing: "Checking your machine...",
  downloading: "Downloading the model...",
  creating_environment: "Setting up the model's engine. This installs PyTorch and takes several minutes...",
  finishing: "Finishing...",
};
const gb = n => (n / 2 ** 30).toFixed(1);
const COMMERCIAL = { not_allowed: "Research and evaluation only. Not for commercial use.", allowed_with_conditions: "Commercial use with conditions.", allowed: "Commercial use allowed." };
const installTimers = {}, refreshedFor = {};

async function loadSetup() {
  let res;
  try { res = await (await api("/admin/models")).json(); } catch (e) { adminOk = false; adminNotes(); $("setup").hidden = true; $("more-models").hidden = true; return; }
  adminOk = true;
  // first downloads only: a model with a build on this machine is managed under Manage Models, even when it cannot run right now
  const todo = res.models.filter(x => x.state !== "ready" && !(x.variants || []).some(v => v.installed));
  const offered = todo.filter(x => !x.blocker);  // what this machine can actually download: the hint on Generate names only these
  $("setup").hidden = todo.length === 0;
  // Generate lists only what is installed: say there is more to download (a video model is the usual surprise), and where.
  const more = $("more-models"); more.replaceChildren(); more.hidden = offered.length === 0;
  if (offered.length) {
    const a = document.createElement("a"); a.href = "#/settings/download"; a.textContent = "Download a Model";
    more.append("More models can be downloaded to this machine: " + offered.map(m => m.display_name).join(", ") + ". ", a);
  }
  adminNotes();
  const list = $("setuplist"); list.replaceChildren();
  for (const mod of todo) list.append(renderSetup(mod));
}

function renderSetup(mod) {
  const box = document.createElement("div"); box.className = "setup-model";
  const title = document.createElement("strong"); title.textContent = `${mod.display_name} · about ${mod.disk_gib} GB to download`;
  const lic = mod.license, cu = lic.commercial_use;
  const facts = document.createElement("div");
  facts.append(badge(lic.name, ""), badge(cu === "allowed" ? "commercial use allowed" : "commercial use: " + cu, cu === "allowed" ? "ok" : cu === "not_allowed" ? "bad" : "warn"));
  const terms = document.createElement("p"); terms.className = "note";
  terms.textContent = (lic.notes || COMMERCIAL[cu] || "") + " "; // the server's own words first; ours only if it sent none
  if (lic.url) { const a = document.createElement("a"); a.href = lic.url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = "Read the full licence"; terms.append(a); }
  const agree = document.createElement("label");
  const tick = document.createElement("input"); tick.type = "checkbox";
  agree.append(tick, " I have read the licence and I accept it.");
  // A licence that excludes territories (MiniMax H3) needs its own confirmation: a separate box, never folded into the licence one.
  let placeTick = null, place = null;
  if (mod.requires_territory_confirmation) {
    const names = { EU: "the European Union", GB: "the United Kingdom", KR: "South Korea", US: "the United States" };
    const where = (lic.excluded_territories || []).map(c => names[c] || c);
    const list = where.length > 1 ? where.slice(0, -1).join(", ") + " or " + where[where.length - 1] : where.join("");
    place = document.createElement("label");
    placeTick = document.createElement("input"); placeTick.type = "checkbox";
    place.append(placeTick, ` I confirm that I am not in ${list || "an excluded territory"}, where this licence does not allow use.`);
  }
  let pick = null;
  if ((mod.variants || []).length > 1) {
    pick = document.createElement("select"); pick.setAttribute("aria-label", "Variant to download");
    for (const v of mod.variants) pick.add(new Option(`${v.label}: about ${v.download_gib || v.disk_gib} GB, ${FIT[v.fit][0]}${v.recommended ? " (recommended)" : ""}`, v.name));
    pick.value = (mod.variants.find(v => v.recommended) || mod.variants[0]).name;
  }
  const actions = document.createElement("div"); actions.className = "generate-actions";
  const go = document.createElement("button"); go.type = "button"; go.textContent = "Download model";
  const stop = document.createElement("button"); stop.type = "button"; stop.className = "secondary"; stop.textContent = "Cancel"; stop.hidden = true;
  actions.append(go, stop);
  const wrap = document.createElement("div"); wrap.className = "bar"; wrap.hidden = true;
  const fill = document.createElement("i"); wrap.append(fill);
  const msg = document.createElement("div"); msg.className = "note"; msg.setAttribute("role", "status");
  let about = null;  // what the picked variant costs and how it fits, size by size
  if (pick) {
    about = document.createElement("div"); about.className = "note";
    const show = () => {
      const v = mod.variants.find(x => x.name === pick.value);
      about.textContent = `${costOf(v)}. ${v.quality_note || ""}`;
      title.textContent = `${mod.display_name} · about ${v.download_gib || v.disk_gib} GB to download`;  // the picked variant, not the suggested one
    };
    pick.addEventListener("change", show); show();
  }
  box.append(title, facts, terms, ...(pick ? [pick, about] : []), agree, ...(place ? [place] : []), actions, wrap, msg);

  const ui = { go, stop, tick, placeTick, wrap, fill, msg };
  const sync = () => { go.disabled = !tick.checked || (placeTick !== null && !placeTick.checked) || !!mod.blocker || mod.install.state === "running"; };
  tick.addEventListener("change", sync);
  if (placeTick) placeTick.addEventListener("change", sync);
  go.addEventListener("click", async () => {
    go.disabled = true; say(msg, "Starting...");
    const body = { accept_license: lic.key };
    if (pick) body.variant = pick.value;
    if (placeTick) body.confirm_territory = placeTick.checked;
    try { mod.install = await (await api(`/admin/models/${encodeURIComponent(mod.id)}/install`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json(); }
    catch (e) { say(msg, e.message, "err"); sync(); return; }
    showInstall(mod, ui);
  });
  stop.addEventListener("click", async () => {
    stop.disabled = true;
    try { await api(`/admin/models/${encodeURIComponent(mod.id)}/install`, { method: "DELETE" }); } catch (e) { say(msg, e.message, "err"); }
    stop.disabled = false;
  });
  if (mod.blocker) say(msg, mod.blocker.message, "err");
  sync();
  if (mod.install.state === "running") { tick.checked = true; if (placeTick) placeTick.checked = true; showInstall(mod, ui); } else showInstall(mod, ui);
  return box;
}

// One place that turns a status into what the person sees, and keeps polling while it runs.
function showInstall(mod, ui) {
  const s = mod.install, running = s.state === "running";
  busy("settings", running); // a dot beside Settings while a download runs, wherever the person is
  ui.go.hidden = running; ui.stop.hidden = !running; ui.wrap.hidden = !running; ui.tick.disabled = running;
  if (ui.sync) ui.sync(); // the buttons that depend on "an install is running" follow every change of the status: finished, cancelled, failed
  if (ui.placeTick) ui.placeTick.disabled = running;
  const timerKey = mod.id + (ui.key || "");
  clearTimeout(installTimers[timerKey]);
  if (running) {
    const frac = s.phase === "downloading" && s.expected_bytes ? Math.min(1, s.downloaded_bytes / s.expected_bytes) : null;
    ui.fill.style.width = frac === null ? "100%" : Math.round(frac * 100) + "%";
    ui.fill.className = frac === null ? "indeterminate" : "";
    const bytes = s.phase === "downloading" ? ` ${gb(s.downloaded_bytes)} of about ${gb(s.expected_bytes)} GB.` : "";
    say(ui.msg, (PHASE_TEXT[s.phase] || "Working...") + bytes);
    installTimers[timerKey] = setTimeout(async () => {
      try { mod.install = await (await api(`/admin/models/${encodeURIComponent(mod.id)}/install`)).json(); }
      catch (e) { say(ui.msg, "Lost contact with the server: " + e.message, "err"); installTimers[timerKey] = setTimeout(() => showInstall(mod, ui), 3000); return; }
      showInstall(mod, ui);
    }, 1000);
    return;
  }
  ui.go.disabled = !ui.tick.checked || (ui.placeTick !== null && !ui.placeTick.checked) || !!mod.blocker;
  if (s.state === "succeeded") {
    if (refreshedFor[mod.id] === s.finished_at) { // second look, after loadSetup() drew this panel again with the server's own state
      if (mod.state === "ready") { say(ui.msg, "Downloaded. Press Use this to switch to it.", "ok"); return; } // another variant of a model that already works
      say(ui.msg, `The download finished, but the model is not ready yet: ${mod.state_detail || mod.state}.`, "err"); return;
    }
    refreshedFor[mod.id] = s.finished_at;
    say(ui.msg, "Finished. Checking the model...");
    refreshManifest().then(() => { $("make").hidden = manifest.models.length === 0; const now = manifest.models.find(x => x.id === mod.id); if (now && now.state === "ready") say($("status"), `${mod.display_name} is installed. You can create with it now.`, "ok"); return Promise.all([loadSetup(), loadModels()]); }).catch(e => say(ui.msg, "Installed, but the list could not be refreshed: " + e.message, "err"));
  } else if (s.state === "failed") { say(ui.msg, s.error ? s.error.message : "The download failed.", "err"); ui.go.textContent = "Try again"; }
  else if (s.state === "cancelled") { say(ui.msg, "Cancelled. What was downloaded is kept, so the next try continues from there."); ui.go.textContent = "Download model"; }
}

// ---- Models: choose, download and delete variants. The server answers 404 unless this page runs on the machine itself, then the panel stays hidden.
const keepOpen = new Set(); // a model whose last variant was just deleted stays listed, so the offer to delete its shared files has a place
const FIT = { fits: ["fits this GPU", "ok"], tight: ["tight on this GPU", "warn"], too_small: ["too big for this GPU", "bad"], unknown: ["not measured here", ""] };

async function loadModels() {
  let res;
  try { res = await (await api("/admin/models")).json(); } catch (e) { adminOk = false; adminNotes(); $("models").hidden = true; return; }
  adminOk = true;
  const mine = res.models.filter(m => (m.variants || []).some(v => v.installed) || m.state === "ready" || keepOpen.has(m.id));
  $("models").hidden = mine.length === 0;
  adminNotes();
  const list = $("modelslist"); list.replaceChildren();
  for (const mod of mine) list.append(renderModel(mod));
}

function renderModel(mod) {
  const box = document.createElement("div"); box.className = "setup-model"; box.dataset.model = mod.id;
  const title = document.createElement("strong"); title.textContent = mod.display_name;
  const lic = mod.license;
  const agree = document.createElement("label"), tick = document.createElement("input"); tick.type = "checkbox";
  agree.append(tick, ` I have read the licence (${lic.name}) and I accept it. Needed to download another variant.`);
  let place = null, placeTick = null;
  if (mod.requires_territory_confirmation) {
    place = document.createElement("label"); placeTick = document.createElement("input"); placeTick.type = "checkbox";
    place.append(placeTick, " I confirm that I am not in a territory this licence excludes.");
  }
  const msg = document.createElement("div"); msg.className = "note"; msg.setAttribute("role", "status");
  const wrap = document.createElement("div"); wrap.className = "bar"; wrap.hidden = true;
  const fill = document.createElement("i"); wrap.append(fill);
  const stop = document.createElement("button"); stop.type = "button"; stop.className = "secondary"; stop.textContent = "Cancel download"; stop.hidden = true;
  const unused = document.createElement("button"); unused.hidden = true; // showInstall() also drives the Download panel's own button
  const ui = { go: unused, stop, tick, placeTick, wrap, fill, msg, key: ":models", sync: null };
  const rows = document.createElement("div"); rows.className = "variants";
  const sync = () => {
    const running = mod.install.state === "running";
    for (const b of rows.querySelectorAll("button[data-act=download]")) b.disabled = running || !tick.checked || (placeTick !== null && !placeTick.checked);
    for (const b of rows.querySelectorAll("button[data-act=use], button[data-act=delete]")) b.disabled = running;
  };
  const done = async text => { try { await refreshManifest(); } catch (e) {} await Promise.all([loadModels(), loadSetup()]); say($("modelsmsg"), text, "ok"); }; // the cards are drawn again: the news goes on a line that stays
  for (const v of mod.variants) rows.append(renderVariant(mod, v, { msg, done, sync }));
  ui.sync = sync;
  tick.addEventListener("change", sync);
  if (placeTick) placeTick.addEventListener("change", sync);
  stop.addEventListener("click", async () => {
    stop.disabled = true;
    try { await api(`/admin/models/${encodeURIComponent(mod.id)}/install`, { method: "DELETE" }); } catch (e) { say(msg, e.message, "err"); }
    stop.disabled = false;
  });
  box.append(title, rows, agree, ...(place ? [place] : []), stop, wrap, msg);
  rows.addEventListener("download", async ev => {
    const body = { accept_license: lic.key, variant: ev.detail };
    if (placeTick) body.confirm_territory = placeTick.checked;
    say(msg, "Starting...");
    try { mod.install = await (await api(`/admin/models/${encodeURIComponent(mod.id)}/install`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json(); }
    catch (e) { say(msg, e.message, "err"); sync(); return; }
    sync(); showInstall(mod, ui);
  });
  sync();
  if (mod.install.state === "running") { tick.checked = true; if (placeTick) placeTick.checked = true; sync(); }
  showInstall(mod, ui);
  return box;
}

// GPU memory of a variant from the measured numbers: one figure for an image model, one per size tier for a video model (with its fit).
function costOf(v) {
  const peaks = v.vram_gib || {};
  if (peaks["1k"] != null) return `about ${peaks["1k"]} GiB of GPU memory`;
  const tiers = Object.entries(peaks).filter(([, gib]) => gib != null)
    .map(([tier, gib]) => `${tier} size about ${gib} GiB${v.fit_by_tier && v.fit_by_tier[tier] ? " (" + FIT[v.fit_by_tier[tier]][0] + ")" : ""}`);
  for (const [tier, gib] of Object.entries(v.too_small_up_to_gib || {})) tiers.push(`${tier} size did not fit a ${gib} GiB card`);
  return tiers.length ? "GPU memory: " + tiers.join(", ") : "GPU memory not measured";
}

function renderVariant(mod, v, ctl) {
  const row = document.createElement("div"); row.className = "variant" + (v.selected ? " current" : ""); row.dataset.variant = v.name;
  const head = document.createElement("div");
  const name = document.createElement("strong"); name.textContent = v.label;
  head.append(name, " ");
  if (v.selected) head.append(badge("in use", "ok"));
  else if (v.installed) head.append(badge("installed", ""));
  if (v.recommended) head.append(badge("recommended for this machine", "ok"));
  head.append(badge(FIT[v.fit][0], FIT[v.fit][1]));
  const facts = document.createElement("div"); facts.className = "note";
  const vram = costOf(v);
  const secs = v.seconds && v.seconds["1k"] != null ? `, about ${Math.round(v.seconds["1k"])} s for a 1024 px image` : "";
  facts.textContent = `${v.installed ? "On this machine" : `${v.download_gib || v.disk_gib} GB to download`} · ${vram}${secs}. ${v.quality_note || ""}`;
  const actions = document.createElement("div"); actions.className = "generate-actions";
  const act = (text, kind, cls) => { const b = document.createElement("button"); b.type = "button"; b.textContent = text; b.dataset.act = kind; if (cls) b.className = cls; actions.append(b); return b; };
  if (v.installed && !v.selected) act("Use this", "use").addEventListener("click", async () => {
    say(ctl.msg, "Switching...");
    try { await api(`/admin/models/${encodeURIComponent(mod.id)}/use`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ variant: v.name }) }); }
    catch (e) { say(ctl.msg, e.message, "err"); return; }
    await ctl.done(`Now using ${v.label}. The next job loads it.`);
  });
  if (!v.installed) act("Download", "download", "").addEventListener("click", () => row.parentElement.dispatchEvent(new CustomEvent("download", { detail: v.name })));
  if (v.installed) act("Delete", "delete", "secondary").addEventListener("click", async () => {
    if (!confirm(`Delete "${v.label}" from this machine? This frees about ${v.own_gib} GB; files other variants still need are kept. You can download it again later.`)) return;
    say(ctl.msg, "Deleting...");
    let res;
    try { res = await (await api(`/admin/models/${encodeURIComponent(mod.id)}/variants/${encodeURIComponent(v.name)}`, { method: "DELETE" })).json(); }
    catch (e) { say(ctl.msg, e.message, "err"); return; }
    const shared = res.shared_offer === true;
    if (shared) keepOpen.add(mod.id);
    await ctl.done(res.message || `Deleted ${v.label}.`);
    if (shared) offerShared(mod, v);
  });
  row.append(head, facts, actions);
  return row;
}

// After the last variant of a group is deleted the files it shared (text encoder, VAE) stay; say how to remove them too.
function offerShared(mod, v) {
  const box = document.querySelector(`#modelslist [data-model="${mod.id}"]`) || $("modelslist");
  const line = document.createElement("div"); line.className = "note";
  const b = document.createElement("button"); b.type = "button"; b.className = "secondary"; b.textContent = "Also delete the shared files";
  b.addEventListener("click", async () => {
    if (!confirm("Delete the files that all variants of this model share? Downloading a variant again will fetch them again.")) return;
    try { const r = await (await api(`/admin/models/${encodeURIComponent(mod.id)}/variants/${encodeURIComponent(v.name)}?include_shared=true`, { method: "DELETE" })).json(); say(line, r.message || "Deleted.", "ok"); }
    catch (e) { say(line, e.message, "err"); }
    keepOpen.delete(mod.id); await loadSetup(); await loadModels();
  });
  line.append("The files shared with other variants were kept. ", b); box.append(line);
}

function badge(text, cls) { const s = document.createElement("span"); s.className = "badge " + cls; s.textContent = text; return s; }
function currentModel() { return manifest.models.find(x => x.id === $("model").value); }
