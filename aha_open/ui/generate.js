"use strict";
const taskLabels = {
  "image.generate": "Generate Image",
  "image.edit": "Edit Image",
  "video.generate": "Generate Video",
};

function onModel() {
  const mod = currentModel(); if (!mod) return;
  const info = $("modelinfo"); info.replaceChildren();
  const lic = mod.license, cu = lic.commercial_use;
  info.append(badge("state: " + mod.state, mod.state === "ready" ? "ok" : "warn"));
  info.append(badge(lic.name + " - commercial use: " + cu, cu === "allowed" ? "ok" : cu === "not_allowed" ? "bad" : "warn"));
  if (lic.required_ui_label) info.append(badge(lic.required_ui_label, ""));
  if (lic.excluded_territories) info.append(badge("not licensed in: " + lic.excluded_territories.join(", "), "warn"));
  if (mod.variant) { const v = mod.variants && mod.variants[mod.variant]; info.append(badge("variant: " + mod.variant + (v && v.recommended ? " (recommended here)" : ""), "")); }
  if (mod.state_detail) info.append(document.createTextNode(" " + mod.state_detail));
  if (lic.notes) { const p = document.createElement("div"); p.textContent = lic.notes; info.append(p); }
  const tasks = $("task");
  tasks.replaceChildren();
  for (const task of mod.tasks) tasks.add(new Option(Object.hasOwn(taskLabels, task) ? taskLabels[task] : task, task));
  onTask();
}

function onTask() {
  const mod = currentModel(), task = $("task").value, caps = mod.capabilities || {};
  const video = task.startsWith("video."), edit = task === "image.edit";
  $("imagebox").hidden = video; $("videobox").hidden = !video; $("editbox").hidden = !edit;
  if (!video) {
    $("size").innerHTML = "";
    for (const s of caps.sizes || []) $("size").add(new Option(`${s.width} x ${s.height}  (${s.aspect_ratio}${s.tier ? ", " + s.tier : ""})`, `${s.width}x${s.height}`));
    $("steps").value = caps.steps ? caps.steps.default : ""; $("transparentrow").hidden = !caps.rgba;
    const native = (caps.sizes || []).some(s => s.tier === "native");
    const lc = caps.loras, list = $("loralist"); list.replaceChildren();
    $("lorabox").hidden = !lc;  // shown for every model that takes LoRA, even with none installed: the Refresh button and the manager live here
    for (const l of (lc && lc.available) || []) {
      const row = document.createElement("div"); row.className = "lora";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.dataset.id = l.id; cb.id = "lora_" + l.id;
      const lab = document.createElement("label"); lab.htmlFor = cb.id; lab.style.margin = "0"; lab.style.color = "inherit"; lab.textContent = l.display_name;
      const sc = document.createElement("input"); sc.type = "text"; sc.inputMode = "decimal"; sc.value = l.default_scale; sc.setAttribute("aria-label", "strength of " + l.display_name);
      row.append(cb, lab, sc);
      if (l.trigger) row.append(badge("trigger: " + l.trigger, ""));
      if (l.license) row.append(badge(l.license, ""));
      list.append(row);
    }
    $("loranote").className = "note";
    $("loranote").textContent = !lc ? "" : lc.available && lc.available.length ? `Up to ${lc.max} at once; strength ${lc.scale.min} to ${lc.scale.max}. Put the trigger words in your prompt.` : "No LoRA is installed on this server yet.";
    $("sizenote").textContent = native ? "1k sizes are fast. Native sizes are the model's own and can need much more GPU memory (they may be very slow on a 24 GB card)." : "";
  } else {
    const fill = (el, list) => { el.innerHTML = ""; for (const v of list || []) el.add(new Option(v, v)); };
    const sized = Array.isArray(caps.sizes) && caps.sizes.length > 0;  // a model that declares exact sizes: pick one by name, never a free pair
    $("vsizerow").hidden = !sized; $("vratiorow").hidden = sized; $("vedgerow").hidden = sized;
    if (sized) {
      $("vsize").innerHTML = "";
      for (const s of fittingSizes(mod).shown) $("vsize").add(new Option(`${s.width} x ${s.height}  (${s.aspect_ratio}${s.tier ? ", " + s.tier : ""})`, `${s.aspect_ratio}|${s.short_edge}`));
    } else { fill($("vratio"), caps.aspect_ratios); fill($("vedge"), caps.short_edges); }
    const d = caps.durations_s;  // {min, max} or a list; a model with one length shows it and does not let it be changed
    const lo = d && d.min !== undefined ? d.min : Array.isArray(d) ? d[0] : "", hi = d && d.max !== undefined ? d.max : Array.isArray(d) ? d[d.length - 1] : "";
    $("vdur").value = lo; $("vdur").readOnly = lo !== "" && lo === hi;
    $("vdur").setAttribute("aria-description", lo === hi ? "this model makes clips of exactly this length" : `between ${lo} and ${hi} seconds`);
    videoNote();
    referenceBox(caps);
  }
}

// References for a video model that takes them: what is allowed comes from the manifest (`max_reference`), never from a model name.
function referenceBox(caps) {
  const max = caps.max_reference || {}, total = max.total || 0;
  $("refbox").hidden = total === 0; $("refs").value = "";
  if (!total) return;
  const kinds = [["image", "image/*"], ["video", "video/*"], ["audio", "audio/*"]].filter(([k]) => (max[k] || 0) > 0);
  $("refs").accept = kinds.map(([, t]) => t).join(",");
  $("reflabel").textContent = caps.reference_required ? "Reference (required)" : "Reference (optional)";
  const parts = kinds.map(([k]) => `${max[k]} ${k}${max[k] > 1 ? "s" : ""}`).join(", ");
  $("refnote").textContent = `Up to ${parts} (${total} in all). The model reads them in the order you choose them; a different order is a different clip.`
    + (caps.reference_required ? " This model needs at least one." : "");
}

// The sizes the variant in use can make on this server, from the manifest's `fit_by_tier` (measured peaks). A size that was not
// measured is offered; if nothing fits, everything is offered (the server then says clearly why a job failed).
function fittingSizes(mod) {
  const all = ((mod && mod.capabilities) || {}).sizes || [];
  const v = mod && mod.variants && mod.variant ? mod.variants[mod.variant] : null, byTier = (v && v.fit_by_tier) || {};
  const shown = all.filter(s => byTier[s.tier] !== "too_small");
  return shown.length ? { shown, hidden: all.filter(s => byTier[s.tier] === "too_small"), variant: v, none: false }
    : { shown: all, hidden: [], variant: v, none: all.length > 0 };
}

// What a video costs, said from the manifest only (no model name in this page): memory per size tier, and that it takes minutes.
function videoNote() {
  const mod = currentModel(), caps = (mod && mod.capabilities) || {}, v = mod && mod.variants && mod.variant ? mod.variants[mod.variant] : null;
  const picked = $("vsize").value.split("|"), size = (caps.sizes || []).find(s => s.aspect_ratio === picked[0] && String(s.short_edge) === picked[1]);
  const gib = size && v && v.vram_gib && size.tier ? v.vram_gib[size.tier] : null;
  const fit = fittingSizes(mod);
  const left = fit.hidden.map(s => `${s.width} x ${s.height}` + (fit.variant.too_small_up_to_gib && fit.variant.too_small_up_to_gib[s.tier] ? ` (did not fit a ${fit.variant.too_small_up_to_gib[s.tier]} GiB card)`
    : fit.variant.vram_gib && fit.variant.vram_gib[s.tier] ? ` (needs about ${fit.variant.vram_gib[s.tier]} GiB)` : "")).join(", ");
  $("vnote").textContent = "A video takes minutes, and the larger sizes take much longer." + (gib ? ` This size needs about ${gib} GiB of GPU memory.` : "")
    + (left ? ` Not offered, more GPU memory than this server has with ${fit.variant.label}: ${left}. Another variant may fit: see Settings, Models.` : "")
    + (fit.none ? ` No size of ${fit.variant.label} fits this server's GPU memory: choose another variant in Settings, Models.` : "")
    + (caps.durations_s && caps.durations_s.min === caps.durations_s.max ? ` This model makes clips of exactly ${caps.durations_s.min} seconds.` : "");
}

async function sha256Hex(buf) { return [...new Uint8Array(await crypto.subtle.digest("SHA-256", buf))].map(b => b.toString(16).padStart(2, "0")).join(""); }

async function uploadFile(file) {
  const data = await file.arrayBuffer();
  const made = await (await api("/v1/uploads", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bytes: data.byteLength, sha256: await sha256Hex(data), media_type: file.type || "image/png" }) })).json();
  const size = made.part_size;
  for (let n = 0, off = 0; off < data.byteLength; n++, off += size) {
    await api(`/v1/uploads/${made.upload_id}/parts/${n}`, { method: "PUT", body: data.slice(off, off + size) });
  }
  await api(`/v1/uploads/${made.upload_id}/complete`, { method: "POST" });
  return made.upload_id;
}

function intOrNull(v) { v = v.trim(); return v === "" ? null : Number(v); }

async function generate() {
  const task = $("task").value, body = { task, model: $("model").value, prompt: $("prompt").value, params: {}, inputs: [] };
  const caps = (currentModel() || {}).capabilities || {};
  const video = task.startsWith("video."), needsReference = video && !$("refbox").hidden && caps.reference_required;
  const requiredInput = task === "image.edit" ? $("files") : needsReference ? $("refs") : null;
  if (requiredInput && !requiredInput.files.length) {
    const edit = requiredInput === $("files");
    const error = edit ? $("fileerror") : $("referror");
    error.textContent = edit ? "Choose a source image." : "Choose at least one reference.";
    requiredInput.setAttribute("aria-invalid", "true");
    requiredInput.focus();
    return;
  }
  $("go").disabled = true;
  $("out").hidden = true;
  $("vid").pause();
  say($("jobline"), "Starting generation…");
  try {
    if (video) {
      const sized = !$("vsizerow").hidden, [ratio, edge] = sized ? $("vsize").value.split("|") : [$("vratio").value, $("vedge").value];
      Object.assign(body.params, { aspect_ratio: ratio, short_edge: Number(edge) || undefined, duration_s: intOrNull($("vdur").value) ?? undefined });
      const vseed = intOrNull($("vseed").value); if (vseed !== null) body.params.seed = vseed;
    } else {
      const [w, h] = $("size").value.split("x").map(Number); Object.assign(body.params, { width: w, height: h });
      const seed = intOrNull($("seed").value), steps = intOrNull($("steps").value);
      if (seed !== null) body.params.seed = seed; if (steps !== null) body.params.steps = steps;
      if (!$("transparentrow").hidden && $("transparent").checked) body.params.transparent = true;
      const picked = [...document.querySelectorAll("#loralist .lora")].filter(r => r.querySelector("input[type=checkbox]").checked)
        .map(r => ({ id: r.querySelector("input[type=checkbox]").dataset.id, scale: Number(r.querySelector("input[type=text]").value) }));
      if (picked.length) body.params.loras = picked;
    }
    if (video && !$("refbox").hidden) {
      const files = [...$("refs").files], role = f => f.type.startsWith("video/") ? "reference_video" : f.type.startsWith("audio/") ? "reference_audio" : "reference";
      for (let i = 0; i < files.length; i++) { say($("jobline"), `Uploading reference ${i + 1} of ${files.length}…`); body.inputs.push({ role: role(files[i]), upload_id: await uploadFile(files[i]) }); }
    }
    if (task === "image.edit") {
      const files = [...$("files").files];
      for (let i = 0; i < files.length; i++) { say($("jobline"), `Uploading image ${i + 1} of ${files.length}…`); body.inputs.push({ role: i ? "reference" : "source", upload_id: await uploadFile(files[i]) }); }
    }
    const job = await (await api("/v1/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    jobId = job.id;
    $("bar").classList.add("indeterminate");
    $("bar").style.width = "0";
    $("cancel").hidden = false;
    $("barwrap").hidden = false;
    poll();
  } catch (e) { say($("jobline"), explain(e), "err"); $("go").disabled = false; }
}

function explain(e) {
  const known = {
    model_not_installed: "This model is not installed on the server.", unsupported_hardware: "This server's hardware cannot run that model.",
    busy: `The server is busy; try again${e.retry ? " in " + Math.ceil(e.retry) + " s" : " shortly"}.`, unauthorized: "Your token is no longer accepted.",
    invalid_request: e.message, upload_incomplete: "An upload did not arrive intact; try again.", forbidden: e.message,
    // a job that failed after it started: the server's own sentence says why (and what to close or choose), so it is shown as the reason
    gpu_out_of_memory: e.message || "The GPU ran out of memory. Close other programs that use it, or choose a smaller size, then try again.",
    generation_failed: `The server could not finish this job: ${String(e.message || "").replace(/[.\s]+$/, "")}. Nothing was kept; you can try again.`,
  };
  return known[e.code] || `${e.code}: ${e.message}`;
}

async function poll() {
  if (polling) return; polling = true; busy("generate", true);
  let pause = 600;
  try {
    for (;;) {
      const j = await (await api("/v1/jobs/" + jobId)).json(), p = j.progress || {};
      lastJob = { id: j.id, status: j.status, error: j.error ? `${j.error.code}: ${j.error.message}` : "" }; // for the Report: what became of the last job
      const knownProgress = j.status === "running" && Number.isFinite(p.fraction) && p.fraction > 0;
      $("bar").classList.toggle("indeterminate", !knownProgress);
      $("bar").style.width = knownProgress ? Math.round(p.fraction * 100) + "%" : "0";
      const where = {
        queued: j.queue_position ? `Queued at position ${j.queue_position}…` : "Waiting in the queue…",
        loading_model: "Loading the model…",
        running: p.total_steps ? `Generating, step ${p.step || 0} of ${p.total_steps}…` : "Generating…",
      }[j.status];
      if (where && $("jobline").textContent !== where) say($("jobline"), where);
      if (j.status === "succeeded") { await show(j); break; }
      if (j.status === "failed") { say($("jobline"), explain(j.error || { code: "failed", message: "failed" }), "err"); break; }
      if (j.status === "cancelled") { say($("jobline"), "Cancelled."); break; }
      pause = Math.min(pause * 1.25, 3000);  // a clip takes many minutes: start quick (a fast job is seen at once), then ask less often
      await new Promise(r => setTimeout(r, pause));
    }
  } catch (e) { say($("jobline"), explain(e), "err"); }
  polling = false; busy("generate", false); $("go").disabled = false; $("cancel").hidden = true; $("barwrap").hidden = true;
}

async function show(j) {
  say($("jobline"), "Loading the result…");
  const r = j.results[0], blob = await (await api(`/v1/jobs/${j.id}/result`)).blob(), url = URL.createObjectURL(blob);
  const image = $("img"), video = $("vid"), download = $("dl");
  const previous = download.getAttribute("href");
  if (previous) URL.revokeObjectURL(previous);
  image.hidden = r.kind !== "image";
  video.hidden = r.kind !== "video";
  if (!image.hidden) {
    image.width = r.width || 1024;
    image.height = r.height || 1024;
  }
  (image.hidden ? video : image).src = url;
  download.href = url;
  download.download = `${j.id}.${r.media_type.split("/")[1]}`;
  const label = (currentModel() && currentModel().license && currentModel().license.required_ui_label) || "";  // a licence duty: shown with the result too
  const format = new Intl.NumberFormat(document.documentElement.lang, { maximumFractionDigits: 0 });
  const info = [r.width && r.height ? `${r.width} × ${r.height}` : null, `${format.format(r.bytes / 1024)} KiB`];
  if (r.kind === "video") info.push(`${r.duration_s} s`, `${r.fps} fps`, r.has_audio ? "With sound" : "No sound");
  if (r.seed !== undefined) info.push("Seed " + r.seed);
  if (r.loras?.length) info.push("LoRA " + r.loras.map(l => l.id + "@" + l.scale).join(" + "));
  if (label) info.push(label);
  $("outinfo").textContent = info.filter(Boolean).join(" · ");
  $("out").hidden = false;
  say($("jobline"), "Result ready to preview and download.", "ok");
}
