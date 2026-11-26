"use strict";
// ---- Report: one text to copy or save and send to whoever supports this installation. Nothing is sent from here.
// The server part (machine, models, end of the logs) comes from /admin/report, which exists only on the machine that runs AHA open.
// The browser part is built here from named fields. The person reads the whole text before it goes anywhere.
const SUPPORT_ZALO = "0908329666"; // the one place the support contact is written; the Report and About sections both read it

function scrubSecrets(text) {
  if (token) text = text.split(token).join("[redacted]"); // the token this page holds, wherever it ended up (a pasted note, an error sentence)
  return text.replace(/\baho_[A-Za-z0-9_-]{6,}/g, "[redacted]").replace(/\bbearer\s+[A-Za-z0-9._~+\/=-]{6,}/gi, "Bearer [redacted]");
}

function browserSection() {
  const lines = ["== Web page ==",
    `Address: ${location.origin}/ui/ (section ${route || "none yet"})`,
    `Browser: ${navigator.userAgent}`,
    `Window: ${innerWidth} x ${innerHeight}, pixel ratio ${devicePixelRatio}, language ${navigator.language}`,
    isConnected && manifest ? `Server answered: ${manifest.provider.name} ${manifest.provider.version}, profile ${manifest.profile}` : "Server answered: no (this page is not connected)"];
  if (manifest) for (const m of manifest.models) lines.push(`Model in the list: ${m.id}, ${m.state}${m.state_detail ? " - " + m.state_detail : ""}`);
  lines.push(lastJob ? `Last job on this page: ${lastJob.id}, ${lastJob.status}${lastJob.error ? ", " + lastJob.error : ""}` : "Last job on this page: none");
  lines.push(pageErrors.length ? "Problems this page noticed (oldest first):" : "Problems this page noticed: none");
  for (const e of pageErrors) lines.push("  " + e);
  return lines.join("\n");
}

// The server's part, or one sentence that says why it is missing: each cause has its own words and its own next step.
async function serverSection() {
  const run = "Run `aha-open report` on the machine that runs AHA open and send its output as well.";
  try { return await (await api("/admin/report")).text(); }
  catch (e) {
    const why = e.status === 404 ? "this page is not running on the machine that runs AHA open, so the server does not offer its logs here."
      : e.code === "unauthorized" ? "the server did not accept the token."
      : e.status ? `the server answered ${e.status} ${e.code}.`
      : `the server could not be reached (${e.message}); it may be stopped.`;
    return `== Server ==\nNot included: ${why}\n${run}`;
  }
}

async function makeReport() {
  const btn = $("reportmake"); btn.disabled = true; say($("reportmsg"), "Collecting...");
  try {
    const note = $("reportnote").value.trim(), parts = [];
    const server = await serverSection();
    if (note) parts.push("== What happened (the person's words) ==\n" + note.slice(0, 2000));
    parts.push(server.trimEnd(), browserSection());
    const text = scrubSecrets(parts.join("\n\n") + "\n");
    $("reporttext").value = text; $("reportout").hidden = false;
    const kib = Math.max(1, Math.round(new Blob([text]).size / 1024));
    say($("reportmsg"), `Report ready: ${kib} KiB. Read it, then copy it or save it as a file.`, "ok");
    $("reporttext").scrollTop = 0;
  } catch (e) { say($("reportmsg"), "Could not build the report: " + e.message, "err"); }
  finally { btn.disabled = false; }
}

async function copyReport() {
  const ta = $("reporttext");
  try { await navigator.clipboard.writeText(ta.value); say($("reportmsg"), "Copied. Paste it into your message.", "ok"); }
  catch (e) { ta.focus(); ta.select(); say($("reportmsg"), "This browser did not allow copying by button. The whole text is selected: press Ctrl+C (or Cmd+C), then paste it.", "err"); }
}

function saveReport() {
  const d = new Date(), pad = n => String(n).padStart(2, "0");
  const name = `aha-open-report-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.txt`;
  const url = URL.createObjectURL(new Blob([$("reporttext").value], { type: "text/plain;charset=utf-8" }));
  const a = document.createElement("a"); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  say($("reportmsg"), `Saved as ${name} in your downloads folder. Send that file.`, "ok");
}

for (const el of document.querySelectorAll("[data-zalo]")) { el.textContent = SUPPORT_ZALO; if (el.tagName === "A") el.href = "https://zalo.me/" + SUPPORT_ZALO; }
$("reportmake").onclick = makeReport; $("reportcopy").onclick = copyReport; $("reportsave").onclick = saveReport;
