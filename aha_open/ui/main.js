"use strict";
$("skip-workspace").addEventListener("click", event => {
  event.preventDefault();
  $("workspace").focus();
});
$("save").onclick = connect; $("token").onkeydown = e => { if (e.key === "Enter") connect(); };
$("token").addEventListener("input", () => {
  if (!$("token").hasAttribute("aria-invalid")) return;
  $("token").removeAttribute("aria-invalid");
  say($("status"), "");
});
for (const [input, error] of [[$("refs"), $("referror")], [$("files"), $("fileerror")]]) {
  input.addEventListener("change", () => {
    input.removeAttribute("aria-invalid");
    error.textContent = "";
  });
}
$("model").onchange = onModel; $("task").onchange = onTask; $("go").onclick = generate; $("vsize").onchange = videoNote;
$("cancel").onclick = async () => { try { await api(`/v1/jobs/${jobId}/cancel`, { method: "POST" }); } catch (e) { say($("jobline"), explain(e), "err"); } };
window.addEventListener("hashchange", () => {
  // An address that is no menu entry (a typo, an old bookmark) is replaced by where a first visit would start, not left in the bar.
  if (!routeOf(location.hash)) { navigate(isConnected && hasReadyModel() ? "generate" : "settings/connect", true); return; }
  render(true);
});
// First visit: no address yet. With a token, connect() decides where to start (Generate, Download or Connect); without one the answer is known.
if (!routeOf(location.hash)) { if (token) landingPending = true; else history.replaceState(null, "", "#/settings/connect"); }
render(false);
if (token) connect();
