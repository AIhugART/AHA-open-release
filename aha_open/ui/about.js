"use strict";
// ---- About: what this is, which version, where the data goes, who supports it, and the models with their licences.
// Everything that can change (version, profile, models, licence text) is read from the manifest the server sent; the sentences are fixed.
function renderAbout() {
  $("about-version").textContent = isConnected && manifest
    ? `${manifest.provider.name} ${manifest.provider.version} (profile ${manifest.profile})`
    : "Not connected. Connect your server to see the version it runs.";
  const list = $("about-models"); list.replaceChildren();
  if (!isConnected || !manifest) return;
  for (const mod of manifest.models) {
    const box = document.createElement("div"); box.className = "setup-model";
    const title = document.createElement("strong"); title.textContent = mod.display_name;
    const lic = mod.license || {}, cu = lic.commercial_use;
    const facts = document.createElement("div");
    facts.append(badge("state: " + mod.state, mod.state === "ready" ? "ok" : "warn"));
    if (lic.name) facts.append(badge(lic.name + (cu ? " - commercial use: " + cu : ""), cu === "allowed" ? "ok" : cu === "not_allowed" ? "bad" : "warn"));
    if (lic.required_ui_label) facts.append(badge(lic.required_ui_label, "")); // a licence duty: the label is shown exactly as the model's licence asks
    if (lic.excluded_territories && lic.excluded_territories.length) facts.append(badge("not licensed in: " + lic.excluded_territories.join(", "), "warn"));
    const notes = document.createElement("p"); notes.className = "note"; notes.textContent = (lic.notes || "") + (lic.notes ? " " : "");
    if (lic.url) { const a = document.createElement("a"); a.href = lic.url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = "Read the licence"; notes.append(a); }
    box.append(title, facts, notes); list.append(box);
  }
}
onEnter["about"] = renderAbout;
