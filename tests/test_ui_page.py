"""Static checks of the web page (aha_open/ui/index.html) that a real browser run showed to be needed (plan 2026-11-24_minimax_h3_plan.md, M4).

The page is plain HTML + one script, so a missing element is not caught at build time: the script fails in the browser, at the moment
the person clicks. Reading the page as the browser would is cheap and catches it here.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "aha_open" / "ui"
HTML = (UI / "index.html").read_text(encoding="utf-8")
# The script lives in one file per menu section (plan 2026-11-25_ui_menu_report_about_plan.md, D5). The checks below read the page the way
# the browser assembles it: the HTML, then the scripts in the order the page loads them.
SCRIPTS = re.findall(r'<script src="\./([A-Za-z0-9_]+\.js)"></script>', HTML)
PAGE = HTML + chr(10).join((UI / name).read_text(encoding="utf-8") for name in SCRIPTS)
CSS = (UI / "theme.css").read_text(encoding="utf-8")


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.external: list[str] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id":
                self.ids.append(value)
            if key in ("src", "href") and value and re.match(r"(https?:)?//", value) and tag in ("script", "link", "img", "iframe"):
                self.external.append(value)


PARSED = _Ids()
PARSED.feed(HTML)


def test_every_element_the_script_asks_for_by_id_exists_in_the_page():
    asked = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', PAGE)) | set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', PAGE))
    assert len(asked) > 30  # the pattern still matches how the script is written, so this check is not empty
    assert sorted(asked - set(PARSED.ids)) == []


def test_no_id_is_used_twice_because_the_second_one_is_never_found():
    duplicated = sorted({i for i in PARSED.ids if PARSED.ids.count(i) > 1})
    assert duplicated == []


def test_the_video_form_has_the_controls_for_a_model_that_declares_exact_sizes():
    for needed in ("vsize", "vsizerow", "vratio", "vedge", "vdur", "vseed", "vnote"):
        assert needed in PARSED.ids, needed
    # the size select has a row of its own: in a shared row the option text (which names the tier) was cut off in a real browser
    assert re.search(r'id="vsizerow"[^>]*class="wide"|class="wide"[^>]*id="vsizerow"', PAGE)
    assert re.search(r"\.row\s*>\s*\.wide\s*\{[^}]*flex:\s*1 1 100%", CSS)


def test_failures_after_a_job_started_become_sentences_not_codes():
    explain = PAGE[PAGE.index("function explain"): PAGE.index("async function poll")]
    for code in ("gpu_out_of_memory", "generation_failed", "model_not_installed", "unsupported_hardware", "busy"):
        assert code + ":" in explain, code
    assert "Nothing was kept" in explain  # a failed job leaves no result: say so


def test_a_licence_duty_label_is_shown_with_the_result_too():
    show = PAGE[PAGE.index("async function show"): PAGE.index('$("save").onclick')]
    assert "required_ui_label" in show


def test_the_page_loads_nothing_from_another_site():
    assert PARSED.external == []  # "nothing leaves this server": no script, stylesheet, image or frame from outside


def test_a_licence_that_excludes_territories_gets_its_own_checkbox_and_the_page_sends_the_confirmation():
    """The Download panel never folds the territory question into the licence box: the server refuses without `confirm_territory: true`."""
    setup = PAGE[PAGE.index("function renderSetup"): PAGE.index("function showInstall")]
    assert "requires_territory_confirmation" in setup and "excluded_territories" in setup
    assert "I confirm that I am not in" in setup and "where this licence does not allow use" in setup
    assert "confirm_territory" in setup and "placeTick.checked" in setup
    # the button needs BOTH boxes while a territory box exists, before and after a download attempt
    assert setup.count("placeTick !== null && !placeTick.checked") == 1
    assert "ui.placeTick !== null && !ui.placeTick.checked" in PAGE[PAGE.index("function showInstall"):]


def test_the_download_panel_says_nothing_that_is_only_true_for_one_model():
    shown = PAGE[PAGE.index("const PHASE_TEXT"): PAGE.index("function badge")]
    assert "image engine" not in shown and "create images now" not in shown


# ---------------------------------------------------------------------------------------------- the Models panel (plan Q4)


MODELS = PAGE[PAGE.index("// ---- Models: choose, download and delete variants"): PAGE.index("function badge")]


def test_the_models_panel_is_loaded_with_the_other_admin_panels_and_hides_itself_off_the_local_machine():
    connect = PAGE[PAGE.index("async function connect"): PAGE.index("function fillModels")]
    assert "loadModels();" in connect
    load = MODELS[MODELS.index("async function loadModels"): MODELS.index("function renderModel")]
    assert '$("models").hidden = true; return;' in load  # a 404 (internet profile, foreign host) leaves nothing to show


def test_deleting_always_asks_first_and_names_what_it_frees():
    delete = MODELS[MODELS.index('act("Delete"'): MODELS.index("row.append(head")]
    assert delete.index("confirm(") < delete.index('method: "DELETE"')
    assert "This frees about ${v.own_gib} GB" in delete and "download it again later" in delete  # its own files, not the shared ones
    assert "v.disk_gib" not in delete
    assert "res.shared_offer === true" in delete and "--include-shared" not in MODELS.split("function offerShared")[0]  # the page never shows a command-line flag
    shared = MODELS[MODELS.index("function offerShared"):]
    assert shared.index("confirm(") < shared.index("include_shared=true")  # the shared files get their own question


def test_a_download_from_the_panel_needs_the_licence_box_and_sends_the_variant_and_the_territory_answer():
    sync = MODELS[MODELS.index("const sync"): MODELS.index("const done")]
    assert "!tick.checked" in sync and "placeTick !== null && !placeTick.checked" in sync
    start = MODELS[MODELS.index('rows.addEventListener("download"'): MODELS.index("sync();\n  if (mod.install.state")]
    assert "accept_license: lic.key" in start and "variant: ev.detail" in start and "confirm_territory = placeTick.checked" in start


def test_the_first_download_still_needs_one_tick_and_one_click_and_can_pick_a_variant():
    setup = PAGE[PAGE.index("function renderSetup"): PAGE.index("function showInstall")]
    assert "body.variant = pick.value" in setup and "v.recommended" in setup  # the recommended one is already selected
    assert setup.count('type = "checkbox"') == 2  # the licence box and the territory box: nothing was added to the first run


def test_server_text_reaches_the_models_panel_only_as_text():
    assert "innerHTML" not in MODELS and "insertAdjacentHTML" not in MODELS and "outerHTML" not in MODELS
    for key in ("label", "quality_note", "message"):
        assert f"v.{key}" in MODELS or f"res.{key}" in MODELS or key in MODELS


def test_two_cards_for_the_same_model_do_not_share_a_polling_timer():
    assert "const timerKey = mod.id + (ui.key" in PAGE and PAGE.count("installTimers[mod.id]") == 0
    assert 'key: ":models"' in MODELS


def test_the_news_about_a_switch_or_a_delete_stays_after_the_cards_are_drawn_again():
    done = MODELS[MODELS.index("const done"): MODELS.index("for (const v of mod.variants)")]
    assert done.index("loadModels()") < done.index('say($("modelsmsg")')


def test_a_second_variant_that_finished_downloading_is_not_called_a_model_that_is_not_ready():
    """Seen in a real run: the card said "not ready yet: ready." because the check assumed the download is what makes the model work."""
    after = PAGE[PAGE.index("function showInstall"): PAGE.index("// ---- Models: choose")]
    second_look = after[after.index("refreshedFor[mod.id] === s.finished_at"):]
    assert second_look.index('mod.state === "ready"') < second_look.index("is not ready yet")
    assert "Press Use this to switch to it." in second_look


def test_the_buttons_of_a_card_are_updated_whenever_the_status_of_its_download_changes():
    """Seen in a real run: after Cancel, Use and Delete stayed disabled because nothing called sync() when the status left "running"."""
    show = PAGE[PAGE.index("function showInstall"): PAGE.index("// ---- Models: choose")]
    assert "if (ui.sync) ui.sync();" in show and show.index("ui.sync()") < show.index("clearTimeout")  # on every call, before the early return of a running install
    assert "ui.sync = sync;" in MODELS


# ---------------------------------------------------------------------------------------------- the menu (plan 2026-11-25_ui_menu_report_about_plan.md, U1)

MENU_ROUTES = {"generate": "page-generate", "settings/connect": "page-connect", "settings/download": "page-download",
               "settings/models": "page-models", "report": "page-report", "about": "page-about"}


def test_every_script_in_the_folder_is_loaded_by_the_page_and_the_order_keeps_boot_last():
    on_disk = sorted(p.name for p in UI.glob("*.js"))
    assert sorted(SCRIPTS) == on_disk  # a script that is not loaded is dead code; a loaded one that is missing is a 404 in the browser
    assert SCRIPTS[0] == "app.js" and SCRIPTS[-1] == "main.js"  # the helpers first, the wiring that needs every function last


def test_the_menu_has_the_four_entries_and_the_three_settings_entries_with_the_agreed_labels():
    nav = HTML[HTML.index('<nav class="site-nav"'): HTML.index("</nav>")]
    assert re.findall(r'data-nav="(\w+)">([^<]+)<', nav) == [("generate", "Generate"), ("settings", "Settings"), ("report", "Report"), ("about", "About")]
    sub = HTML[HTML.index('<nav id="subnav"'): HTML.index("</nav>", HTML.index('<nav id="subnav"'))]
    assert re.findall(r'data-sub="([\w/]+)">([^<]+)<', sub) == [
        ("settings/connect", "Connect Your Server"), ("settings/download", "Download a Model"), ("settings/models", "Manage Models")]


def test_every_menu_address_has_a_section_with_a_heading_and_every_link_in_the_page_goes_to_a_known_address():
    for address, page in MENU_ROUTES.items():
        assert f'"{address}": {{ page: "{page}"' in PAGE, address
        section = HTML[HTML.index(f'id="{page}"'):]
        section = section[: section.index('<div id="page-', 10)] if '<div id="page-' in section[10:] else section
        assert re.search(r"<h[12][ >]", section), page
    for target in re.findall(r'href="#/([^"]*)"', HTML):
        assert target in MENU_ROUTES, target


def test_the_token_in_the_address_is_read_and_removed_before_the_menu_looks_at_the_address():
    """`aha-open open` puts the token in the fragment. The menu also lives in the fragment: if it read the address first, the token would be
    taken for an unknown route, and a token left in the address would reach bookmarks and the history."""
    app = (UI / "app.js").read_text(encoding="utf-8")
    take = app.index("location.hash.match(/token=")
    assert take < app.index('history.replaceState(null, "", location.pathname)') < app.index("function routeOf")
    main = (UI / "main.js").read_text(encoding="utf-8")
    assert main.index("routeOf(location.hash)") < main.index("render(false)") < main.index("if (token) connect();")


def test_a_first_visit_starts_where_the_person_can_act():
    connected = PAGE[PAGE.index("function connected"): PAGE.index("async function connect")]
    assert 'hasReadyModel() ? "generate" : "settings/download"' in connected and '"settings/connect"' in connected
    main = (UI / "main.js").read_text(encoding="utf-8")
    assert "landingPending = true" in main  # with a token the answer waits for the server; an address typed by the person is never replaced
    assert "if (!routeOf(location.hash))" in main


def test_download_and_manage_say_why_they_are_empty_instead_of_vanishing():
    for ident in ("remote-download", "remote-models", "setup-none", "models-none"):
        assert f'id="{ident}"' in HTML, ident
    notes = PAGE[PAGE.index("function adminNotes"): PAGE.index("async function loadSetup")]
    assert "adminOk === false" in notes and "$(\"remote-download\")" in notes and "$(\"models-none\")" in notes
    assert "aha-open open" in HTML  # the sentence tells where to go


def test_a_running_job_or_download_leaves_a_dot_on_its_menu_entry_and_clears_it():
    assert PAGE.count('busy("generate", true)') == 1 and PAGE.count('busy("generate", false)') == 1
    assert 'busy("settings", running)' in PAGE
    assert "a.busy::after" in CSS


def test_the_lora_file_manager_is_in_manage_models_and_the_form_keeps_only_the_choice_of_loras():
    models_page = HTML[HTML.index('id="page-models"'): HTML.index('id="page-report"')]
    assert 'id="loraadmin"' in models_page
    generate_page = HTML[HTML.index('id="page-generate"'): HTML.index('id="page-connect"')]
    assert 'id="loraadmin"' not in generate_page and 'id="lorarefresh"' in generate_page and 'id="loralist"' in generate_page


# ---------------------------------------------------------------------------------------------- the Report section (plan 2026-11-25_ui_menu_report_about_plan.md, U3)

REPORT_JS = (UI / "report.js").read_text(encoding="utf-8")


def test_the_support_contact_is_written_once_and_the_page_fills_it_in():
    assert REPORT_JS.count('SUPPORT_ZALO = "0908329666"') == 1
    assert "0908329666" not in HTML  # the page asks the constant: a second copy would be changed in one place and not the other
    assert 'href="https://zalo.me/' not in HTML and 'zalo.me/" + SUPPORT_ZALO' in REPORT_JS
    assert "data-zalo" in HTML


def test_the_report_never_reads_the_prompt_or_the_images_and_the_token_only_to_remove_it():
    assert '$("prompt")' not in REPORT_JS and ".prompt" not in REPORT_JS and "files" not in REPORT_JS
    scrub = REPORT_JS[REPORT_JS.index("function scrubSecrets"): REPORT_JS.index("function browserSection")]
    assert "split(token)" in scrub  # the token is used to take it OUT of the text
    elsewhere = re.sub(r"//[^\n]*", "", REPORT_JS.replace(scrub, ""))  # code outside that function, comments dropped
    assert not re.search(r"(?:\(|\[|\$\{|=\s|\+\s|,\s)token\b", elsewhere)  # and is not read anywhere else (the word inside a sentence is fine)
    job = PAGE[PAGE.index("lastJob = {"): PAGE.index("lastJob = {") + 160]
    assert "j.id" in job and "j.status" in job and "prompt" not in job and "params" not in job  # what the report keeps of the last job


def test_everything_in_the_report_goes_through_the_secret_removal_and_the_page_errors_hold_no_query():
    assert "scrubSecrets(parts.join(" in REPORT_JS
    assert "aho_" in REPORT_JS and "bearer" in REPORT_JS.lower() and "text.split(token).join" in REPORT_JS
    assert 'path.split("?")[0]' in PAGE  # a LoRA name or any other detail sent in the query is not kept


def test_a_missing_server_part_says_which_cause_it_is_and_what_to_do_next():
    section = REPORT_JS[REPORT_JS.index("async function serverSection"): REPORT_JS.index("async function makeReport")]
    for cause in ("e.status === 404", 'e.code === "unauthorized"', "e.status ?", "could not be reached"):
        assert cause in section, cause
    assert "aha-open report" in section


def test_a_blocked_clipboard_selects_the_text_and_says_so_and_the_file_has_a_dated_name():
    copy = REPORT_JS[REPORT_JS.index("async function copyReport"): REPORT_JS.index("function saveReport")]
    assert "ta.select()" in copy and "Ctrl+C" in copy
    save = REPORT_JS[REPORT_JS.index("function saveReport"):]
    assert "aha-open-report-" in save and "URL.revokeObjectURL" in save


def test_the_report_text_is_shown_for_reading_before_anything_can_be_copied_and_nothing_is_sent():
    for ident in ("reportnote", "reportmake", "reporttext", "reportcopy", "reportsave", "reportremote"):
        assert f'id="{ident}"' in HTML, ident
    assert 'id="reporttext" name="reporttext" class="report-text" readonly' in HTML  # the person reads it; they do not edit what the supporter will see
    assert re.search(r'id="reportout" hidden', HTML)  # the Copy and Save buttons exist only after a report was made
    assert "fetch(" not in REPORT_JS and "sendBeacon" not in REPORT_JS and "XMLHttpRequest" not in REPORT_JS  # the only request is api("/admin/report"), a GET to this server
    assert REPORT_JS.count("api(") == 1 and 'api("/admin/report")' in REPORT_JS


# ---------------------------------------------------------------------------------------------- the About section (plan 2026-11-25_ui_menu_report_about_plan.md, U4)

ABOUT_JS = (UI / "about.js").read_text(encoding="utf-8")


def test_about_takes_the_version_the_models_and_the_licence_label_from_the_manifest_and_shows_server_text_as_text():
    assert "manifest.provider.version" in ABOUT_JS and "manifest.models" in ABOUT_JS and "required_ui_label" in ABOUT_JS
    assert "innerHTML" not in ABOUT_JS and "insertAdjacentHTML" not in ABOUT_JS and "outerHTML" not in ABOUT_JS
    assert 'a.rel = "noopener noreferrer"' in ABOUT_JS  # a licence link from the server opens in another tab without handing over this page


def test_about_states_the_licence_the_privacy_and_the_support_contact_without_a_second_copy_of_the_number():
    about = HTML[HTML.index('id="page-about"'):]
    assert "Apache License 2.0" in about and "no telemetry" in about and "data-zalo" in about
    assert "0908329666" not in about
    links = re.findall(r'href="(https?://[^"]+)"', about)
    assert links == ["https://github.com/AIhugART/AHA-open-release"]  # only <a>: nothing is loaded from it


def test_about_is_drawn_each_time_it_is_shown_and_again_when_the_connection_changes():
    assert 'onEnter["about"] = renderAbout' in ABOUT_JS
    connected = PAGE[PAGE.index("function connected"): PAGE.index("async function connect")]
    assert "onEnter[route]" in connected  # connect() ends in connected(): a page opened on #/about shows the version once it is known


def test_every_script_of_the_page_is_inside_what_the_wheel_packages():
    """`package-data` lists `ui/*`: a script put in a sub-folder would work from a checkout and be missing from the installed program."""
    import fnmatch
    import tomllib

    patterns = tomllib.loads((UI.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]["aha_open"]
    for name in SCRIPTS + ["theme.css", "index.html"]:
        assert any(fnmatch.fnmatch("ui/" + name, pattern) for pattern in patterns), name


def test_generate_says_that_more_models_can_be_downloaded_and_where_and_says_nothing_when_it_cannot():
    """Seen by the product owner: the form lists only installed models, so nobody who has only the image model finds out where video is."""
    generate_page = HTML[HTML.index('id="page-generate"'): HTML.index('id="page-connect"')]
    assert 'id="more-models"' in generate_page
    load = PAGE[PAGE.index("async function loadSetup"): PAGE.index("function renderSetup")]
    assert "offered.map(m => m.display_name)" in load and 'href = "#/settings/download"' in load  # names come from the server's list; the link goes to the download section
    assert '$("more-models").hidden = true; return;' in load  # off the machine itself (admin 404) there is nothing to download from here
    assert "more.hidden = offered.length === 0" in load
    # an installed model that cannot run now, or one this machine cannot take, is never advertised as downloadable
    assert "!(x.variants || []).some(v => v.installed)" in load and "todo.filter(x => !x.blocker)" in load


def test_generate_offers_only_the_video_sizes_the_variant_in_use_can_make():
    js = (UI / "generate.js").read_text(encoding="utf-8")
    assert "fit_by_tier" in js and 'byTier[s.tier] !== "too_small"' in js
    assert "for (const s of fittingSizes(mod).shown) $(\"vsize\").add" in js  # the list is built from the filtered sizes
    assert "Not offered" in js  # a size that is left out is said, with the reason
    assert "none: all.length > 0" in js and "No size of" in js  # nothing fits: said too, never silently offered


def test_variant_costs_are_shown_per_size_for_video_models():
    js = (UI / "settings.js").read_text(encoding="utf-8")
    assert "fit_by_tier" in js and js.count("costOf(v)") >= 3  # the definition, the download pick and the models list
