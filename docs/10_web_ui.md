# The web page (`/ui/`)

The page AHA open serves at `http://127.0.0.1:8890/ui/` (open it with `aha-open open`: the token goes in the address fragment, which the browser never sends,
and the page removes it from the address at once). It is plain HTML and a few scripts in `aha_open/ui/`; it only calls `/v1/*` and, on the machine that runs
AHA open, `/admin/*`, like any other client. It loads nothing from another site.

## The menu

One page, one section per menu entry. The address says which (`#/generate`, `#/settings/connect`, ...), so Back, a bookmark and a link to one section work, and a job that
is running or a model that is downloading **keeps going while another section is shown**; the menu entry shows a dot while it runs.

| Menu | Address | What it is |
|---|---|---|
| **Generate** | `#/generate` | Create images (`image.generate`, `image.edit`, with LoRA) and videos. The model, task, prompt, size, seed, steps; the preview and the original file to download. It lists only the models that are installed; when more can be downloaded (a video model, for example) a line at the top names them and links to *Download a Model* |
| **Settings ▸ Connect Your Server** | `#/settings/connect` | The access token. After a successful connection the page goes on by itself |
| **Settings ▸ Download a Model** | `#/settings/download` | Download a model: its licence (and, where a licence excludes territories, its own confirmation box), the variant, progress, **Cancel** |
| **Settings ▸ Manage Models** | `#/settings/models` | Every variant of an installed model (size, measured GPU memory and time, fit on this GPU — for a video model, size by size): **Use this**, **Download**, **Delete**; the LoRA file manager (add, remove) |
| **Report** | `#/report` | Make a support report, read it, copy it or save it (below) |
| **About** | `#/about` | The version and profile the server reports, the licence of AHA open, where the data goes, the support contact, and every model with its licence (a licence's required label, such as "MiniMax H3", is shown exactly as given) |

**Where a first visit starts** (no address in the link): with no token or no server, *Connect Your Server*; connected but no model that works yet, *Download a Model*;
otherwise *Generate*. An address typed or bookmarked is never replaced; one that is not a menu entry is replaced by the starting section.

**On another machine** (the `lan`/`internet` profiles, or any host that is not the machine itself) `/admin/*` answers `404` ([07_security.md](07_security.md)). *Download a
Model* and *Manage Models* then stay in the menu and say so, and say where to go: open the page on the machine that runs AHA open.

## Which variant, which sizes

The server decides what fits from the GPU it reports (its memory, or the operator's `[gpu] memory_limit_gib`) and the measured peak
of each variant. The download picker starts on the recommended variant and says its memory per size; **Manage Models** marks every
variant *fits*, *tight*, *too big* or *not measured*. On **Generate**, a video size that the variant in use cannot make is left out
and a sentence says which and why (another variant may fit it); a size that was not measured is offered.

## Report

For when something does not work and the person who can help is not at the machine. Press **Create report**; the page shows the whole text; then **Copy** or **Save as .txt**.
**Nothing is sent by AHA open**: the person sends it by Zalo (0908329666), email or chat. If the chat says the message is too long, they send the file.

| Part | Comes from | Notes |
|---|---|---|
| What happened | the person's own words (optional) | |
| Machine, Gateway, Models, Logs | `GET /admin/report` (the same text as `aha-open report`), [04_architecture.md](04_architecture.md) section 6.2 | only on the machine that runs AHA open. When the server is stopped, run `aha-open report` instead |
| Web page | the browser: address and section, browser and window, what the server answered (version, profile, each model's state), the last job's id, status and error, the problems the page itself noticed (failed requests with their status code and the server's sentence, script errors) | |

**It does not hold**: the token (any `aho_…` or `Bearer …` is replaced, and so is the exact token the page holds, even if it was pasted into the note), the prompts, the images,
`config.toml`, the database. The data folder is written `<data>` and the home folder `~`. When a part cannot be had, the report says which cause it is (not the machine itself · token
not accepted · the server's own status · server unreachable, maybe stopped) and what to do next.

How to read one (for whoever receives it): first the **Code** line (is the running gateway the code on disk?), then the **time zone** (log lines are local time; the header is UTC), then
the **age** of each log (a log older than the failure is a hypothesis, not evidence), then the end of the logs (a repeated line is shown once with its count).

## For developers

| File | Role |
|---|---|
| `index.html` | The sections and the menu; the scripts are loaded in this order |
| `app.js` | Token, `api()`, connecting, the router (`ROUTES`, `render`, `navigate`), the problems buffer |
| `settings.js` | Download a Model, Manage Models, the LoRA file manager |
| `generate.js` | The Generate form, upload, job polling, preview |
| `report.js` | The Report section; the support contact is the one constant `SUPPORT_ZALO` (About reads it too) |
| `about.js` | The About section, from the manifest |
| `main.js` | Wiring, last: it needs every function above |

The scripts share one global scope (they are not modules), so a function moved between files needs no import. `tests/test_ui_page.py` reads the page the way the browser assembles
it (the HTML, then the scripts in load order) and has a negative test for each rule above.
