# Playbooks: the web page, and a real model

## The web page (`/ui/`) on a sandbox with a browser tool  (✅ run 2026-11-23, fake worker)

1. `run_sandbox_start` (fake worker), then `run_ui_url`. The address carries the sandbox token in the fragment; the token stops
   working when `run_sandbox_stop` deletes the sandbox. Never do this for `live`: `run_ui_url` has no such option on purpose.
2. Navigate a browser tool to that address.
3. **Do not take a page snapshot once the token is in the page** (a snapshot prints the value of a filled password field).
   Read and drive the page with an evaluate call instead: wait until `#make` is not hidden, set `#prompt`, click `#go`, wait until
   `#img` is not hidden, then read `#img.naturalWidth/Height`, `#jobline`, `#outinfo` and the `#dl` link.
4. Check the console for warnings, close the browser, `run_sandbox_stop`.

Measured 2026-11-23: connected to `aha-open 0.1.0.dev0 (profile local)`, models `example-image` and `example-video`, a 256x256 image
shown with a download link, "Done.", 0 console errors or warnings. A background tab does not paint, so clicking through
`element.click()` in the page is used; looking at layout and real mouse clicks remain a human check.

## A real model (`qwen-image-2.1`, INT8)  (✅ run 2026-11-24, RTX 3090 Ti, 31.7 GiB RAM)

Needs about 9.5 GiB of VRAM and 17.6 GiB of free RAM (measured). The tool refuses below 20 GiB free RAM and says the number.

1. Close other programs until `check_harness` / the refusal message shows enough free RAM.
2. `run_sandbox_start(real_model=true, allow_gpu=true)`  (weights from the installed model, `<data dir>/models/qwen-image-2.1/int8`, worker environment
   `workers/qwen_image/.venv`; set `AHA_OPEN_QWEN_WEIGHTS` or `weights_dir` to use another folder).
3. `run_scenario("real_generate_1k_seed_42")`: the first job loads the model (about 90 s cold, measured), then draws. The verdict
   carries evidence level real model, and the result's sha256 is recorded.
4. Run it again: `diff_runs("real_generate_1k_seed_42")` must say "no difference" (the same seed gave byte-identical images
   between two environments on 2026-11-23). A failure: `diagnose_job` with the job id.
5. `run_sandbox_stop`.

Measured 2026-11-24 (scenario `real_generate_1k_seed_42`, seed 42, aspect 1:1, default 40 steps): both runs PASS 8/8 with evidence
real model; the image is 1024x1024; the sha256 of the result is
`c142df9e69e1c1d9539f1eab63c734380dd4dc938b3dd539a53cf4237c69ba45` **in both runs** (so the same seed gave byte-identical output;
`diff_runs` said "no difference"). First run 85 s including the cold load (the worker's own stage lines: ready at +43 s), second run
40 s with the model resident. The worker's RSS peaked at 17.6 GiB and `ram_available` fell to 3.2 GiB at "pipeline ready", so the
machine was close to its limit even with 20 GiB free at the start. The sha256 belongs to this prompt, seed, weights and software
stack on this machine; whether another machine gives the same bytes is not measured.
