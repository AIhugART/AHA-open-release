# Third-party licenses

AHA open's own code is Apache-2.0 (see `LICENSE`). It is installed together with the packages below. This list was
generated on 2026-11-23 from the metadata of the packages installed in two clean environments (the gateway as resolved by
`pip install` of the wheel; the worker from its lock file); it is a convenience, not legal advice, and the license text shipped by each package is the authority.
No package is GPL/AGPL/LGPL-licensed or has an unknown license. Re-checked on 2026-11-23 by reading the `License`,
`License-Expression` and classifier fields of every installed package in two environments (28 packages in the gateway's
development environment, which also holds pytest and ruff; 52 in the worker's): none is GPL/AGPL/LGPL and none lacks a
declaration (a few declare only a classifier such as "BSD License"). What this check does not cover: the license *texts*
shipped by each package, and the NVIDIA libraries inside the PyTorch wheel (the wheel's own `LICENSE` file carries NVIDIA
copyright notices and points to NVIDIA's EULA for the CUTLASS Python DSL; the statement below about the CUDA runtime
libraries comes from PyTorch's distribution model and was not re-verified here). Two points need care:

- `certifi` and `tqdm` are MPL-2.0 (file-level copyleft: it applies to those packages' own files, not to AHA open).
- The PyTorch CUDA wheels used by the worker embed NVIDIA CUDA runtime libraries under NVIDIA's own license. AHA open
  does not redistribute them: the operator's machine downloads them from PyTorch when the worker environment is created.

Model weights are not part of this repository and are not listed here; see `docs/05_models.md` for each model's license.

## Gateway (`aha-open`)

| Package | Version | License (as declared in its metadata) |
|---|---|---|
| annotated-doc | 0.0.7 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.16.0 | MIT |
| cffi | 2.1.1 | MIT-0 |
| click | 8.5.1 | BSD-3-Clause |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
| fastapi | 0.165.0 | MIT |
| h11 | 0.16.0 | MIT License |
| idna | 3.20 | BSD-3-Clause |
| opentelemetry-api | 1.45.2 | Apache-2.0 |
| pycparser | 3.12 | BSD-3-Clause |
| pydantic | 2.14.4 | MIT |
| pydantic_core | 2.50.5 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| typing-inspection | 0.4.5 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |

## Qwen-Image worker (`workers/qwen_image`, locked in `workers/qwen_image/uv.lock`)

| Package | Version | License (as declared in its metadata) |
|---|---|---|
| accelerate | 1.15.0 | Apache Software License |
| annotated-doc | 0.0.7 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.16.0 | MIT |
| bitsandbytes | 0.50.2 | MIT |
| certifi | 2026.11.20 | Mozilla Public License 2.0 (MPL 2.0) |
| charset-normalizer | 3.5.3 | MIT |
| click | 8.5.1 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| diffusers | 0.41.0.dev0 | Apache Software License |
| filelock | 4.3.0 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT License |
| hf-xet | 1.8.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD License |
| huggingface_hub | 1.32.0 | Apache Software License |
| idna | 3.20 | BSD-3-Clause |
| importlib_metadata | 10.2.0 | Apache-2.0 |
| jinja2 | 3.1.6 | BSD License |
| lazy-importer | 0.1.0 | Apache-2.0 |
| markdown-it-py | 4.3.0 | MIT License |
| markupsafe | 3.0.4 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT License |
| mpmath | 1.3.0 | BSD License |
| networkx | 3.7 | BSD-3-Clause |
| numpy | 2.2.6 | BSD License |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| peft | 0.22.0 | Apache Software License |
| pillow | 12.2.0 | MIT-CMU |
| psutil | 7.0.0 | BSD License |
| pydantic | 2.14.4 | MIT |
| pydantic_core | 2.50.5 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| pyyaml | 6.0.4 | MIT License |
| regex | 2026.11.19 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache Software License |
| rich | 15.0.0 | MIT License |
| safetensors | 0.8.0 | Apache Software License |
| setuptools | 84.2.1 | MIT |
| shellingham | 1.5.4 | ISC License (ISCL) |
| sympy | 1.14.0 | BSD License |
| tokenizers | 0.23.2 | Apache Software License |
| torch | 2.10.0+cu128 | BSD-3-Clause |
| torchvision | 0.25.0+cu128 | BSD |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 5.17.0 | Apache 2.0 License |
| typer | 0.29.2 | MIT |
| typing-inspection | 0.4.5 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| zipp | 4.1.2 | MIT |
