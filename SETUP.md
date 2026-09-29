# Setup guide for AI coding agents

This file is written for an AI coding agent (Claude Code or similar) installing
and running this project for a developer. A person can follow it too.

**Agent instructions.** Work through the sections in order. Run the checks,
report what you find, and fix what you can. Where a step is marked **ASK**,
stop and ask the user; do not guess. Never download a model, install a system
package, or touch the user's photo folder without asking first. Explain each
command before you run it if the user has not seen it before.

## 1. What the tool does

It sorts a folder of photos into categories using local AI models served by
[Ollama](https://ollama.com). No photo leaves the user's network.

1. **Phase 1, describe.** A vision model describes every photo and flags
   accidental shots (motion blur, pocket shots, no subject). Output:
   `descriptions_only.json` in the photo folder. Slow: about 4-6 seconds per photo
   on an RTX 4090.
2. **Phase 2, categorize.** A text model assigns every photo to a category, using
   the descriptions only. Minutes, and it can be rerun without Phase 1.
3. **Report.** `image_categories.html` in the photo folder: review photos, move
   them between categories, trash them, and export a script that moves the files
   into one folder per category and/or writes the category into each photo's
   caption. The tool itself never moves, renames or deletes a photo.

Categories persist across runs in a YAML file, each with a rule describing what
belongs in it.

## 2. Requirements

| Need | Details |
|---|---|
| OS | macOS, Linux, or Windows. The exported organize script is bash: on Windows run it in Git Bash or WSL. |
| git, uv | [uv](https://docs.astral.sh/uv/) installs Python 3.12 and all dependencies. |
| Ollama | On this machine or another one on the network, with one vision model and one text model. |
| GPU memory | Only one model is loaded at a time, so the larger of the two models must fit. See section 5. |
| exiftool | Optional. Only for writing categories into photo captions. |

Supported photo formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.heic`,
`.heif`. Videos are skipped. For HEIC photos the report creates JPEG previews in a
hidden `.image-categorizer-previews` folder inside the photo folder.

## 3. Check the machine

Detect the OS and report which tools are present:

```bash
uname -s 2>/dev/null || ver          # Darwin, Linux, or Windows
git --version
uv --version
```

Install what is missing, after asking:

| Tool | macOS | Linux | Windows (PowerShell) |
|---|---|---|---|
| git | `xcode-select --install` or `brew install git` | distro package manager | `winget install Git.Git` |
| uv | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

## 4. Get the code

```bash
git clone https://github.com/DrBenedictPorkins/image-categorizer.git
cd image-categorizer
uv sync
uv run python main.py --help
```

`uv sync` downloads Python 3.12 if needed and installs PyTorch and Transformers
(several GB; only the HuggingFace provider uses them). Run every command in
this guide from the repository root: the report template is loaded by relative
path.

## 5. Ollama and models

**ASK** the user where Ollama runs:

- **This machine.** Check with `ollama --version`. If it fails, install it from
  <https://ollama.com/download> (macOS and Windows installers; on Linux
  `curl -fsSL https://ollama.com/install.sh | sh`). Start it with the app or
  `ollama serve`. Host: `http://localhost:11434`.
- **Another machine** (for example a GPU server). Get its address. The server
  must listen on the network: `OLLAMA_HOST=0.0.0.0:11434` in the Ollama service
  environment on that machine. Host: `http://<server>:11434`.

Check that the server answers and list the models it has. Replace
`http://localhost:11434` with the user's host here and in the capability check
below:

```bash
curl -s -m 10 http://localhost:11434/api/tags | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin)['models']]"
```

Find the GPU memory where Ollama runs: `nvidia-smi` on NVIDIA, `sysctl hw.memsize`
on Apple Silicon (unified memory). For a remote server, **ASK** the user, or
run it over SSH if they allow it (`ssh <server> nvidia-smi`).

| GPU memory | Vision model (Phase 1) | Text model (Phase 2) | Status |
|---|---|---|---|
| 24 GB | `qwen3.6:35b` (23 GB) | `mistral-small3.2:24b` (15 GB) | Tested: 1,195 photos |
| 12-16 GB | `qwen3-vl:8b` (6.1 GB) | `qwen3:14b` (9.3 GB) | Untested |
| 8 GB, or a 16 GB Mac | `qwen3-vl:4b` (3.3 GB) | `qwen3:8b` (5.2 GB) | Untested |

Smaller models describe less accurately and follow the output format less
reliably. **ASK** which pair to use, and say which one was tested.

The Phase 1 model must read images. Check its capabilities (`vision` must be
listed):

```bash
curl -s -m 10 http://localhost:11434/api/show -d '{"model": "qwen3.6:35b"}' | python3 -c "import json,sys; print(json.load(sys.stdin).get('capabilities'))"
```

**ASK** before pulling: each model is a multi-GB download. Pull on the machine
that runs Ollama:

```bash
ollama pull qwen3.6:35b
ollama pull mistral-small3.2:24b
```

## 6. Configure

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

`.env` already holds the tested model pair. Edit `OLLAMA_HOST` to the user's
server, and the two model lines only if the user chose another pair:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.6:35b              # vision model
OLLAMA_TEXT_MODEL=mistral-small3.2:24b
```

Values in `.env` override variables exported in the shell. Leave
`CATEGORIES_FILE` commented out for now: the smoke test below sets it on the
command line, and a value in `.env` would override that. The other optional
settings are described in `.env.example`.

## 7. Check that Python can reach Ollama

`curl` succeeding is not enough, especially on macOS. Test from Python:

```bash
uv run python -c "import os, requests; from dotenv import load_dotenv; load_dotenv('.env', override=True); h = os.environ['OLLAMA_HOST']; print(h, requests.get(h + '/api/tags', timeout=5).status_code)"
```

Expect `200`. If it fails with `No route to host` while `curl` works, see
**macOS Local Network Privacy** in section 11.

## 8. Smoke test on the sample images

`samples/` holds eleven small public-domain images: people, pets, food, places,
a document, a drawing, and two accidental shots. Point the category file at a
temporary path so the test does not create the user's real saved list, and
delete it first so every run starts from no saved categories:

```bash
rm -f /tmp/ic-smoke-categories.yaml
CATEGORIES_FILE=/tmp/ic-smoke-categories.yaml uv run python main.py samples --provider ollama --no-html
```

On Windows PowerShell:

```powershell
Remove-Item "$env:TEMP\ic-smoke-categories.yaml" -ErrorAction SilentlyContinue
$env:CATEGORIES_FILE="$env:TEMP\ic-smoke-categories.yaml"; uv run python main.py samples --provider ollama --no-html
```

Pass criteria:

- The run ends with `Processing complete!` and no tracebacks.
- `samples/categorization_results.json` and `samples/image_categories.html` exist.
- `accidental-motion-blur.jpg` and `accidental-pocket.jpg` are in
  `Accidental Shots`:

```bash
uv run python -c "import json; d = json.load(open('samples/categorization_results.json')); [print(i['filename'], '->', i['primary_category']) for i in d['images']]"
```

Category names vary from run to run; only the two accidental shots are
checked. The whole run takes about a minute on a 24 GB GPU; loading each model the first
time adds up to a minute more. Show the user the `Category list (N):` block from
the output and open the report: `open samples/image_categories.html` (macOS),
`xdg-open` (Linux) or `start` (Windows). The generated files in `samples/` are
ignored by git.

## 9. Run on the user's photos

**ASK** for the photo folder. The tool writes three files into it
(`descriptions_only.json`, `categorization_results.json`,
`image_categories.html`) and does not change any photo. The first Phase 2 run
creates the saved category list at `~/.config/image-categorizer/categories.yaml`
(or `CATEGORIES_FILE`); tell the user.

For large folders, describe once and categorize separately:

```bash
# Phase 1: describe every photo. Progress is saved every 10 photos; rerunning
# the same command resumes and retries failed photos.
uv run python main.py "/path/to/photos" --description-provider ollama --describe-only \
    --init-categories "Portraits,Family Photos,Accidental Shots"

# Optional: build the category list and stop, so the user can edit it first
uv run python main.py "/path/to/photos" --categorization-provider ollama \
    --categorize-from "/path/to/photos/descriptions_only.json" --plan-categories --max-categories 12

# Phase 2: categorize and open the report
uv run python main.py "/path/to/photos" --categorization-provider ollama \
    --categorize-from "/path/to/photos/descriptions_only.json"
```

Phase 1 can take hours for thousands of photos. Run it in the background with
a log (`... > describe.log 2>&1`) and report progress from lines like
`[ 38%] Describing image 460/1195: IMG_0460.JPG`.

- `--init-categories` gives the vision model category hints.
- `--plan-categories` writes the list to the saved category file
  (`~/.config/image-categorizer/categories.yaml` unless `CATEGORIES_FILE` is set)
  and stops. It is plain YAML: a `categories:` list of `name`, `rule` and
  `status` entries. The user can rename, delete, or rewrite rules there.
- `--max-categories` applies to one run only. Pass it again in Phase 2 to keep
  the cap; without it, Phase 2 may add up to five categories for photos that fit
  none of the saved ones.

## 10. Using the report

Tell the user what the report can do and where the results go. Files the
report downloads land in the browser's download folder, usually `~/Downloads`.

- Move photos by dragging or with each photo's category menu; trash and restore;
  rename, create, merge and remove categories; edit each category's rule.
- **Save categories** downloads `categories.yaml`. Moving it to
  `~/.config/image-categorizer/categories.yaml` (or the path in
  `CATEGORIES_FILE`) makes future runs use these categories and rules.
- **Re-sort** on a category downloads `resort.json` for a category that holds
  misfiled photos. Run `uv run python main.py --resort ~/Downloads/resort.json`:
  the text model turns the user's note into rule changes, asks for approval,
  previews 10 photos, then re-sorts that category and rebuilds the report.
- **Export script** downloads `organize_images.sh`: move into folders, write
  `Category: <name>` into each photo's caption, or both. Nothing is deleted;
  trashed photos go to a `Trash` folder. Captions need exiftool
  (`brew install exiftool`, `sudo apt install libimage-exiftool-perl`, or
  <https://exiftool.org> on Windows). Photos app search on Mac and iPhone
  finds captions after import.

Review the downloaded script with the user before running it:
`bash ~/Downloads/organize_images.sh`. It contains the photo folder's absolute
path, so it can be run from any directory, and it asks for a key press before
changing anything.

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No route to host` from Python, `curl` works (macOS) | macOS Local Network Privacy blocks the process. Terminal apps normally have access; processes started by a detached terminal multiplexer server (tmux, herdr) have no app to grant it to. Run from a plain Terminal window, restart the multiplexer server from Terminal, or use an SSH tunnel (`ssh -L 11435:localhost:11434 <server>`, then `OLLAMA_HOST=http://localhost:11435`). |
| `Cannot connect to Ollama` | Server not running, wrong `OLLAMA_HOST`, or a remote server listening only on localhost (set `OLLAMA_HOST=0.0.0.0:11434` in its service environment). |
| `Warning: Model '...' not found` | The model is not pulled on the server: `ollama pull <model>`. |
| Every Phase 1 photo ends as `failed`, `Invalid JSON` | The model is not a vision model, or it is too small to follow the JSON format. Use one from the table in section 5. |
| `off-list category` messages in Phase 2 | Normal: those photos are asked again; photos that stay off-list go to `Unsorted`. |
| Report opens but images are missing | The report must stay in the photo folder; it loads photos by relative name. |
| `Template file not found` | Run from the repository root. |

## 12. Reference

| Option | Meaning |
|---|---|
| `--provider ollama` | Phase 1 and 2 in one run |
| `--description-provider ollama --describe-only` | Phase 1 only |
| `--categorization-provider ollama --categorize-from FILE` | Phase 2 only |
| `--plan-categories` | With `--categorize-from`: build the category list, then stop |
| `--max-categories N` | Cap the number of categories |
| `--init-categories "A,B"` or a file | Category hints for Phase 1 |
| `--resort FILE` / `--yes` | Re-sort one category from the report's export / skip confirmations |
| `--no-html` | Do not open the report in a browser |
| `--model NAME` / `-m` | Override `OLLAMA_MODEL` for one run |
| `-p NAME` | Short for `--provider` |

| Variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2-vision:latest` if unset; set it in `.env` |
| `OLLAMA_TEXT_MODEL` | `llama3.2:latest` if unset; set it in `.env` |
| `OLLAMA_TIMEOUT` | `300` seconds |
| `OLLAMA_MAX_RETRIES` / `OLLAMA_RETRY_DELAY` | `2` / `1.0` seconds, doubled per retry |
| `CATEGORIES_FILE` | `~/.config/image-categorizer/categories.yaml` |
| `MAX_CATEGORIES` | no cap |

The HuggingFace provider (`--provider huggingface`) runs models in-process
without Ollama; see `docs/HUGGINGFACE_PROVIDER.md`. It is less maintained.
