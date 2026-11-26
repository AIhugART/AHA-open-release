#!/bin/sh
# Text to image with nothing but curl and a POSIX shell (Linux, macOS, Git Bash on Windows). It shows the raw HTTP calls.
#
#   ./generate.sh "a red fox in the snow" fox.png
#   AHA_OPEN_MODEL=qwen-image-2.1 AHA_OPEN_WIDTH=1024 AHA_OPEN_HEIGHT=1024 ./generate.sh "a red fox"
#
# Settings: AHA_OPEN_URL (default http://127.0.0.1:8890), AHA_OPEN_TOKEN (default: the local token file),
#           AHA_OPEN_MODEL / AHA_OPEN_WIDTH / AHA_OPEN_HEIGHT (default qwen-image-2.1, 1024x1024; see GET /v1/manifest).
# Limits of this script: the prompt must be one line; JSON is read with sed, which is enough for the few simple fields used here
# (use jq or a real JSON parser for anything bigger - see docs/aha_open_api/03_build_a_client.md).
set -eu

PROMPT=${1:?usage: generate.sh "prompt" [out.png]}
OUT=${2:-out.png}
BASE=${AHA_OPEN_URL:-http://127.0.0.1:8890}
MODEL=${AHA_OPEN_MODEL:-qwen-image-2.1}
WIDTH=${AHA_OPEN_WIDTH:-1024}
HEIGHT=${AHA_OPEN_HEIGHT:-1024}

TOKEN=${AHA_OPEN_TOKEN:-}
if [ -z "$TOKEN" ]; then
  if [ -n "${AHA_OPEN_DATA_DIR:-}" ]; then DATA=$AHA_OPEN_DATA_DIR
  elif [ -n "${LOCALAPPDATA:-}" ]; then DATA=$LOCALAPPDATA/AHA-open
  else DATA=${XDG_DATA_HOME:-$HOME/.local/share}/aha-open
  fi
  [ -r "$DATA/local_client.token" ] && TOKEN=$(cat "$DATA/local_client.token")
fi
[ -n "$TOKEN" ] || { echo "error: no token. Set AHA_OPEN_TOKEN (docs/aha_open_api/01_quickstart.md)." >&2; exit 1; }

# The token goes to curl through a pipe (-H @-), so it never appears in the process list. Nothing here prints it.
api() { printf 'Authorization: Bearer %s\n' "$TOKEN" | curl -sS --fail-with-body -H @- "$@"; }
field() { sed -n 's/.*"'"$1"'" *: *"\([^"]*\)".*/\1/p' | head -n 1; }

ESCAPED=$(printf '%s' "$PROMPT" | sed 's/\\/\\\\/g; s/"/\\"/g')
BODY=$(printf '{"task":"image.generate","model":"%s","prompt":"%s","params":{"width":%s,"height":%s},"client_request_id":"curl-%s-%s"}' \
  "$MODEL" "$ESCAPED" "$WIDTH" "$HEIGHT" "$$" "$(date +%s)")

JOB=$(api -X POST -H 'Content-Type: application/json' -d "$BODY" "$BASE/v1/jobs") || { echo "error: $JOB" >&2; exit 1; }
ID=$(printf '%s' "$JOB" | field id)
echo "job $ID accepted" >&2

PAUSE=1
while :; do
  JOB=$(api "$BASE/v1/jobs/$ID") || { echo "error: $JOB" >&2; exit 1; }
  STATUS=$(printf '%s' "$JOB" | field status)
  case $STATUS in
    succeeded) break ;;
    failed|cancelled) echo "error: the job $STATUS: $JOB" >&2; exit 1 ;;
  esac
  sleep "$PAUSE"
done

api -o "$OUT" "$BASE/v1/jobs/$ID/result"
echo "saved $OUT" >&2
