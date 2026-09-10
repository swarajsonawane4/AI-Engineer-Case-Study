#!/usr/bin/env bash
# Rebuild smart-inquiry-triage.pdf from index.html.
#
# Chrome needs the deck over http rather than file:// for the web fonts to
# load, so this serves the folder on a scratch port, prints, and cleans up.
set -euo pipefail
cd "$(dirname "$0")"

PORT=8799
OUT="smart-inquiry-triage.pdf"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

[ -x "$CHROME" ] || { echo "Chrome not found at $CHROME"; exit 1; }

python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 1

"$CHROME" --headless=new --disable-gpu --no-sandbox \
  --run-all-compositor-stages-before-draw --virtual-time-budget=15000 \
  --no-pdf-header-footer --print-to-pdf="$OUT" \
  "http://localhost:$PORT/index.html" >/dev/null 2>&1

if command -v pdfinfo >/dev/null 2>&1; then
  PAGES=$(pdfinfo "$OUT" | awk '/^Pages:/{print $2}')
  echo "$OUT rebuilt — $PAGES pages"
  [ "$PAGES" = "16" ] || echo "  WARNING: expected 16 pages. A slide is probably overflowing."
else
  echo "$OUT rebuilt"
fi
