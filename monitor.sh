#!/usr/bin/env bash
# Live progress monitor for the koGPT data-prep / training Runs.
# Usage:  ./monitor.sh [RUN_ID] [TARGET_M]
#   RUN_ID   : Vessl run id (default: the current prepare run)
#   TARGET_M : target tokens in millions (default 7000 = 7B)
# Watches `vessl run logs` and draws a live bar with %/ETA.

VESSL="$HOME/.local/share/vessl-cli-venv/bin/vessl"
ID="${1:-369367255126}"
TARGET_M="${2:-7000}"

prev=0; prev_t=$(date +%s); rate=0
echo "monitoring run $ID (target ${TARGET_M}M tokens) — Ctrl+C to stop"
while true; do
  logs=$("$VESSL" run logs "$ID" 2>/dev/null)
  cur=$(echo "$logs" | grep -oE 'train: [0-9]+M' | tail -1 | grep -oE '[0-9]+')
  [ -z "$cur" ] && cur=0
  now=$(date +%s)
  if [ "$cur" -gt "$prev" ] && [ "$prev" -gt 0 ]; then
    dt=$((now - prev_t)); [ "$dt" -lt 1 ] && dt=1
    rate=$(( (cur - prev) * 60 / dt ))     # M tokens per minute
  fi
  prev=$cur; prev_t=$now

  pct=$(( cur * 100 / TARGET_M )); [ "$pct" -gt 100 ] && pct=100
  filled=$(( pct * 30 / 100 )); [ "$filled" -gt 30 ] && filled=30
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((30 - filled)) '' | tr ' ' '-')
  eta=""
  [ "$rate" -gt 0 ] && eta=$(printf ' | ~%dmin left' $(( (TARGET_M - cur) / rate )))
  printf '\r[%s] %3s%%  %5sM/%sM%s      ' "$bar" "$pct" "$cur" "$TARGET_M" "$eta"

  if echo "$logs" | grep -q 'train done\|\[prep\] all done'; then
    printf '\n✅ prepare done — train_ko.bin / val_ko.bin on the kogpt-data volume\n'
    break
  fi
  if echo "$logs" | grep -qiE 'Traceback|Error:|No space left'; then
    printf '\n⚠️  possible error — check: %s run logs %s | tail -40\n' "$VESSL" "$ID"
    break
  fi
  sleep 15
done
