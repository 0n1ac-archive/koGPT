#!/usr/bin/env bash
# Live monitor for the koGPT TRAINING run (step/loss/ppl/tok-s + ETA).
# Usage:  ./monitor_train.sh [RUN_ID] [MAX_STEPS]
#   RUN_ID    : Vessl run id  (default: the current medium run)
#   MAX_STEPS : total steps    (default 14000 = medium 7B)
VESSL="$HOME/.local/share/vessl-cli-venv/bin/vessl"
ID="${1:-369367255213}"
MAX="${2:-14000}"

prev=0; prev_progress_t=0; rate=0; last_report=-1
echo "monitoring training run $ID (max_steps=$MAX) — Ctrl+C to stop"
while true; do
  L=$("$VESSL" run logs "$ID" 2>/dev/null)
  step=$(echo "$L" | grep -oE '\[train\] step +[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  [ -z "$step" ] && step=0
  loss=$(echo "$L" | grep -oE 'loss [0-9.]+'  | tail -1 | grep -oE '[0-9.]+')
  toks=$(echo "$L" | grep -oE '[0-9]+k tok/s'  | tail -1)
  ppl=$(echo  "$L" | grep -oE 'ppl [0-9.]+'    | tail -1 | grep -oE '[0-9.]+')

  now=$(date +%s)
  if [ "$step" -gt "$prev" ]; then
    if [ "$prev_progress_t" -gt 0 ]; then
      dt=$((now - prev_progress_t)); [ "$dt" -lt 1 ] && dt=1
      rate=$(( (step - prev) * 60 / dt ))          # steps per minute
    fi
    prev_progress_t=$now
  fi
  prev=$step

  pct=$(( step * 100 / MAX )); [ "$pct" -gt 100 ] && pct=100
  filled=$(( pct * 30 / 100 )); [ "$filled" -gt 30 ] && filled=30
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((30 - filled)) '' | tr ' ' '-')
  eta=""
  if [ "$rate" -gt 0 ]; then
    rem=$(( (MAX - step) / rate ))               # minutes left
    eta=$(printf ' | ETA ~%dh%02dm' $((rem / 60)) $((rem % 60)))
  fi
  # Print a new line only when the training log advances. This works in
  # terminals that do not render carriage-return progress bars correctly.
  if [ "$step" -ne "$last_report" ]; then
    printf '[%s] %3s%%  step %s/%s | loss %s | ppl %s | %s%s\n' \
           "$bar" "$pct" "$step" "$MAX" "${loss:-–}" "${ppl:-–}" "${toks:-–}" "$eta"
    last_report=$step
  fi

  if echo "$L" | grep -q 'stopped at step'; then
    printf '\n✅ training finished. best.pt / latest.pt are on the kogpt-ckpt volume.\n'; break
  fi
  if echo "$L" | grep -qiE 'Traceback|CUDA out of memory|Error:|No such file'; then
    printf '\n⚠️  error detected — %s run logs %s | tail -40\n' "$VESSL" "$ID"; break
  fi
  sleep 30
done
