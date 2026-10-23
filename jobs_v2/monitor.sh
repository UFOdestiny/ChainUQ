#!/usr/bin/env bash
# Event stream for the full run: stage changes, new tracebacks, OOM kills, process memory, judge server.
A=/u/kytrio/nvme_bduw/chainuq_artifacts
O=/projects/bduw/kytrio/chainuq_overflow
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)/..
snap() { cat $A/logs/run_everything.txt $A/logs/full_*/timeline.txt 2>/dev/null | sort; }
declare -A cnt
for f in $(find -L $A/logs $O/logs -name "*.log" 2>/dev/null); do cnt[$f]=$(grep -cE "Traceback|CUDA out of memory|Killed" "$f" 2>/dev/null); done
prev=$(snap); fails=0; ook=$(grep oom_kill $CG/memory.events | awk '{print $2}'); warned=0
while true; do
  cur=$(snap)
  comm -13 <(echo "$prev") <(echo "$cur") | sed 's/^/STAGE /'
  prev=$cur
  pgrep -f "jobs_v2/run_everything.sh" > /dev/null || { echo "ALERT run_everything.sh not running"; grep -q "evaluation done" $A/logs/run_everything.txt && echo "DONE all stages"; exit 0; }
  for f in $(find -L $A/logs $O/logs -name "*.log" -mmin -3 2>/dev/null); do
    n=$(grep -cE "Traceback|CUDA out of memory|Killed" "$f" 2>/dev/null)
    if [ "${n:-0}" -gt "${cnt[$f]:-0}" ]; then echo "ERROR new in $f: $(grep -E 'Error|Killed|memory' "$f" | tail -1 | cut -c1-160)"; fi
    cnt[$f]=$n
  done
  k=$(grep oom_kill $CG/memory.events | awk '{print $2}'); [ "$k" -gt "$ook" ] && echo "ALERT cgroup OOM kill (total $k)"; ook=$k
  anon=$(awk '/^anon /{print $2}' $CG/memory.stat)
  if [ "$anon" -gt 90000000000 ]; then [ $warned -eq 0 ] && echo "WARN process (anon) memory $((anon/1000000000)) GB of 120"; warned=1; else warned=0; fi
  if ! grep -q "main run done" $A/logs/run_everything.txt 2>/dev/null; then
    if curl -sf -m 30 localhost:8100/v1/models > /dev/null; then fails=0; else fails=$((fails+1)); [ $fails -eq 2 ] && echo "WARN judge server :8100 unresponsive for 2 checks"; fi
  fi
  sleep 60
done
