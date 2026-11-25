#!/usr/bin/env bash
# Event stream for the test2 run: timeline lines, new tracebacks, OOM kills, process memory.
O=/projects/bduw/kytrio/chainuq_overflow
T=$O/logs/test2/timeline.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)/..
declare -A cnt
for f in $(find -L $O/logs/test2 -name "*.log" 2>/dev/null); do cnt[$f]=$(grep -cE "Traceback|CUDA out of memory|Killed" "$f" 2>/dev/null); done
prev=$(cat $T 2>/dev/null); ook=$(grep oom_kill $CG/memory.events | awk '{print $2}'); warned=0
while true; do
  cur=$(cat $T 2>/dev/null)
  comm -13 <(echo "$prev" | sort) <(echo "$cur" | sort) | sed 's/^/STAGE /'
  prev=$cur
  grep -q "test2 repair done" $T && { echo "DONE test2"; exit 0; }
  pgrep -f "jobs_v2/test2_(run|repair).sh" > /dev/null || { echo "ALERT test2_run.sh not running"; exit 0; }
  for f in $(find -L $O/logs/test2 -name "*.log" -mmin -3 2>/dev/null); do
    n=$(grep -cE "Traceback|CUDA out of memory|Killed" "$f" 2>/dev/null)
    if [ "${n:-0}" -gt "${cnt[$f]:-0}" ]; then echo "ERROR new in $f: $(grep -E 'Error|Killed|memory' "$f" | tail -1 | cut -c1-160)"; fi
    cnt[$f]=$n
  done
  k=$(grep oom_kill $CG/memory.events | awk '{print $2}'); [ "$k" -gt "$ook" ] && echo "ALERT cgroup OOM kill (total $k)"; ook=$k
  anon=$(awk '/^anon /{print $2}' $CG/memory.stat)
  if [ "$anon" -gt 90000000000 ]; then [ $warned -eq 0 ] && echo "WARN process (anon) memory $((anon/1000000000)) GB of 120"; warned=1; else warned=0; fi
  sleep 60
done
