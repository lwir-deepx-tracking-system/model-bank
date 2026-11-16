#!/bin/bash
# Fan follows the NPU module temperature too (the board fan driver only looks at the CPU temperature).
# Every 5 s: read the hottest NPU core (dxrt-cli -s). Wanted fan value by NPU temperature:
#   below 65 C: nothing (the board decides)   65: 150   72: 200   78 and above: 255
# The value is only ever raised above what the board chose. The board driver writes its own value back within a second,
# so the wanted value is written again every 0.1 s (no process is started for the wait).
P=/sys/class/hwmon/hwmon8/pwm1
want=0
exec 9<> <(:)
while true; do
  t=$(/usr/local/bin/dxrt-cli -s 2>/dev/null | grep -oE "temperature [0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  if [ -n "$t" ]; then
    if   [ "$t" -ge 78 ]; then want=255
    elif [ "$t" -ge 72 ]; then want=200
    elif [ "$t" -ge 65 ]; then want=150
    elif [ "$t" -le 60 ]; then want=0      # 5 C gap before letting go, so it does not flap
    fi
  fi
  for i in {1..50}; do
    if [ "$want" -gt 0 ]; then read -r cur < $P; [ "$cur" -lt "$want" ] && echo $want > $P; fi
    read -r -t 0.1 -u 9
  done
done
