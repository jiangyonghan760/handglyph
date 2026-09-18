#!/usr/bin/env bash
# handglyph 启动脚本（macOS / Linux）
# 用法： ./handglyph.sh doctor

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# 找一个可用的 Python 3
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
      PY="$cand"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo ""
  echo "没找到 Python 3.9 或以上版本。"
  echo "  macOS:  brew install python3"
  echo "  Ubuntu: sudo apt install python3 python3-pip"
  echo "  或去 https://www.python.org/downloads/ 下载"
  echo ""
  exit 1
fi

exec "$PY" handglyph.py "$@"
