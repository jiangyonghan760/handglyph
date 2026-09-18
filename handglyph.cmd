@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem 切到脚本所在目录 —— 与 handglyph.sh 的 `cd "$(dirname "$0")"` 对齐。
rem 不切的话，在别的目录里双击或调用它，`library`、`backgrounds`、
rem 输出文件都会落在"当前目录"而不是程序目录，用户会找不到产物。
pushd "%~dp0"

rem 找 Python。本机常同时装着好几个 Python，直接取第一个可能挑到没装依赖的那个，
rem 结果"程序明明能用却报缺依赖"。所以分两轮：
rem   第一轮 —— 挑一个能 import numpy/PIL/scipy 的（开箱即用）
rem   第二轮 —— 退而求其次，挑任意可用的，由程序自己提示缺什么怎么装
set PY=
rem 允许用环境变量直接指定解释器（比如本机有个装好依赖的 venv，用它最省事）
if defined HANDGLYPH_PYTHON set PY=%HANDGLYPH_PYTHON%
if not defined PY for %%X in ("py" "python" "python3") do call :pick_deps %%X
if not defined PY for %%X in ("py" "python" "python3") do call :pick_any %%X
if not defined PY (
  echo.
  echo 没找到 Python。请先安装 Python 3.9 或以上版本：
  echo   https://www.python.org/downloads/
  echo 安装时务必勾选 "Add Python to PATH"。
  echo.
  popd
  pause
  exit /b 1
)

rem 版本门槛校验。原来只挑"能 import 依赖的"，不查版本 ——
rem 于是 Python 3.8 也能通过第一轮，然后在程序里以各种莫名其妙的语法错炸掉。
rem 这里提前拦一次，给出明确的一行提示（与 .sh 的判定一致）。
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo 找到的 Python 版本太旧（需要 3.9 或以上）：& "%PY%" -V
  echo 请升级后重试，或用 HANDGLYPH_PYTHON 指定一个较新的解释器。
  echo.
  popd
  pause
  exit /b 1
)

"%PY%" handglyph.py %*
set RC=%errorlevel%
popd
exit /b %RC%

:pick_deps
if defined PY exit /b
%~1 -c "import numpy,PIL,scipy" >nul 2>nul || exit /b
set PY=%~1
exit /b

:pick_any
if defined PY exit /b
%~1 -c "pass" >nul 2>nul || exit /b
set PY=%~1
exit /b
