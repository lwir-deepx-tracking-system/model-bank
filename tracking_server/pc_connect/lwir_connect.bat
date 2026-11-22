@echo off
rem LWIR connect: finds the Orange Pi tracking server by itself and opens its page. Keep this file next to lwir_connect.ps1.
rem   lwir_connect.bat             find the board and open the page
rem   lwir_connect.bat -Install    also open the page whenever the board is plugged in, from Windows start (asks first)
rem   lwir_connect.bat -Uninstall  remove that again
setlocal
if not exist "%~dp0lwir_connect.ps1" (
  echo lwir_connect.ps1 is missing next to this file. Download the zip from the board page and unpack it first.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0lwir_connect.ps1" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%
