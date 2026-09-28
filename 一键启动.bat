@echo off
setlocal
title AI Football Feedback - VSCode Launcher

echo ============================================
echo   AI Football Feedback - VSCode Launcher
echo ============================================
echo.
echo This script opens VSCode and runs the default Build Task.
echo VSCode task file: .vscode\tasks.json
echo.

set "VSCODE_EXE="
for /f "delims=" %%I in ('where code 2^>nul') do (
    if not defined VSCODE_EXE set "VSCODE_EXE=%%I"
)
if not defined VSCODE_EXE if exist "%LocalAppData%\Programs\Microsoft VS Code\Code.exe" (
    set "VSCODE_EXE=%LocalAppData%\Programs\Microsoft VS Code\Code.exe"
)
if not defined VSCODE_EXE if exist "%ProgramFiles%\Microsoft VS Code\Code.exe" (
    set "VSCODE_EXE=%ProgramFiles%\Microsoft VS Code\Code.exe"
)

if not defined VSCODE_EXE (
    echo [ERROR] VSCode executable was not found.
    echo.
    echo Please start inside VSCode manually:
    echo   1. Open this folder: %~dp0
    echo   2. Press Ctrl + Shift + B
    echo   3. Run the default build task.
    echo.
    echo To enable this .bat launcher:
    echo   1. In VSCode press Ctrl + Shift + P
    echo   2. Search: Shell Command: Install 'code' command in PATH
    echo   3. Restart your terminal, then run this script again.
    echo.
    pause
    exit /b 1
)

echo [1/2] Opening project in VSCode...
"%VSCODE_EXE%" -r "%~dp0."

echo [2/2] Running VSCode default Build Task...
timeout /t 2 /nobreak >nul
"%VSCODE_EXE%" -r "%~dp0." --command workbench.action.tasks.build

echo.
echo If the task does not start automatically, press Ctrl + Shift + B in VSCode.
echo Backend and frontend should run in VSCode integrated terminals.
echo.
pause
