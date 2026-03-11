@echo off

rem Set GRADIO_TEMP_DIR to the script root directory
set "GRADIO_TEMP_DIR=%~dp0.gradio-tmp"
if not exist "%GRADIO_TEMP_DIR%" mkdir "%GRADIO_TEMP_DIR%"

call powershell -ExecutionPolicy Bypass -File .\Start_ChatterBox.ps1 -Turbo