@echo off

echo Starting Chatterbox Gradio Server in a new window...

call ".venv\scripts\activate.bat"

start "Chatterbox" /high python gradio_tts_app.py

call ".venv\Scripts\deactivate.bat"