#!/usr/bin/env python3
"""
Entry point for running skyrimnet_chatterbox as a module:
    python -m skyrimnet_chatterbox
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("skyrimnet_chatterbox.skyrimnet_chatterbox", run_name="__main__", alter_sys=True)
