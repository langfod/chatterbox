#!/usr/bin/env python3
"""
Entry point for running skyrimnet_chatterbox as a module:
    python -m skyrimnet_chatterbox

Works both when run directly (python -m skyrimnet_chatterbox) and
when frozen with PyInstaller.
"""

if __name__ == "__main__":
    # Direct import works in both normal Python and PyInstaller frozen environments
    # (runpy.run_module doesn't work when frozen)
    try:
        from skyrimnet_chatterbox import skyrimnet_chatterbox
    except ImportError:
        # Fallback for when running from within the package directory
        import skyrimnet_chatterbox
    
    # The module's if __name__ == "__main__" block won't run on import,
    # so we need to call the main logic explicitly
    import sys
    sys.argv  # Ensure argv is available
    
    # Re-run the module's main block logic
    from argparse import Namespace
    
    # Parse args and run
    args = skyrimnet_chatterbox.parse_arguments()
    
    # Handle cleanup arguments that exit immediately
    if args.clearoutput:
        skyrimnet_chatterbox.logger.info("Clearing output directories...")
        count = skyrimnet_chatterbox.clear_output_directories()
        skyrimnet_chatterbox.logger.info(f"Cleared {count} output directories. Exiting.")
        exit(0)
    
    if args.clearcache:
        skyrimnet_chatterbox.logger.info("Clearing cache files...")
        count = skyrimnet_chatterbox.clear_cache_files()
        skyrimnet_chatterbox.logger.info(f"Cleared {count} cache files. Exiting.")
        exit(0)
    
    # Validate mutually exclusive options
    if args.multilingual and args.turbo:
        skyrimnet_chatterbox.logger.error("Cannot use both --multilingual and --turbo flags together. Turbo only supports English.")
        exit(1)
    
    skyrimnet_chatterbox.MULTILINGUAL = args.multilingual
    skyrimnet_chatterbox.TURBO = args.turbo
    skyrimnet_chatterbox.set_seed(20250527)
    model = skyrimnet_chatterbox.load_model()
    
    # Test generation to warm up model
    w, _ = skyrimnet_chatterbox.generate_audio(text="ping", job_id=42)
    
    skyrimnet_chatterbox.demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
