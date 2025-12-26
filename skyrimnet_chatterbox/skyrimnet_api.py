
#!/usr/bin/env python3
"""
SkyrimNet Simplified FastAPI TTS Service
"""
from argparse import ArgumentParser
import sys
from pathlib import Path
from loguru import logger
import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path
import random
import os
from time import perf_counter_ns


# Handle PyInstaller frozen environment
if getattr(sys, 'frozen', False):
    # Running as compiled PyInstaller executable
    bundle_dir = Path(sys._MEIPASS)
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
else:
    # Running from source - add directories to Python path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    echo_tts_dir = parent_dir / "echo_tts"
    
    # Add current dir for sibling imports (shared_*.py)
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    # Add echo_tts dir so its internal imports work (autoencoder, model, etc.)
    if str(echo_tts_dir) not in sys.path:
        sys.path.insert(0, str(echo_tts_dir))

# Standard library imports
import tempfile
from typing import Optional
#import numpy as np
# Third-party imports
import uvicorn
import time
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

# Local imports - Handle both direct execution and module execution
try:
    from cache_utils import (
        load_conditionals_cache,
        save_conditionals_cache,
        get_cache_key,
        save_torchaudio_wav,
        init_conditional_memory_cache,
        clear_output_directories,
        clear_cache_files
    )
    from chatterbox.shared_utils import validate_language_id
    from chatterbox.tensor_utils import initialize_model_dtype, safe_conditional_to_dtype
    from shared_config import get_tts_params, DEFAULT_CACHE_CONFIG, DEFAULT_TTS_PARAMS, SUPPORTED_LANGUAGE_CODES
except ImportError:
    from .cache_utils import (
        load_conditionals_cache,
        save_conditionals_cache,
        get_cache_key,
        save_torchaudio_wav,
        init_conditional_memory_cache,
        clear_output_directories,
        clear_cache_files
    )
    from .chatterbox.shared_utils import validate_language_id
    from .chatterbox.tensor_utils import initialize_model_dtype, safe_conditional_to_dtype
    from .shared_config import get_tts_params, DEFAULT_CACHE_CONFIG, DEFAULT_TTS_PARAMS, SUPPORTED_LANGUAGE_CODES


# =============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# Global model state
CURRENT_MODEL = None
IGNORE_PING = False
CACHED_TEMP_DIR = None
SILENCE_AUDIO_PATH = "assets/silence_100ms.wav"
_CONFIG_CACHE = None
_CONFIG_FILE_PATH = "skyrimnet_config.txt"
CFG_VALUE = 2.1
INFERENCE_TIMESTEPS = 5
# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================
def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true',help="Create a EXTERNAL facing public link using Gradio's servers")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=False, default=7860, help="Port to run the server on (default: 7860)")
    parser.add_argument("--inbrowser", action='store_true', help="Open the UI in a new browser window")
    parser.add_argument("--multilingual", action='store_true', default=False, help="Use the multilingual model (requires more VRAM)")
    parser.add_argument("--clearoutput", action='store_true', help="Remove all folders in audio output directory and exit")
    parser.add_argument("--clearcache", action='store_true', help="Remove all .pt cache files and exit")
    return parser.parse_args()
args = parse_arguments()


# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cached_temp_dir():
    """Get or create the cached temporary directory"""
    global CACHED_TEMP_DIR
    
    if CACHED_TEMP_DIR is None:
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Created cached temp directory: {CACHED_TEMP_DIR}")
    elif not CACHED_TEMP_DIR.exists():
        # Recreate if it was somehow deleted
        CACHED_TEMP_DIR = Path(tempfile.mkdtemp(prefix="skyrimnet_tts_"))
        logger.info(f"Recreated cached temp directory: {CACHED_TEMP_DIR}")
    
    return CACHED_TEMP_DIR

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(title="SkyrimNet TTS API", description="Simplified TTS API service", version="1.0.0")

# Request logging middleware (logs ALL requests, even undefined endpoints)
#@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the incoming request
    logger.info(f"📥 INCOMING REQUEST: {request.method} {request.url}")
    logger.info(f"   Headers: {dict(request.headers)}")
    logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
    
    # Log query parameters if any
    if request.query_params:
        logger.info(f"   Query params: {dict(request.query_params)}")
    
    # Try to log request body for POST requests (be careful with large files)
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                # For JSON requests, we can log the body
                body = await request.body()
                if len(body) < 2000:  # Only log small bodies
                    logger.info(f"   Body: {body.decode('utf-8')}")
                else:
                    logger.info(f"   Body: <large body {len(body)} bytes>")
            elif "multipart/form-data" in content_type:
                logger.info(f"   Body: <multipart form data>")
            else:
                logger.info(f"   Body: <{content_type}>")
        except Exception as e:
            logger.warning(f"   Body: <failed to read body: {e}>")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log the response
        logger.info(f"📤 RESPONSE: {response.status_code} for {request.method} {request.url.path}")
        logger.info(f"   Processing time: {process_time:.4f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ REQUEST FAILED: {request.method} {request.url.path}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Processing time: {process_time:.4f}s")
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ENDPOINTS
# =============================================================================


##@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS audio from text with specified speaker voice
    """
    global IGNORE_PING
    try:
        logger.info(f"Post tts_to_audio - Processing TTS to audio with request: "
                   f"text='{request.text}' speaker_wav='{request.speaker_wav}' "
                   f"language='{request.language}' accent={request.accent} save_path='{request.save_path}'")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if request.text == "ping":
            if IGNORE_PING is None:
                IGNORE_PING = "pending"
            else:
                logger.info("Ping request received, sending silence audio.")            
                return FileResponse(
                    path=SILENCE_AUDIO_PATH,
                    filename=request.save_path,
                    media_type="audio/wav"
                )
    
        language = request.language
        language, _ = language.split("-") if language is not None and "-" in language else ("en", None)

        # Get latents from speaker
        speaker_wav = request.speaker_wav or "malecommoner"
        text = request.text
        language = request.language or "en"

        wav_tensor = CURRENT_MODEL.generate(
                    **generate_args
                    )

        wav_out_path = str(save_torchaudio_wav(wav_tensor, CURRENT_MODEL.sr, audio_path=audio_prompt_path))
        wav_out_path = generate_audio_file(
            model=CURRENT_MODEL,
            language=language,
            speaker_wav=speaker_wav,
            text=text
        )
                    
        if IGNORE_PING == "pending":
            IGNORE_PING = True
            Path(wav_out_path).unlink(missing_ok=True)
            wav_out_path = SILENCE_AUDIO_PATH      
        return FileResponse(
            path=str(wav_out_path),
            filename=request.save_path,
            media_type="audio/wav"
        )             
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /tts_to_audio - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/create_and_store_latents")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    language: str = Form("en"),
    wav_file: UploadFile = File(...)
):
    setup_model_seed()
    """
    Create and store latent embeddings from uploaded audio file
    """    
    try:
        logger.info(f"POST /create_and_store_latents - Creating and storing latents for speaker: {speaker_name}, language: {language}, file: {wav_file.filename}")
        
        if not CURRENT_MODEL:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        
        # Validate file type
        if not wav_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")
        
        language, _ = language.split("-") if language is not None and "-" in language else ("en", None)

        # Use cached temp directory and create unique filename
        temp_dir = get_cached_temp_dir()
        #unique_filename = f"{uuid.uuid4().hex}_{wav_file.filename}"
        temp_audio_path = temp_dir.joinpath(wav_file.filename)
        
        try:
            with open(temp_audio_path, "wb") as buffer:
                content = await wav_file.read()
                buffer.write(content)            
            
            # Get latents from audio
            get_latent_from_audio(CURRENT_MODEL, language, str(temp_audio_path))
            
            logger.info(f"Successfully created latents for speaker: {speaker_name}")
            
            return {
                "message": f"Latents created and stored for speaker '{speaker_name}'",
                "speaker_name": speaker_name,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")
            
        finally:
            temp_audio_path.unlink(missing_ok=True)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /create_and_store_latents - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": CURRENT_MODEL is not None,
    }

#WHISPER_DEFAULT_SETTINGS = {
##    "whisper_model": "turbo",
#    "temperature": 0.0,
#    "temperature_increment_on_fallback": 0.2,
#    "no_speech_threshold": 0.6,
#    "logprob_threshold": -1.0,
#    "compression_ratio_threshold": 2.4,
#    "condition_on_previous_text": True,
#    "verbose": False,
#    "task": "transcribe",
#}
#def transcribe(audio_path: str, **whisper_args):
#    """Transcribe the audio file using whisper"""
#    global whisper_model
#
#    # Set configs & transcribe
#    if whisper_args["temperature_increment_on_fallback"] is not None:
#        whisper_args["temperature"] = tuple(
#            np.arange(whisper_args["temperature"], 1.0 + 1e-6, whisper_args["temperature_increment_on_fallback"])
#        )
#    else:
#        whisper_args["temperature"] = [whisper_args["temperature"]]
#
#    del whisper_args["temperature_increment_on_fallback"]
#
#    transcript = whisper_model.transcribe(
#        audio_path,
#        **whisper_args,
#    )
#
#    return transcript
#@app.post('/v1/audio/transcriptions')
#async def transcriptions(model: str = Form(...),
#                         file: UploadFile = File(...),
#                         response_format: Optional[str] = Form(None),
#                         language: Optional[str] = Form(None),
#                         prompt: Optional[str] = Form(None),
#                         temperature: Optional[float] = Form(None)):
#
#    model = initialize_whisper_model()
#    if file is None:
#        raise HTTPException(
#            status_code=status.HTTP_400_BAD_REQUEST,
#            detail=f"Bad Request, bad file"
#            )
#    if response_format is None:
#        response_format = 'json'
#    if response_format not in ['json',
#                           'text',
#                           'srt',
#                           'verbose_json',
#                           'vtt']:
#        raise HTTPException(
#            status_code=status.HTTP_400_BAD_REQUEST,
#            detail=f"Bad Request, bad response_format"
#            )
#    if temperature is None:
#        temperature = 0.0
#    if temperature < 0.0 or temperature > 1.0:
#        raise HTTPException(
#            status_code=status.HTTP_400_BAD_REQUEST,
#            detail=f"Bad Request, bad temperature"
#            )
#
#    filename = file.filename
#    fileobj = file.file
#    
#    whisper_tempdir = Path("whisper_temp")
#    whisper_tempdir.mkdir(parents=True, exist_ok=True)
#    upload_name = whisper_tempdir.joinpath(filename)
#    upload_file = open(upload_name, 'wb')
#    shutil.copyfileobj(fileobj, upload_file)
#    upload_file.close()
#
#    settings = WHISPER_DEFAULT_SETTINGS.copy()
#    settings['temperature'] = temperature
#    if language is not None:
#        settings['language'] = language # TODO: check  ISO-639-1  format
#
#    transcript = transcribe(audio_path=upload_name, **settings)
#
#    if upload_name:
#        os.remove(upload_name)
#
#    if response_format in ['text']:
#        return Response(content=transcript['text'], media_type="text/plain")
#
#    if response_format in ['srt']:
#        ret = ""
#        for seg in transcript['segments']:
#            
#            td_s = timedelta(milliseconds=seg["start"]*1000)
#            td_e = timedelta(milliseconds=seg["end"]*1000)
#
#            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02},{td_s.microseconds//1000:03}'
#            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02},{td_e.microseconds//1000:03}'
#
#            ret += '{}\n{} --> {}\n{}\n\n'.format(seg["id"], t_s, t_e, seg["text"].strip())
#        ret += '\n'
#        return Response(content=ret, media_type="text/plain")
#
#    if response_format in ['vtt']:
#        ret = "WEBVTT\n\n"
#        for seg in transcript['segments']:
#            td_s = timedelta(milliseconds=seg["start"]*1000)
#            td_e = timedelta(milliseconds=seg["end"]*1000)
#
#            t_s = f'{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}'
#            t_e = f'{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}'
#
#            ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"].strip())
#        return Response(content=ret, media_type="text/plain")
#
#    if response_format in ['verbose_json']:
#        transcript.setdefault('task', WHISPER_DEFAULT_SETTINGS['task'])
#        transcript.setdefault('duration', transcript['segments'][-1]['end'])
#        if transcript['language'] == 'ja':
#            transcript['language'] = 'japanese'
#        return transcript
#
#    return {'text': transcript['text']}
# =============================================================================
# CATCH-ALL ROUTE CONFIGURATION
# =============================================================================

def setup_catch_all_route():
    """
    Set up catch-all route for undefined API endpoints.
    This should only be called when NOT mounting Gradio UI.
    """
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_endpoints(request: Request, path: str):
        """
        Catch-all route to log attempts to access undefined endpoints
        This helps with debugging missing routes and API discovery
        """
        logger.warning(f"🚫 UNDEFINED ENDPOINT: {request.method} /{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     POST /create_and_store_latents") 
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"Endpoint not found: {request.method} /{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ]
            }
        )


def setup_api_only_catch_all_route():
    """
    Set up a limited catch-all route that only catches API paths when Gradio is mounted.
    This avoids conflicts with Gradio's routing while still providing API endpoint discovery.
    """
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_undefined_api_endpoints(request: Request, path: str):
        """
        Catch-all route for undefined /api/* endpoints only
        This helps with API debugging without interfering with Gradio
        """
        logger.warning(f"🚫 UNDEFINED API ENDPOINT: {request.method} /api/{path}")
        logger.warning(f"   Full URL: {request.url}")
        logger.warning(f"   Available API endpoints:")
        logger.warning(f"     POST /tts_to_audio")
        logger.warning(f"     POST /create_and_store_latents") 
        logger.warning(f"     GET  /health")
        logger.warning(f"     GET  /docs (Swagger UI)")
        logger.warning(f"     GET  /redoc (ReDoc)")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "error": f"API endpoint not found: {request.method} /api/{path}",
                "available_endpoints": [
                    "POST /tts_to_audio",
                    "POST /create_and_store_latents",
                    "GET /health",
                    "GET /docs",
                    "GET /redoc"
                ],
                "note": "For the Gradio UI, visit the root path '/'"
            }
        )

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize application environment
    initialize_application_environment("SkyrimNet TTS API")
    load_skyrimnet_config()
    # Set up full catch-all route for standalone API mode
    setup_catch_all_route()
    
    # Load model with standardized initialization
    try:
        CURRENT_MODEL = initialize_model_with_cache(
            device=args.device,
        )
        #logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    
    try:
        # Warmup TTS model with a short dummy synthesis
        wav_tensor0, _ = generate_audio_tensor_file(model=CURRENT_MODEL, text="Warming up sweetrolls.", language='en', speaker_wav='malecommoner')
        del wav_tensor0  # Free memory
        #wav_out_path = generate_audio_file(
        #    model=CURRENT_MODEL,
        #    language='en',
        #    speaker_wav='malecommoner',
        #    text="Warming up 1 sweetroll."
        #)
        wav_out_path = generate_audio_file(
            model=CURRENT_MODEL,
            language='en',
            speaker_wav='malecommoner',
            text="Drema, keep that chaurus off my back while I bleed this sphere-dog dry—my ward’s cracking and I need one clean breath to drop another flame thrall between us and those bolts."
        )
        wav_out_path = generate_audio_file(
            model=CURRENT_MODEL,
            language='en',
            speaker_wav='malecommoner',
            text="(furious) Drema, keep that chaurus off my back while I bleed this sphere-dog dry—my ward’s cracking and I need one clean breath to drop another flame thrall between us and those bolts."
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to warm up TTS model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    

    # Start server
    logger.info(f"Starting server on {args.server}:{args.port}")
    uvicorn.run(
        app, 
        host=args.server, 
        port=args.port, 
        log_level="info",
        access_log=False,  # Disable uvicorn's access logging to use our format
        log_config=None    # Use default Python logging instead of uvicorn's custom format
    )
