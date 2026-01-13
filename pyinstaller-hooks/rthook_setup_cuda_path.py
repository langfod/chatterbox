"""
Runtime hook to ensure CUDA libraries are available in system PATH.
This hook adds the CUDA installation directory to the system PATH so that
CUDA DLLs can be found at runtime and preloads them before PyTorch.
"""
import os
import sys
import ctypes

def preload_cuda_dlls(cuda_bin):
    """Preload CUDA DLLs from system path to ensure they take precedence over bundled ones."""
    if os.name != 'nt':
        return
    
    # Order matters! Load dependencies first
    cuda_dlls_to_preload = [
        'cudart64_12.dll',      # CUDA Runtime - base dependency
        'cublas64_12.dll',      # Basic Linear Algebra
        'cublasLt64_12.dll',    # cuBLAS Light
        'cufft64_11.dll',       # FFT
        'cufftw64_11.dll',      # FFTW interface
        'cusolver64_11.dll',    # Linear solvers
        'cusparse64_12.dll',    # Sparse matrices
        'curand64_10.dll',      # Random numbers
        'nvrtc64_120_0.dll',    # Runtime compilation
    ]
    
    print("=== Preloading System CUDA DLLs ===")
    for dll_name in cuda_dlls_to_preload:
        dll_path = os.path.join(cuda_bin, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
                print(f"✅ Preloaded: {dll_name}")
            except Exception as e:
                print(f"⚠️ Failed to preload {dll_name}: {e}")
        else:
            print(f"⚠️ Not found for preload: {dll_name}")
    print("=== End Preloading ===\n")

def setup_cuda_path():
    """Add CUDA installation to PATH if available."""
    print("=== CUDA Path Setup Hook ===")
    
    # Try multiple CUDA path environment variables
    cuda_path = None
    for env_var in ['CUDA_PATH_V12_9', 'CUDA_PATH_V12_8', 'CUDA_PATH_V12_6', 'CUDA_PATH']:
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            print(f"{env_var} environment variable: {cuda_path}")
            break
    
    cuda_bin = None
    
    if cuda_path:
        if 'CUDA_HOME' not in os.environ:
            os.environ['CUDA_HOME'] = cuda_path
            print(f"✅ Set CUDA_HOME to: {os.environ['CUDA_HOME']}")
        cuda_bin = os.path.join(cuda_path, 'bin')
        print(f"Expected CUDA bin directory: {cuda_bin}")
        
        if os.path.exists(cuda_bin):
            # Add CUDA bin directory to PATH FIRST (prepend)
            current_path = os.environ.get('PATH', '')
            if cuda_bin not in current_path:
                os.environ['PATH'] = cuda_bin + os.pathsep + current_path
                print(f"✅ Added CUDA bin directory to PATH: {cuda_bin}")
            else:
                # Move it to front if already present
                path_parts = current_path.split(os.pathsep)
                path_parts = [p for p in path_parts if p != cuda_bin]
                os.environ['PATH'] = cuda_bin + os.pathsep + os.pathsep.join(path_parts)
                print(f"✅ CUDA bin directory moved to front of PATH: {cuda_bin}")
            
            # Add as DLL directory for Windows
            if os.name == 'nt':
                try:
                    os.add_dll_directory(cuda_bin)
                    print(f"✅ Added CUDA bin as DLL directory: {cuda_bin}")
                except Exception as e:
                    print(f"⚠️ Could not add DLL directory: {e}")
                
            # List some key CUDA DLLs that should be available
            key_dlls = ['cudart64_12.dll', 'cublas64_12.dll', 'nvrtc64_120_0.dll','cufft64_11.dll','cusparse64_12.dll','cufftw64_11.dll','cusolver64_11.dll']
            for dll in key_dlls:
                dll_path = os.path.join(cuda_bin, dll)
                if os.path.exists(dll_path):
                    print(f"✅ Found CUDA DLL: {dll}")
                else:
                    print(f"❌ Missing CUDA DLL: {dll}")
        else:
            print(f"❌ CUDA bin directory not found: {cuda_bin}")
            cuda_bin = None
    else:
        print("❌ CUDA_PATH environment variable not set.")
        print("Please ensure NVIDIA CUDA toolkit is installed and CUDA_PATH is set.")
        print("Expected: CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x")
    
    # Also add common CUDA locations to PATH as fallback
    fallback_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
    ]
    
    current_path = os.environ.get('PATH', '')
    for fallback_path in fallback_paths:
        if os.path.exists(fallback_path) and fallback_path not in current_path:
            os.environ['PATH'] = fallback_path + os.pathsep + current_path
            print(f"✅ Added fallback CUDA path: {fallback_path}")
            # Also add as DLL directory
            if os.name == 'nt':
                try:
                    os.add_dll_directory(fallback_path)
                except:
                    pass
            if cuda_bin is None:
                cuda_bin = fallback_path
            current_path = os.environ['PATH']
            break
    
    print("=== End CUDA Path Setup ===\n")
    
    # CRITICAL: Preload system CUDA DLLs before PyTorch loads
    if cuda_bin and os.path.exists(cuda_bin):
        preload_cuda_dlls(cuda_bin)

# Call setup function when this hook is loaded
setup_cuda_path()