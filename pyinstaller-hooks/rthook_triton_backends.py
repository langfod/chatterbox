"""
Runtime hook to fix Triton backend discovery in PyInstaller frozen applications.

Triton uses entry_points() to discover backends, which doesn't work in PyInstaller.
This hook patches the entry_points function to return the NVIDIA backend.
"""
import sys

# Only run in frozen environment
if getattr(sys, 'frozen', False):
    print("[Triton Hook] Patching entry_points for Triton backend discovery...")
    
    # Patch importlib.metadata.entry_points to return triton backends
    try:
        if sys.version_info >= (3, 10):
            import importlib.metadata as metadata_module
        else:
            import importlib_metadata as metadata_module
        
        _original_entry_points = metadata_module.entry_points
        
        class FakeEntryPoint:
            """Fake entry point that mimics the real one."""
            def __init__(self, name, value, group):
                self.name = name
                self.value = value
                self.group = group
        
        class FakeTritonEntryPoints:
            """Fake entry points result for Triton backends only."""
            def __init__(self):
                self._eps = [FakeEntryPoint('nvidia', 'triton.backends.nvidia', 'triton.backends')]
            
            def select(self, group=None, name=None):
                result = self._eps
                if group and group != 'triton.backends':
                    return []
                if name:
                    result = [ep for ep in result if ep.name == name]
                return result
            
            def __iter__(self):
                return iter(self._eps)
            
            def __len__(self):
                return len(self._eps)
        
        class EntryPointsWrapper:
            """Wrapper that intercepts .select() calls for triton.backends."""
            def __init__(self, original_result):
                self._original = original_result
            
            def select(self, group=None, name=None):
                # Intercept triton.backends queries
                if group == 'triton.backends':
                    eps = [FakeEntryPoint('nvidia', 'triton.backends.nvidia', 'triton.backends')]
                    if name:
                        eps = [ep for ep in eps if ep.name == name]
                    return eps
                # Pass through to original for everything else
                return self._original.select(group=group, name=name)
            
            def __iter__(self):
                return iter(self._original)
            
            def __len__(self):
                return len(self._original)
            
            def __getattr__(self, name):
                # Forward any other attribute access to the original
                return getattr(self._original, name)
        
        def patched_entry_points(*args, **kwargs):
            """Patched entry_points that includes Triton backends."""
            # Check if this is a direct call for triton.backends (old style API)
            group = kwargs.get('group', None)
            if group == 'triton.backends':
                return FakeTritonEntryPoints()
            
            # Call the original
            try:
                result = _original_entry_points(*args, **kwargs)
            except Exception as e:
                print(f"[Triton Hook] Original entry_points failed: {e}")
                raise
            
            # Wrap the result to intercept .select() calls
            return EntryPointsWrapper(result)
        
        metadata_module.entry_points = patched_entry_points
        print("[Triton Hook] Successfully patched entry_points")
        
    except Exception as e:
        print(f"[Triton Hook] Warning: Failed to patch entry_points: {e}")
        import traceback
        traceback.print_exc()
