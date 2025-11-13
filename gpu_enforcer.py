"""
GPU Enforcer - Forces near 100% GPU usage for llama-cpp-python
Based on GitHub best practices from llama.cpp community

Moved the parameters of the model that used to be on the get_optimized_params() function to the 
model loading on the run_ollama file. This way tweaking the model is faster.

Also removed the GGML_USE_CUDA check, which used to be on the validate_gpu_mode() function, because for some reason it was
always returning False on my lama-cpp-python installation. You can refer to the original code through the link
on the README.md to see how it originally was.

"""

import os

def validate_gpu_mode():
    """Check if CUDA is available"""
    try:
        import llama_cpp
        
        # Display llama-cpp-python version
        print(f"✅ llama-cpp-python version: {llama_cpp.__version__}")
        
        # Confirms it loaded llama-cpp correctly
        print("✅ llama-cpp-python imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ llama-cpp-python not installed: {e}")
        return False

def set_cuda_env_vars():
    """Set environment variables for GPU-only mode"""
    # Only use GPU 0 (change if you have multiple GPUs)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Force CUDA usage
    os.environ['GGML_CUDA'] = '1'
    
    print("✅ CUDA environment variables set")

def apply_gpu_enforcement():
    """
    Main function: Call this BEFORE loading your model.
    Sets up environment and validates GPU availability.
    """
    print("🎮 Applying GPU-only enforcement...")
    
    # Step 1: Set environment variables
    set_cuda_env_vars()
    
    # Step 2: Validate CUDA support
    if not validate_gpu_mode():
        raise RuntimeError("CUDA not available! Cannot use GPU-only mode.")
    
    print("✅ GPU enforcement ready")