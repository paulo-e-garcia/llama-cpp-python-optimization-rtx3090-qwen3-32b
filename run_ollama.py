from llama_cpp import Llama
from gpu_enforcer import apply_gpu_enforcement

# Apply GPU enforcement FIRST
apply_gpu_enforcement()

# Load model with GPU-only settings
# This is optimized to run hf.co/unsloth/Qwen3-32B-GGUF:Q5_K_M on a RTX3090
n_ctx = 7168 # Context Window Size
model = Llama(
    # hf.co/unsloth/Qwen3-32B-GGUF:Q5_K_M
    model_path="/usr/share/ollama/.ollama/models/blobs/sha256-bc96ba6bc5ed7d78540d2ca7fe66234c71b821480496c6ac8d12d8c406429446",
    
    # GPU Settings
    n_gpu_layers=-1,
    offload_kqv=True,
    
    # Context optimization
    n_ctx=n_ctx, 
    n_batch=1024,
    n_ubatch=512,
    
    # Memory Optimization
    use_mmap=True,
    use_mlock=False,
    flash_attn=True,
    
    # Performance tuning
    n_threads=1,
    n_threads_batch=1,  

    # KV cache quantization for more context 
    # This makes my llama-cpp-python install crash, so it's disabled. Your mileage may vary.
    #type_k=4, 
    #type_v=4,

    # Chat format
    verbose=False,
    chat_format="chatml"
)

print("✅ Model loaded on GPU!")

# Starts the input to interact with the model
while True:
    prompt = input('Prompt (type /exit to, well, exit): ')
    if prompt == '/exit':
        break
    
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Provide concise, relevant responses, without gibberish of any kind. Don't show any thinking!!"},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens = n_ctx,
        temperature=0.5, # Medium low so it returns most factual responses.
        top_p=0.5, # On the lower side so it returns conservative responses.
        stop=["<|im_end|>", "<|endoftext|>"]  # Proper stop tokens
    )

    print(response['choices'][0]['message']['content'].replace('<think>\n\n</think>\n\n', '').strip())
    
    