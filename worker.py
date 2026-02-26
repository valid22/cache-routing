import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
WORKER_ID = int(os.environ.get("WORKER_ID", 0))

# Initialize a custom LRU Cache back-port using standard dict ordering
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Evict least recently used (first item)
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

lru_cache = LRUCache(100)

logger.info(f"Worker {WORKER_ID} initializing HuggingFace model...")
# Using a small, fast model for PoC
try:
    generator = pipeline("text-generation", model="distilgpt2")
    logger.info(f"Worker {WORKER_ID} model initialized successfully.")
except Exception as e:
    logger.error(f"Worker {WORKER_ID} failed to initialize model: {e}")
    generator = None

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/infer")
def infer(req: InferenceRequest):
    prefix = req.prompt[:20]
    
    cached_result = lru_cache.get(prefix)
    
    if cached_result is not None:
        logger.info(f"Worker {WORKER_ID}: Cache HIT for prefix '{prefix}'")
        cached = True
        result = cached_result
    else:
        logger.info(f"Worker {WORKER_ID}: Cache MISS for prefix '{prefix}'")
        cached = False
        if generator:
            generated = generator(req.prompt, max_new_tokens=20, num_return_sequences=1)
            result = generated[0]['generated_text']
        else:
            result = f"{req.prompt} (simulated fallback generation)"
            
        lru_cache.put(prefix, result)
        
    return {
        "worker_id": WORKER_ID,
        "cached": cached,
        "result": result
    }
