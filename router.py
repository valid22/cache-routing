import argparse
import time
import hashlib
import random
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

# Global configuration that will be set on startup
MODE = "random"
WORKERS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]

# Use a single async client for all requests
http_client = httpx.AsyncClient()

class InferenceRequest(BaseModel):
    prompt: str

def get_worker_url_random():
    return random.choice(WORKERS)

def get_worker_url_hash(prompt: str):
    prefix = prompt[:20]
    # Compute SHA256 of the prefix
    hashed = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
    # Map to worker via: int(hash, 16) % num_workers
    worker_index = int(hashed, 16) % len(WORKERS)
    return WORKERS[worker_index]

@app.post("/infer")
async def infer(req: InferenceRequest):
    start_time = time.time()
    
    if MODE == "random":
        worker_url = get_worker_url_random()
    else:
        worker_url = get_worker_url_hash(req.prompt)
        
    try:
        response = await http_client.post(
            f"{worker_url}/infer",
            json={"prompt": req.prompt},
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        worker_id = data.get("worker_id")
        cached = data.get("cached")
        result = data.get("result")
    except Exception as e:
        return {"error": str(e)}

    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "routing_mode": MODE,
        "worker_id": worker_id,
        "cached": cached,
        "latency_ms": latency_ms,
        "result": result
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "hash"], default="random")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    MODE = args.mode
    print(f"Starting router in {MODE} mode on port {args.port}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
