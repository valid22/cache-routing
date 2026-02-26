import asyncio
import httpx
import time
import statistics

ROUTER_URL = "http://localhost:8000/infer"

# 2 phrases, 3 trials each = 6 requests total.
PHRASES = [
    "Write a short python script to add two numbers. ",
    "Explain the theory of relativity in simple terms. "
]
TRIALS = 3

async def send_request(client, prompt: str, req_id: int):
    print(f"[{req_id}/{len(PHRASES)*TRIALS}] Sending request to router...")
    start_time = time.time()
    try:
        # High timeout because local CPU inference can be slow
        response = await client.post(ROUTER_URL, json={"prompt": prompt}, timeout=120.0)
        response.raise_for_status()
        elapsed = time.time() - start_time
        data = response.json()
        print(f"[{req_id}/{len(PHRASES)*TRIALS}] Completed in {elapsed:.2f}s | "
              f"Worker: {data.get('worker_id')} | Cached: {data.get('cached')}")
        return data, elapsed
    except Exception as e:
        print(f"[{req_id}/{len(PHRASES)*TRIALS}] Request failed: {e}")
        return None, 0

async def run_benchmark():
    prompts = []
    # Create the sequence of requests:
    # Example: phrase 1 trial 1, phrase 2 trial 1, phrase 1 trial 2, etc.
    # Or just all trials for phrase 1, then all for phrase 2. Let's interleave them.
    for trial in range(TRIALS):
        for phrase in PHRASES:
            # We add a trial identifier to the prompt to make it slightly different visually for the user,
            # but the first 20 characters will remain the same.
            prompt = phrase + f" (Trial {trial + 1})"
            prompts.append(prompt)
            
    total_reqs = len(prompts)
    print(f"Starting simplified benchmark with {total_reqs} requests (sequential)...")
    
    results = []
    async with httpx.AsyncClient() as client:
        for i, p in enumerate(prompts):
            res, lat = await send_request(client, p, i+1)
            if res:
                # Add latency into the result dict for easier stat calculation
                res['latency_ms'] = lat * 1000
                results.append(res)
        
    valid_results = [r for r in results if r is not None and "error" not in r]
    
    if not valid_results:
        print("All requests failed.")
        return
        
    latencies = [r["latency_ms"] for r in valid_results]
    hits = sum(1 for r in valid_results if r["cached"])
    
    avg_latency = statistics.mean(latencies)
    hit_rate = (hits / len(valid_results)) * 100
    
    routing_mode = valid_results[0].get("routing_mode", "unknown")
    
    print("\n--- Benchmark Results ---")
    print(f"Routing Mode | Avg Latency (ms) | Cache Hit Rate | Total Requests")
    print("-" * 75)
    print(f"{routing_mode:<12} | {avg_latency:<16.2f} | {hit_rate:.1f}%          | {len(valid_results)}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
