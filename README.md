# Cache-Aware Routing for Distributed Inference (PoC)

🚀 **Benchmark Highlights:**
By switching from traditional Random routing to Cache-Aware (Hash) routing on a 3-worker cluster, we achieved:
- **12.6x Speedup**: Average latency dropped from **75.39ms** to just **5.98ms**!
- **100% Cache Efficiency**: Cache hit rate increased from **83.3%** to **100%**.
- Zero redundant ML compute across the workers for shared prompts.

---

## Problem Statement
In distributed ML inference architectures, identical or highly similar requests (sharing common prefixes) are often naturally load-balanced across multiple worker nodes using random or round-robin routing. As a result, each worker must independently compute and cache identical request segments, leading to redundant work, suboptimal cache hit rates, and higher overall latency. 

## Why Cache Locality Matters
When requests sharing a common prefix (e.g., standard system prompts, shared conversational context) are routed to the same worker, that worker's LRU cache can serve the repeated computation instantly. Consistent-hash routing ensures that specific prefixes are deterministically mapped to specific workers, maximizing cache utility ("cache locality"), drastically reducing expensive ML compute operations, and lowering average latencies.

## Architecture Diagram

```text
       [Client/Benchmark]
               | 
               v
       +------------------+
       |   API Gateway    |  <-- Routes based on hash(prompt[:20])
       |   (Router Node)  |      or randomly.
       +------------------+
          /      |      \
         /       |       \
        v        v        v
  [Worker 1] [Worker 2] [Worker 3]
  (Cache A)  (Cache B)  (Cache C)
    + LLM      + LLM      + LLM
```

## How Routing Affects Cache Hit Rate
- **Random Routing**: A shared prefix has a `1/N` chance of hitting the worker that previously generated it. Cache hit rate scales poorly as cluster size `N` increases.
- **Cache-Aware (Hash) Routing**: A shared prefix has a `100%` chance of hitting the designated worker. Consequently, the cluster-wide cache hit rate approaches the true duplicate request rate of the incoming traffic.

## Benchmark Results
*Results based on sending 6 interleaved requests for 2 distinct phrases:*

| Routing Mode | Avg Latency (ms) | Cache Hit Rate | Total Requests |
|--------------|------------------|----------------|----------------|
| **random**   | 75.39            | 83.3%          | 6              |
| **hash**     | **5.98**         | **100.0%**     | 6              |

## How to Run

### 1. Start the Workers
Use Docker Compose to spin up 3 worker nodes. The workers use a lightweight HuggingFace model (`distilgpt2`) to simulate realistic text generation workloads.

```bash
docker-compose up -d
```

Ensure they are running on ports 8001, 8002, and 8003.

### 2. Start the Router
Install dependencies and run the router locally.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-router.txt
```

Run in **random** mode:
```bash
python router.py --mode=random
```
*(In a separate terminal)*

### 3. Run Benchmark (Random)
With the router running in random mode:
```bash
python benchmark.py
```
Note the metrics. 

### 4. Switch to Hash Routing
Stop the router (`Ctrl+C`), and restart it in **hash** mode:
```bash
python router.py --mode=hash
```

### 5. Run Benchmark (Hash)
Run the benchmark script again:
```bash
python benchmark.py
```
Observe the increased cache hit rate and reduced average and P95 latencies!

### Cleanup 
Tear down the worker nodes:
```bash
docker-compose down
```
