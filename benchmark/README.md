# ASR-API Benchmark

Benchmark script to test ASR-API performance with multiple transcription requests.

## Setup

1. Install dependencies with uv:
```bash
# From project root
uv sync --extra benchmark

# Or install directly in benchmark directory
cd benchmark
uv pip install -r requirements.txt
```

2. Add audio files to `data/` directory:
   - Supported formats: `.wav`, `.mp3`, `.m4a`
   - Example files are provided in `data/` directory

## Usage

### Basic usage
```bash
# Install only benchmark dependencies (requests)
uv pip install requests

# Run benchmark directly with python (recommended - avoids installing CUDA dependencies)
python3 benchmark/benchmark.py --number-of-runs 10

# Or using uv run (installs ALL project dependencies including CUDA packages)
uv run benchmark/benchmark.py --number-of-runs 10
```

### With custom API URL
```bash
uv run benchmark/benchmark.py --number-of-runs 20 --api-url http://localhost:9000
```

### Concurrent requests
```bash
uv run benchmark/benchmark.py --number-of-runs 50 --concurrent 5
```

### All options
```bash
uv run benchmark/benchmark.py \
  --number-of-runs 100 \
  --api-url http://asr.api.k25.local \
  --data-dir data \
  --concurrent 3
```

## Arguments

- `--number-of-runs`: Number of transcription requests to execute (default: 10)
- `--api-url`: ASR-API URL (default: http://asr.api.k25.local)
- `--data-dir`: Directory containing audio files (default: data)
- `--concurrent`: Number of concurrent requests (default: 1)

## Output

The script provides:
- Real-time progress of each request
- Summary statistics:
  - Total runs, successful/failed counts
  - Throughput (requests/sec)
  - Latency statistics (avg, median, min, max, stddev)
  - Percentiles (P50, P95, P99)

## Example Output

```
📁 Found 1 audio file(s) in data
🔄 Running 10 transcription(s)...
🌐 API URL: http://asr.api.k25.local
⚡ Concurrent requests: 1
------------------------------------------------------------
✅ Run 1/10: example.wav - 2.34s
✅ Run 2/10: example.wav - 2.28s
...
------------------------------------------------------------
📊 BENCHMARK SUMMARY
------------------------------------------------------------
Total runs: 10
Successful: 10
Failed: 0
Total time: 23.45s
Throughput: 0.43 requests/sec

⏱️  LATENCY STATISTICS (successful requests)
  Average: 2.35s
  Median:  2.30s
  Min:     2.15s
  Max:     2.50s
  StdDev:  0.10s
  P50:     2.30s
  P95:     2.48s
  P99:     2.50s
```

