# 15. Concurrency & Async Patterns

> Choose the right concurrency model and manage parallel operations safely.

## Overview

MCP tools often perform I/O-bound operations (network calls, file reads, database queries) that benefit from async execution. This section covers when to use async vs sync, how to manage concurrency limits, and patterns for parallel operations.

## Requirements

### MUST

- **Document sync vs async** - in tool descriptions
- **Limit concurrent operations** - prevent resource exhaustion
- **Handle cancellation gracefully** - clean up on cancel
- **Avoid blocking the event loop** - in async contexts

### SHOULD

- **Prefer async for I/O-bound work** - network, file, database
- **Use sync for CPU-bound work** - or offload to thread pool
- **Set concurrency limits per tool** - based on downstream capacity
- **Propagate cancellation** - to child operations

### MAY

- **Support configurable concurrency** - per-request overrides
- **Implement backpressure** - when overwhelmed
- **Use worker pools** - for CPU-intensive operations

## When to Use Async vs Sync

| Operation Type | Recommendation | Reason |
|---------------|----------------|--------|
| HTTP requests | **Async** | Network I/O waits |
| Database queries | **Async** | Network/disk I/O |
| File reads (small) | Either | Minimal difference |
| File reads (large) | **Async** | Disk I/O waits |
| CPU computation | **Sync** (thread pool) | Blocks event loop |
| Subprocess calls | **Async** | Process I/O waits |
| In-memory operations | **Sync** | No I/O involved |

## Async Tool Implementation

### Basic Async Tool

```python
import asyncio
from typing import List

@mcp.tool()
async def fetch_urls(urls: List[str]) -> dict:
    """Fetch multiple URLs concurrently.

    Execution: Async (I/O-bound network operations)
    Concurrency: Up to 10 parallel requests
    """
    results = await fetch_all_with_limit(urls, max_concurrent=10)
    return asdict(success_response(data={"results": results}))
```

### Sync Tool (CPU-bound)

```python
@mcp.tool()
def compute_hash(data: str, algorithm: str = "sha256") -> dict:
    """Compute cryptographic hash of data.

    Execution: Sync (CPU-bound computation)
    """
    import hashlib
    hasher = hashlib.new(algorithm)
    hasher.update(data.encode())
    return asdict(success_response(data={
        "hash": hasher.hexdigest(),
        "algorithm": algorithm
    }))
```

## Concurrency Limiting

### Semaphore Pattern

```python
import asyncio
from typing import List, Any, Callable, TypeVar

T = TypeVar('T')

class ConcurrencyLimiter:
    """Limit concurrent async operations."""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def run(self, coro):
        """Run coroutine with concurrency limit."""
        async with self.semaphore:
            return await coro

    async def gather(
        self,
        coros: List,
        return_exceptions: bool = False
    ) -> List[Any]:
        """Run multiple coroutines with concurrency limit."""
        tasks = [self.run(coro) for coro in coros]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

# Per-tool limiters
TOOL_LIMITERS = {
    "fetch_urls": ConcurrencyLimiter(10),
    "query_database": ConcurrencyLimiter(5),
    "call_external_api": ConcurrencyLimiter(3),
}

async def fetch_all_with_limit(urls: List[str], max_concurrent: int = 10) -> List[dict]:
    """Fetch URLs with concurrency limit."""
    limiter = ConcurrencyLimiter(max_concurrent)

    async def fetch_one(url: str) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    return {
                        "url": url,
                        "status": response.status,
                        "content_length": response.headers.get("content-length")
                    }
        except Exception as e:
            return {"url": url, "error": str(e)}

    return await limiter.gather([fetch_one(url) for url in urls])
```

### Token Bucket Rate Limiter

```python
import asyncio
import time

class TokenBucket:
    """Token bucket for rate limiting async operations."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Returns: Time waited in seconds
        """
        async with self.lock:
            wait_time = 0.0

            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return wait_time

                # Calculate wait time for enough tokens
                needed = tokens - self.tokens
                sleep_time = needed / self.rate
                wait_time += sleep_time
                await asyncio.sleep(sleep_time)

# Usage
rate_limiter = TokenBucket(rate=10.0, capacity=20)  # 10 req/sec, burst of 20

@mcp.tool()
async def rate_limited_operation(data: str) -> dict:
    """Operation with rate limiting."""
    wait_time = await rate_limiter.acquire()
    if wait_time > 0:
        logger.debug(f"Rate limited, waited {wait_time:.2f}s")

    result = await perform_operation(data)
    return asdict(success_response(data=result))
```

## Parallel Operations

### Gather with Error Handling

```python
async def parallel_fetch(items: List[str]) -> dict:
    """Fetch items in parallel, handling individual failures."""

    async def fetch_item(item_id: str) -> dict:
        try:
            result = await external_api.get(item_id)
            return {"id": item_id, "status": "success", "data": result}
        except asyncio.TimeoutError:
            return {"id": item_id, "status": "timeout"}
        except Exception as e:
            return {"id": item_id, "status": "error", "error": str(e)}

    # Execute all in parallel
    results = await asyncio.gather(
        *[fetch_item(item) for item in items],
        return_exceptions=False  # We handle exceptions in fetch_item
    )

    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    return asdict(success_response(
        data={
            "results": succeeded,
            "errors": failed,
            "total": len(items),
            "succeeded": len(succeeded)
        },
        warnings=[f"{len(failed)} items failed"] if failed else None
    ))
```

### Streaming Results

```python
from typing import AsyncIterator

async def stream_results(
    items: List[str],
    max_concurrent: int = 5
) -> AsyncIterator[dict]:
    """Process items and yield results as they complete."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_item(item: str) -> dict:
        async with semaphore:
            return await heavy_processing(item)

    # Create tasks
    tasks = {
        asyncio.create_task(process_item(item)): item
        for item in items
    }

    # Yield results as they complete
    for coro in asyncio.as_completed(tasks.keys()):
        try:
            result = await coro
            yield {"status": "success", "result": result}
        except Exception as e:
            yield {"status": "error", "error": str(e)}

@mcp.tool()
async def process_batch_streaming(items: List[str]) -> dict:
    """Process batch with streaming progress updates."""
    results = []
    async for result in stream_results(items):
        results.append(result)

    return asdict(success_response(data={"results": results}))
```

## CPU-Bound Work in Async Context

### Thread Pool Executor

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Shared thread pool for CPU-bound work
cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu_worker")

async def run_cpu_bound(func, *args, **kwargs):
    """Run CPU-bound function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        cpu_executor,
        partial(func, *args, **kwargs)
    )

def expensive_computation(data: bytes) -> bytes:
    """CPU-intensive compression (sync function)."""
    import zlib
    return zlib.compress(data, level=9)

@mcp.tool()
async def compress_data(data: str) -> dict:
    """Compress data without blocking event loop.

    Execution: Async wrapper around CPU-bound compression
    """
    data_bytes = data.encode()

    # Run in thread pool to avoid blocking
    compressed = await run_cpu_bound(expensive_computation, data_bytes)

    return asdict(success_response(data={
        "original_size": len(data_bytes),
        "compressed_size": len(compressed),
        "ratio": len(compressed) / len(data_bytes)
    }))
```

### Process Pool for Heavy Computation

```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

# Process pool for truly CPU-intensive work
process_executor = ProcessPoolExecutor(max_workers=2)

async def run_in_process(func, *args):
    """Run function in separate process."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(process_executor, func, *args)

def heavy_ml_inference(model_path: str, input_data: dict) -> dict:
    """ML inference that should run in separate process."""
    # This runs in a separate process
    model = load_model(model_path)
    return model.predict(input_data)

@mcp.tool()
async def run_inference(input_data: dict) -> dict:
    """Run ML inference without blocking main process."""
    result = await run_in_process(
        heavy_ml_inference,
        "/models/classifier.pkl",
        input_data
    )
    return asdict(success_response(data={"prediction": result}))
```

## Cancellation Handling

### Graceful Cancellation

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def cancellable_operation():
    """Context manager that handles cancellation gracefully."""
    task = asyncio.current_task()
    try:
        yield
    except asyncio.CancelledError:
        logger.info(f"Operation cancelled: {task.get_name()}")
        # Perform cleanup
        await cleanup_resources()
        raise  # Re-raise to propagate cancellation

async def long_running_task(items: List[str]) -> List[dict]:
    """Long-running task with cancellation checkpoints."""
    results = []

    for i, item in enumerate(items):
        # Check for cancellation periodically
        if i % 10 == 0:
            await asyncio.sleep(0)  # Yield to allow cancellation

        try:
            result = await process_item(item)
            results.append(result)
        except asyncio.CancelledError:
            logger.info(f"Cancelled at item {i}/{len(items)}")
            # Return partial results
            return results

    return results

@mcp.tool()
async def process_with_timeout(items: List[str], timeout: float = 60.0) -> dict:
    """Process items with timeout and partial results."""
    try:
        async with asyncio.timeout(timeout):
            results = await long_running_task(items)
            return asdict(success_response(data={
                "results": results,
                "complete": True
            }))
    except asyncio.TimeoutError:
        # Task was cancelled, but we may have partial results
        return asdict(success_response(
            data={"results": results, "complete": False},
            warnings=[f"Operation timed out after {timeout}s, returning partial results"]
        ))
```

## Async Context Variables

```python
from contextvars import ContextVar
from typing import Optional

# Request-scoped context
request_id: ContextVar[str] = ContextVar("request_id", default="")
client_id: ContextVar[str] = ContextVar("client_id", default="anonymous")
start_time: ContextVar[float] = ContextVar("start_time", default=0.0)

@asynccontextmanager
async def request_context(req_id: str, cli_id: str):
    """Set up request context for async operations."""
    token1 = request_id.set(req_id)
    token2 = client_id.set(cli_id)
    token3 = start_time.set(time.monotonic())

    try:
        yield
    finally:
        request_id.reset(token1)
        client_id.reset(token2)
        start_time.reset(token3)

async def nested_operation():
    """Access context from nested async calls."""
    # Context automatically propagates through async calls
    logger.info(
        "Processing",
        extra={
            "request_id": request_id.get(),
            "client_id": client_id.get(),
            "elapsed_ms": (time.monotonic() - start_time.get()) * 1000
        }
    )

@mcp.tool()
async def main_operation(data: dict) -> dict:
    """Main operation with context propagation."""
    async with request_context(str(uuid.uuid4()), "client_123"):
        result = await nested_operation()
        return asdict(success_response(data=result))
```

## LLM Response Ergonomics for Concurrent Operations

When tools perform concurrent operations, structure responses for optimal LLM consumption:

### Progress and Status Reporting

```python
@mcp.tool()
async def batch_process(items: List[str]) -> dict:
    """Process items in parallel with LLM-friendly progress reporting.

    WHEN TO USE:
    - Processing multiple independent items
    - Need progress visibility during long operations

    Returns structured progress suitable for LLM status checks.
    """
    total = len(items)
    limiter = ConcurrencyLimiter(max_concurrent=5)

    results = await limiter.gather([process_item(item) for item in items])

    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    return asdict(success_response(
        data={
            # Summary first for quick LLM comprehension
            "summary": f"Processed {len(succeeded)}/{total} items successfully",

            # Structured counts for programmatic use
            "counts": {
                "total": total,
                "succeeded": len(succeeded),
                "failed": len(failed),
                "concurrency_limit": 5
            },

            # Results with clear success/failure markers
            "results": succeeded,

            # Errors separate and actionable
            "errors": [
                {"item": r["item"], "reason": r.get("error", "Unknown")}
                for r in failed
            ] if failed else None
        },
        # Surface-level warnings for immediate LLM attention
        warnings=[f"{len(failed)} items failed - see errors array"] if failed else None
    ))
```

### Timeout Context for LLMs

```python
@mcp.tool()
async def query_with_timeout(query: str, timeout_seconds: float = 30.0) -> dict:
    """Execute query with timeout, providing context on timing.

    Returns timing metadata so LLMs can adjust future timeout expectations.
    """
    start = time.monotonic()

    try:
        async with asyncio.timeout(timeout_seconds):
            result = await execute_query(query)
            elapsed = time.monotonic() - start

            return asdict(success_response(
                data={
                    "result": result,
                    "timing": {
                        "elapsed_seconds": round(elapsed, 2),
                        "timeout_seconds": timeout_seconds,
                        "headroom_seconds": round(timeout_seconds - elapsed, 2)
                    }
                },
                # Help LLM understand if timeout is tight
                warnings=[
                    "Query used >80% of timeout budget"
                ] if elapsed > timeout_seconds * 0.8 else None
            ))

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return asdict(error_response(
            message=f"Query timed out after {timeout_seconds}s",
            error_code="TIMEOUT",
            error_type="timeout",
            data={
                "timeout_seconds": timeout_seconds,
                "suggestion": "Consider increasing timeout or simplifying query"
            },
            remediation="Retry with timeout_seconds=60 or break into smaller queries"
        ))
```

### Cancellation Feedback

```python
@mcp.tool()
async def cancellable_operation(
    items: List[str],
    max_items: int = 100
) -> dict:
    """Operation that can be cancelled with partial results.

    Provides clear feedback on cancellation state for LLM understanding.
    """
    processed = []

    for i, item in enumerate(items[:max_items]):
        # Checkpoint every 10 items
        if i % 10 == 0:
            await asyncio.sleep(0)

        try:
            result = await process_item(item)
            processed.append(result)
        except asyncio.CancelledError:
            return asdict(success_response(
                data={
                    "results": processed,
                    "cancellation": {
                        "was_cancelled": True,
                        "processed_before_cancel": len(processed),
                        "total_requested": len(items[:max_items]),
                        "remaining": len(items[:max_items]) - len(processed)
                    }
                },
                warnings=["Operation was cancelled - returning partial results"]
            ))

    return asdict(success_response(
        data={
            "results": processed,
            "cancellation": {
                "was_cancelled": False,
                "processed": len(processed),
                "total_requested": len(items[:max_items])
            }
        }
    ))
```

### Concurrency State Exposure

```python
@mcp.tool()
async def get_concurrency_status() -> dict:
    """Get current concurrency state for system tools.

    WHEN TO USE:
    - Debugging slow operations
    - Understanding system capacity
    - Tuning batch sizes

    Exposes concurrency limits so LLMs can make informed decisions.
    """
    return asdict(success_response(
        data={
            "limits": {
                "http_requests": {
                    "max_concurrent": 10,
                    "description": "Maximum parallel HTTP requests"
                },
                "database_queries": {
                    "max_concurrent": 5,
                    "description": "Maximum parallel database connections"
                },
                "file_operations": {
                    "max_concurrent": 20,
                    "description": "Maximum parallel file I/O operations"
                }
            },
            "recommendations": {
                "small_batch": "For <10 items, concurrency overhead may exceed benefit",
                "large_batch": "For >100 items, consider pagination to avoid timeouts",
                "mixed_operations": "Avoid mixing slow and fast operations in same batch"
            }
        }
    ))
```

### Response Chunking for Large Results

```python
@mcp.tool()
async def process_large_dataset(
    dataset_id: str,
    chunk_size: int = 50
) -> dict:
    """Process large dataset with chunked results.

    Returns paginated results to avoid overwhelming LLM context.
    """
    items = await load_dataset(dataset_id)
    total = len(items)

    if total > chunk_size:
        # Process first chunk, provide continuation info
        first_chunk = items[:chunk_size]
        results = await process_batch(first_chunk)

        return asdict(success_response(
            data={
                "results": results,
                "pagination": {
                    "returned": len(results),
                    "total": total,
                    "has_more": True,
                    "next_offset": chunk_size,
                    "chunk_size": chunk_size
                }
            },
            warnings=[
                f"Large dataset ({total} items) - only first {chunk_size} returned. "
                f"Use offset parameter to retrieve more."
            ]
        ))

    # Small dataset - return all
    results = await process_batch(items)
    return asdict(success_response(
        data={
            "results": results,
            "pagination": {
                "returned": len(results),
                "total": total,
                "has_more": False
            }
        }
    ))
```

## Anti-Patterns

### Don't: Block the Event Loop

```python
# Bad: Blocking call in async function
@mcp.tool()
async def bad_async_tool(url: str):
    response = requests.get(url)  # Blocks event loop!
    return response.json()

# Good: Use async HTTP client
@mcp.tool()
async def good_async_tool(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Don't: Create Unlimited Concurrent Tasks

```python
# Bad: Unbounded concurrency
@mcp.tool()
async def bad_batch(urls: List[str]):
    # Could create thousands of simultaneous connections!
    results = await asyncio.gather(*[fetch(url) for url in urls])
    return results

# Good: Bounded concurrency
@mcp.tool()
async def good_batch(urls: List[str]):
    limiter = ConcurrencyLimiter(max_concurrent=10)
    results = await limiter.gather([fetch(url) for url in urls])
    return results
```

### Don't: Ignore Cancellation

```python
# Bad: Swallowing CancelledError
async def bad_task():
    try:
        await long_operation()
    except asyncio.CancelledError:
        pass  # Silently ignores cancellation!

# Good: Clean up and propagate
async def good_task():
    try:
        await long_operation()
    except asyncio.CancelledError:
        await cleanup()
        raise  # Propagate cancellation
```

## Related Documents

- [Timeout & Resilience](./12-timeout-resilience.md) - Timeout handling
- [Observability & Telemetry](./05-observability-telemetry.md) - Async logging
- [Tool Discovery](./13-tool-discovery.md) - Documenting async vs sync

---

**Navigation:** [← Feature Flags](./14-feature-flags.md) | [Index](./README.md)
