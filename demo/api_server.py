#!/usr/bin/env python3
"""
FastAPI server for the Bug Hunter Demo with streaming support.

Provides endpoints for running baseline and ACE modes with real-time progress.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ace import Generator, Reflector, Curator, OfflineAdapter, Playbook, Sample
from ace.llm_providers import LiteLLMClient

# Import demo modules
sys.path.insert(0, str(Path(__file__).parent))
from buggy_code_samples import BUGGY_SAMPLES, get_train_test_split
from bug_hunter_environment import BugHunterEnvironment

app = FastAPI(title="Bug Hunter Demo API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
demo_state = {
    "baseline_running": False,
    "ace_running": False,
    "ace_playbook": None,  # Pre-trained playbook
    "training_complete": False,
}

# Load pre-trained playbook if it exists
PLAYBOOK_PATH = Path(__file__).parent / "pretrained_playbook.json"


def load_pretrained_playbook():
    """Load pre-trained playbook from disk if it exists."""
    if PLAYBOOK_PATH.exists():
        try:
            playbook = Playbook.load_from_file(str(PLAYBOOK_PATH))
            demo_state["ace_playbook"] = playbook
            demo_state["training_complete"] = True
            print(f"✅ Loaded pre-trained playbook with {len(playbook.bullets())} strategies")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load playbook: {e}")
            return False
    else:
        print("ℹ️  No pre-trained playbook found. Run 'python demo/pretrain_playbook.py' first.")
        return False


# Try to load playbook on startup
load_pretrained_playbook()


async def pretrain_ace() -> AsyncGenerator[str, None]:
    """Pre-train ACE on training samples to build playbook."""
    try:
        # Get training samples only (first 6)
        train_samples_raw, _ = get_train_test_split(train_size=6)
        
        yield f"data: {json.dumps({'type': 'pretrain_start', 'total': len(train_samples_raw)})}\n\n"
        
        client = LiteLLMClient(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=1000
        )
        
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)
        environment = BugHunterEnvironment()
        
        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator,
            max_refinement_rounds=1,
            enable_observability=False
        )
        
        # Convert to Sample objects
        train_samples = [
            Sample(
                question=sample["code"],
                ground_truth=sample["ground_truth"],
                context=f"Language: {sample['language']}, Bug Type: {sample['bug_type']}",
                metadata={"id": sample["id"], "severity": sample["severity"]}
            )
            for sample in train_samples_raw
        ]
        
        # Train on samples
        for i, sample in enumerate(train_samples):
            yield f"data: {json.dumps({'type': 'pretrain_progress', 'sample_id': i + 1, 'total': len(train_samples)})}\n\n"
            
            # Run adapter on single sample
            adapter.run([sample], environment, epochs=1)
            
            # Send current playbook size
            bullet_count = len(adapter.playbook.bullets())
            yield f"data: {json.dumps({'type': 'pretrain_update', 'sample_id': i + 1, 'strategies': bullet_count})}\n\n"
        
        # Store the trained playbook globally
        demo_state["ace_playbook"] = adapter.playbook
        demo_state["training_complete"] = True
        
        # Send final strategies
        final_bullets = [
            {
                "content": bullet.content,
                "helpful": bullet.helpful_count,
                "harmful": bullet.harmful_count
            }
            for bullet in adapter.playbook.bullets()
        ]
        
        yield f"data: {json.dumps({'type': 'pretrain_complete', 'strategies': final_bullets, 'count': len(final_bullets)})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def stream_baseline_demo() -> AsyncGenerator[str, None]:
    """Stream baseline bug detection results on TEST samples."""
    try:
        client = LiteLLMClient(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=1000
        )
        
        generator = Generator(client)
        environment = BugHunterEnvironment()
        playbook = Playbook()  # Empty playbook for baseline
        
        # Get TEST samples only (last 4)
        _, test_samples_raw = get_train_test_split(train_size=6)
        
        # Convert samples
        samples = [
            Sample(
                question=sample["code"],
                ground_truth=sample["ground_truth"],
                context=f"Language: {sample['language']}, Bug Type: {sample['bug_type']}",
                metadata={"id": sample["id"], "severity": sample["severity"]}
            )
            for sample in test_samples_raw
        ]
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'start', 'mode': 'baseline', 'total': len(samples)})}\n\n"
        
        total_tokens = 0
        total_time = 0
        
        for i, sample in enumerate(samples):
            start_time = time.time()
            
            # Send progress
            yield f"data: {json.dumps({'type': 'progress', 'sample_id': i + 1, 'status': 'processing'})}\n\n"
            
            # Generate response
            output = generator.generate(
                question=f"Analyze this code and identify any bugs:\n\n{sample.question}",
                context="You are a code reviewer. Identify bugs, explain the issue, and suggest a fix.",
                playbook=playbook
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Extract token usage from raw output
            tokens_used = 0
            if hasattr(output, 'raw') and output.raw:
                usage = output.raw.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
            total_tokens += tokens_used
            
            # Evaluate
            env_result = environment.evaluate(sample, output)
            
            # Send result with full response text
            result = {
                'type': 'result',
                'sample_id': i + 1,
                'accuracy': env_result.metrics.get("accuracy", 0),
                'tokens': tokens_used,
                'time': elapsed,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'response': output.final_answer  # Full response
            }
            yield f"data: {json.dumps(result)}\n\n"
            
            await asyncio.sleep(0.1)  # Small delay for UI
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'total_tokens': total_tokens, 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def stream_ace_demo() -> AsyncGenerator[str, None]:
    """Stream ACE bug detection results on TEST samples using pre-trained playbook."""
    try:
        # Ensure pre-training is complete
        if not demo_state["training_complete"] or demo_state["ace_playbook"] is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'ACE not pre-trained. Run /api/pretrain first.'})}\n\n"
            return
        
        client = LiteLLMClient(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=1000
        )
        
        generator = Generator(client)
        environment = BugHunterEnvironment()
        
        # Use the pre-trained playbook!
        playbook = demo_state["ace_playbook"]
        
        # Log playbook info for debugging
        print(f"🧠 ACE using playbook with {len(playbook.bullets())} strategies")
        for i, bullet in enumerate(list(playbook.bullets())[:3], 1):
            print(f"   Strategy {i}: {bullet.content[:80]}...")
        
        # Get TEST samples only (last 4)
        _, test_samples_raw = get_train_test_split(train_size=6)
        
        # Convert samples
        samples = [
            Sample(
                question=sample["code"],
                ground_truth=sample["ground_truth"],
                context=f"Language: {sample['language']}, Bug Type: {sample['bug_type']}",
                metadata={"id": sample["id"], "severity": sample["severity"]}
            )
            for sample in test_samples_raw
        ]
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'start', 'mode': 'ace', 'total': len(samples)})}\n\n"
        
        total_tokens = 0
        total_time = 0
        strategies_count = len(playbook.bullets())
        
        # Process samples using pre-trained playbook (no further learning during race)
        for i, sample in enumerate(samples):
            # Start timer for THIS sample
            sample_start_time = time.time()
            
            # Send progress
            yield f"data: {json.dumps({'type': 'progress', 'sample_id': i + 1, 'status': 'processing'})}\n\n"
            
            # Generate using pre-trained playbook
            output = generator.generate(
                question=f"Analyze this code and identify any bugs:\n\n{sample.question}",
                context="You are a code reviewer. Identify bugs, explain the issue, and suggest a fix.",
                playbook=playbook
            )
            
            # Stop timer for THIS sample
            sample_elapsed = time.time() - sample_start_time
            total_time += sample_elapsed
            
            # Extract token usage from raw output
            tokens_used = 0
            if hasattr(output, 'raw') and output.raw:
                usage = output.raw.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
            total_tokens += tokens_used
            
            # Evaluate
            env_result = environment.evaluate(sample, output)
            
            # Send result with full response text
            result = {
                'type': 'result',
                'sample_id': i + 1,
                'accuracy': env_result.metrics.get("accuracy", 0),
                'tokens': tokens_used,
                'time': elapsed,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'response': output.final_answer,  # Full response
                'strategies_count': strategies_count  # Using pre-trained strategies
            }
            yield f"data: {json.dumps(result)}\n\n"
            
            await asyncio.sleep(0.1)  # Small delay for UI
        
        # Send learned strategies from pre-trained playbook
        final_bullets = [
            {
                "content": bullet.content,
                "helpful": bullet.helpful_count,
                "harmful": bullet.harmful_count
            }
            for bullet in playbook.bullets()
        ]
        yield f"data: {json.dumps({'type': 'strategies', 'strategies': final_bullets})}\n\n"
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'total_tokens': total_tokens, 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse("demo/frontend/index.html")


@app.get("/api/stream/baseline")
async def stream_baseline():
    """Stream baseline mode evaluation."""
    return StreamingResponse(
        stream_baseline_demo(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/stream/ace")
async def stream_ace():
    """Stream ACE mode evaluation."""
    return StreamingResponse(
        stream_ace_demo(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/pretrain")
async def pretrain():
    """Pre-train ACE on training samples to build playbook."""
    return StreamingResponse(
        pretrain_ace(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/reset")
async def reset_demo():
    """Reset demo state to allow re-running."""
    demo_state["ace_playbook"] = None
    demo_state["training_complete"] = False
    demo_state["baseline_running"] = False
    demo_state["ace_running"] = False
    return {"status": "reset"}


@app.get("/api/samples")
async def get_samples():
    """Get information about train/test split."""
    train, test = get_train_test_split(train_size=6)
    return {
        "train_count": len(train),
        "test_count": len(test),
        "total_count": len(BUGGY_SAMPLES)
    }


@app.get("/api/playbook/status")
async def playbook_status():
    """Check if pre-trained playbook is available."""
    if demo_state["training_complete"] and demo_state["ace_playbook"]:
        return {
            "available": True,
            "strategies_count": len(demo_state["ace_playbook"].bullets()),
            "source": "pretrained"
        }
    else:
        return {
            "available": False,
            "strategies_count": 0,
            "source": None,
            "message": "Run 'python demo/pretrain_playbook.py' to generate playbook"
        }


# Mount static files
app.mount("/static", StaticFiles(directory="demo/frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Bug Hunter Demo Server...")
    print("📊 Training: 6 samples | Testing: 4 samples")
    
    # Show playbook status
    if demo_state["training_complete"] and demo_state["ace_playbook"]:
        print(f"✅ Playbook loaded with {len(demo_state['ace_playbook'].bullets())} strategies")
        print(f"📚 Sample strategies:")
        for i, bullet in enumerate(list(demo_state["ace_playbook"].bullets())[:2], 1):
            print(f"   {i}. {bullet.content[:70]}...")
    else:
        print("⚠️  No playbook loaded! Run: python demo/pretrain_playbook.py")
    
    print("🌐 Open http://localhost:8000 in your browser")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
