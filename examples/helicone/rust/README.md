# Conversation Constructor

Fast and efficient Rust application to construct complete conversations from Helicone JSONL log chunks.

## Overview

This tool processes JSONL log files in chronological order and constructs complete conversations by tracking session IDs. Only conversations that have their **FIRST message** present in the logs are collected.

### Processing Order
- **Files**: af (oldest) → ae → ad → ac → ab → aa (newest)
- **Within each file**: Last line (oldest) → First line (newest)

## Build & Run

```bash
# Build in release mode (optimized)
cargo build --release

# Run
cargo run --release

# Or run directly
./target/release/conversation-constructor
```

## Output

Creates conversations in **TWO formats**:

### 1. Full Format (`../complete_conversations/full/`)
Complete JSON files with all raw data:
- `conversation_<session_id>.json` - One file per conversation

```json
{
  "session_id": "...",
  "message_count": 10,
  "first_message_time": "2025-10-22T00:09:31.085Z",
  "last_message_time": "2025-10-22T00:15:42.123Z",
  "messages": [...]
}
```

### 2. Delta Format (`../complete_conversations/delta/`)
Incremental turns in JSONL (one turn per line, **no repeated context**):
- `conversation_<session_id>.jsonl` - One file per conversation

Each line represents a single conversation turn:
```jsonl
{"type":"user","step":1,"timestamp":"...","request_id":"...","content":"..."}
{"type":"assistant","step":2,"timestamp":"...","request_id":"...","content":"..."}
{"type":"tool_call","step":3,"timestamp":"...","request_id":"...","tool":"...","tool_use_id":"...","input":{...}}
{"type":"tool_result","step":4,"timestamp":"...","request_id":"...","tool_use_id":"...","content":"{...}","status":"success"}
```

**Delta format benefits:**
- No repeated context between turns
- Easy sequential processing
- Much smaller file sizes
- Perfect for conversation flow analysis

## Performance

Optimizations:
- Compiled with `opt-level = 3` and LTO
- Efficient JSON parsing with `serde_json`
- Streaming line processing
- Minimal allocations

Expected performance: Processing ~1M log lines in <10 seconds on modern hardware.

## Example Results

