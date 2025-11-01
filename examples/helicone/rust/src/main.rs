use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
struct LogEntry {
    #[serde(flatten)]
    data: Value,
}

#[derive(Debug, Serialize)]
struct Conversation {
    session_id: String,
    message_count: usize,
    first_message_time: String,
    last_message_time: String,
    messages: Vec<Value>,
}

#[derive(Debug, Serialize)]
struct ConversationTurn {
    #[serde(rename = "type")]
    turn_type: String,
    step: usize,
    timestamp: String,
    request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
}

/// Check if a log entry represents the first message in a conversation
fn is_first_message(entry: &Value) -> bool {
    if let Some(request_body) = entry.get("request_body") {
        if let Some(messages) = request_body.get("messages") {
            if let Some(messages_array) = messages.as_array() {
                // First message: exactly 1 message with role "user"
                if messages_array.len() == 1 {
                    if let Some(first_msg) = messages_array.first() {
                        if let Some(role) = first_msg.get("role") {
                            return role.as_str() == Some("user");
                        }
                    }
                }
            }
        }
    }
    false
}

/// Extract session ID from a log entry
fn get_session_id(entry: &Value) -> Option<String> {
    entry
        .get("request_properties")?
        .get("Helicone-Session-Id")?
        .as_str()
        .map(String::from)
}

/// Get timestamp from a log entry
fn get_timestamp(entry: &Value, field: &str) -> String {
    entry
        .get(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Get request ID from a log entry
fn get_request_id(entry: &Value) -> String {
    entry
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Extract the NEW user message from a request (last message in the array)
fn extract_user_message(entry: &Value) -> Option<String> {
    let messages = entry.get("request_body")?.get("messages")?.as_array()?;
    let last_msg = messages.last()?;

    // Check if it's a user message
    if last_msg.get("role")?.as_str()? == "user" {
        // Extract text content from content array
        if let Some(content_array) = last_msg.get("content").and_then(|c| c.as_array()) {
            for item in content_array {
                if item.get("type")?.as_str()? == "text" {
                    return item.get("text").and_then(|t| t.as_str()).map(String::from);
                }
            }
        }
    }
    None
}

/// Extract assistant response content
fn extract_assistant_content(entry: &Value) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let timestamp = get_timestamp(entry, "response_created_at");
    let request_id = get_request_id(entry);

    if let Some(content_array) = entry
        .get("response_body")
        .and_then(|rb| rb.get("content"))
        .and_then(|c| c.as_array())
    {
        for item in content_array {
            if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                match item_type {
                    "text" => {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            turns.push(ConversationTurn {
                                turn_type: "assistant".to_string(),
                                step: 0, // Will be set later
                                timestamp: timestamp.clone(),
                                request_id: request_id.clone(),
                                content: Some(text.to_string()),
                                tool: None,
                                tool_use_id: None,
                                input: None,
                                status: None,
                            });
                        }
                    }
                    "tool_use" => {
                        if let (Some(tool_name), Some(tool_id), Some(input)) = (
                            item.get("name").and_then(|n| n.as_str()),
                            item.get("id").and_then(|i| i.as_str()),
                            item.get("input"),
                        ) {
                            turns.push(ConversationTurn {
                                turn_type: "tool_call".to_string(),
                                step: 0,
                                timestamp: timestamp.clone(),
                                request_id: request_id.clone(),
                                content: None,
                                tool: Some(tool_name.to_string()),
                                tool_use_id: Some(tool_id.to_string()),
                                input: Some(input.clone()),
                                status: None,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    turns
}

/// Extract tool results from request messages
fn extract_tool_results(entry: &Value) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let timestamp = get_timestamp(entry, "request_created_at");
    let request_id = get_request_id(entry);

    if let Some(messages) = entry
        .get("request_body")
        .and_then(|rb| rb.get("messages"))
        .and_then(|m| m.as_array())
    {
        // Look for the last tool_result in messages (the new one)
        for msg in messages.iter().rev() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(content_array) = msg.get("content").and_then(|c| c.as_array()) {
                    for item in content_array {
                        if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                            if let (Some(tool_use_id), Some(content)) = (
                                item.get("tool_use_id").and_then(|i| i.as_str()),
                                item.get("content").and_then(|c| c.as_str()),
                            ) {
                                turns.push(ConversationTurn {
                                    turn_type: "tool_result".to_string(),
                                    step: 0,
                                    timestamp: timestamp.clone(),
                                    request_id: request_id.clone(),
                                    content: Some(content.to_string()),
                                    tool: None,
                                    tool_use_id: Some(tool_use_id.to_string()),
                                    input: None,
                                    status: Some("success".to_string()),
                                });
                                // Only take the first tool_result we find (most recent)
                                return turns;
                            }
                        }
                    }
                }
                break; // Stop after checking the last user message
            }
        }
    }

    turns
}

/// Convert a full conversation to incremental turns
fn conversation_to_turns(messages: &[Value]) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let mut step = 0;

    for entry in messages {
        // Check for new user message
        if let Some(user_msg) = extract_user_message(entry) {
            step += 1;
            let timestamp = get_timestamp(entry, "request_created_at");
            let request_id = get_request_id(entry);

            turns.push(ConversationTurn {
                turn_type: "user".to_string(),
                step,
                timestamp,
                request_id,
                content: Some(user_msg),
                tool: None,
                tool_use_id: None,
                input: None,
                status: None,
            });
        }

        // Check for tool results (comes before assistant response)
        let tool_results = extract_tool_results(entry);
        for mut turn in tool_results {
            step += 1;
            turn.step = step;
            turns.push(turn);
        }

        // Extract assistant response (text and/or tool calls)
        let assistant_turns = extract_assistant_content(entry);
        for mut turn in assistant_turns {
            step += 1;
            turn.step = step;
            turns.push(turn);
        }
    }

    turns
}

/// Process a single chunk file in reverse order (last line first)
fn process_chunk_file(
    path: &Path,
    active_conversations: &mut HashMap<String, Vec<Value>>,
    skipped_sessions: &mut HashSet<String>,
    stats: &mut Stats,
) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let reader = BufReader::new(file);

    // Read all lines into memory (needed for reverse processing)
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    println!("  📄 Processing {} lines...", lines.len());

    // Process in reverse order (last line is oldest)
    for line in lines.iter().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        stats.total_lines += 1;

        // Parse JSON
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                println!("  ⚠️  JSON parse error: {}", e);
                continue;
            }
        };

        // Get session ID
        let session_id = match get_session_id(&entry) {
            Some(id) => id,
            None => continue,
        };

        // Check if this is a first message
        if is_first_message(&entry) {
            // Only start tracking if we haven't seen this session yet
            if !active_conversations.contains_key(&session_id) {
                active_conversations.insert(session_id.clone(), vec![entry]);
                stats.first_messages_found += 1;

                if stats.first_messages_found % 100 == 0 {
                    println!("  ✓ Found {} conversation starts", stats.first_messages_found);
                }
            }
        } else {
            // Add to conversation if we're tracking this session
            if let Some(messages) = active_conversations.get_mut(&session_id) {
                messages.push(entry);
            } else {
                // Track incomplete sessions (started before our logs)
                skipped_sessions.insert(session_id);
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct Stats {
    total_lines: usize,
    first_messages_found: usize,
}

/// Process all chunk files in chronological order
fn process_all_chunks(chunks_dir: &Path) -> Result<HashMap<String, Vec<Value>>> {
    // Files in order from oldest to newest
    let chunk_files = [
        "helicone-chunk-af.jsonl",
        "helicone-chunk-ae.jsonl",
        "helicone-chunk-ad.jsonl",
        "helicone-chunk-ac.jsonl",
        "helicone-chunk-ab.jsonl",
        "helicone-chunk-aa.jsonl",
    ];

    let mut active_conversations: HashMap<String, Vec<Value>> = HashMap::new();
    let mut skipped_sessions: HashSet<String> = HashSet::new();
    let mut stats = Stats::default();

    for chunk_file in &chunk_files {
        let chunk_path = chunks_dir.join(chunk_file);

        if !chunk_path.exists() {
            println!("⚠️  Warning: {} not found, skipping...", chunk_file);
            continue;
        }

        println!("\n📄 Processing {}...", chunk_file);

        process_chunk_file(
            &chunk_path,
            &mut active_conversations,
            &mut skipped_sessions,
            &mut stats,
        )?;
    }

    println!("\n📊 Processing complete!");
    println!("  Total lines processed: {}", stats.total_lines);
    println!("  Complete conversations found: {}", active_conversations.len());
    println!("  Incomplete sessions skipped: {}", skipped_sessions.len());

    Ok(active_conversations)
}

/// Save conversations in both formats
fn save_conversations(
    conversations: HashMap<String, Vec<Value>>,
    output_dir: &Path,
) -> Result<()> {
    // Create output directories
    let full_dir = output_dir.join("full");
    let delta_dir = output_dir.join("delta");

    fs::create_dir_all(&full_dir)
        .with_context(|| format!("Failed to create directory: {}", full_dir.display()))?;
    fs::create_dir_all(&delta_dir)
        .with_context(|| format!("Failed to create directory: {}", delta_dir.display()))?;

    let total = conversations.len();
    let mut saved = 0;

    for (session_id, messages) in conversations {
        // Save full conversation as JSON
        let full_file = full_dir.join(format!("conversation_{}.json", session_id));
        let first_time = get_timestamp(messages.first().unwrap(), "request_created_at");
        let last_time = get_timestamp(messages.last().unwrap(), "request_created_at");

        let conversation = Conversation {
            session_id: session_id.clone(),
            message_count: messages.len(),
            first_message_time: first_time,
            last_message_time: last_time,
            messages: messages.clone(),
        };

        let file = File::create(&full_file)
            .with_context(|| format!("Failed to create file: {}", full_file.display()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &conversation)?;

        // Save incremental turns as JSONL
        let delta_file = delta_dir.join(format!("conversation_{}.jsonl", session_id));
        let turns = conversation_to_turns(&messages);

        let file = File::create(&delta_file)
            .with_context(|| format!("Failed to create file: {}", delta_file.display()))?;
        let mut writer = BufWriter::new(file);

        for turn in turns {
            serde_json::to_writer(&mut writer, &turn)?;
            writeln!(&mut writer)?;
        }

        saved += 1;
        if saved % 100 == 0 {
            println!("  💾 Saved {}/{} conversations...", saved, total);
        }
    }

    println!("\n💾 Saved {} conversations:", total);
    println!("  - Full format: {}", full_dir.display());
    println!("  - Delta format: {}", delta_dir.display());

    Ok(())
}

fn main() -> Result<()> {
    println!("🚀 Starting conversation construction...\n");

    // Setup paths - chunks are in .private/Test Date/chunks
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join(".private")
        .join("Test Date");
    let chunks_dir = project_root.join("chunks");
    let output_dir = project_root.join("complete_conversations");

    println!("📁 Chunks directory: {}", chunks_dir.display());
    println!("📁 Output directory: {}", output_dir.display());

    // Process all chunks
    let conversations = process_all_chunks(&chunks_dir)?;

    // Save results
    let count = conversations.len();
    if count > 0 {
        save_conversations(conversations, &output_dir)?;
        println!("\n✅ Done! {} complete conversations constructed.", count);
    } else {
        println!("\n⚠️  No complete conversations found!");
    }

    Ok(())
}
