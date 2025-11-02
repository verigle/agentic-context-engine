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

/// Extract session ID from a log entry
fn get_session_id(entry: &Value) -> Option<String> {
    entry
        .get("request_properties")?
        .get("Helicone-Session-Id")?
        .as_str()
        .map(String::from)
}

/// Extract user ID from a log entry
fn get_user_id(entry: &Value) -> Option<String> {
    entry
        .get("request_user_id")?
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

/// Extract all user text messages from a messages array
fn extract_user_text_messages(messages: &[Value]) -> Vec<String> {
    let mut user_messages = Vec::new();

    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            if let Some(content_array) = msg.get("content").and_then(|c| c.as_array()) {
                for item in content_array {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            user_messages.push(text.to_string());
                        }
                    }
                }
            }
        }
    }

    user_messages
}

/// Extract NEW user messages by comparing current with previous messages
fn extract_new_user_messages(entry: &Value, prev_user_messages: &[String]) -> Vec<String> {
    let messages = match entry
        .get("request_body")
        .and_then(|rb| rb.get("messages"))
        .and_then(|m| m.as_array())
    {
        Some(m) => m,
        None => return Vec::new(),
    };

    let current_user_messages = extract_user_text_messages(messages);

    // Find messages that weren't in the previous set
    let prev_set: HashSet<_> = prev_user_messages.iter().collect();
    current_user_messages
        .into_iter()
        .filter(|msg| !prev_set.contains(&msg))
        .collect()
}

/// Get the first N user messages from a conversation for overlap detection
fn get_first_user_messages(messages: &[Value], n: usize) -> Vec<String> {
    if messages.is_empty() {
        return Vec::new();
    }

    let first_entry = &messages[0];
    if let Some(msg_array) = first_entry
        .get("request_body")
        .and_then(|rb| rb.get("messages"))
        .and_then(|m| m.as_array())
    {
        let user_msgs = extract_user_text_messages(msg_array);
        user_msgs.into_iter().take(n).collect()
    } else {
        Vec::new()
    }
}

/// Get the last N user messages from a conversation for overlap detection
fn get_last_user_messages(messages: &[Value], n: usize) -> Vec<String> {
    if messages.is_empty() {
        return Vec::new();
    }

    let last_entry = messages.last().unwrap();
    if let Some(msg_array) = last_entry
        .get("request_body")
        .and_then(|rb| rb.get("messages"))
        .and_then(|m| m.as_array())
    {
        let user_msgs = extract_user_text_messages(msg_array);
        let start = user_msgs.len().saturating_sub(n);
        user_msgs.into_iter().skip(start).collect()
    } else {
        Vec::new()
    }
}

/// Check if conversation B is a continuation of conversation A
/// by checking if B's first messages overlap with A's last messages
fn has_message_overlap(
    conv_a: &[Value],
    conv_b: &[Value],
    user_a: &Option<String>,
    user_b: &Option<String>,
) -> bool {
    // Must have same user
    if user_a != user_b || user_a.is_none() {
        return false;
    }

    // Check for message overlap (compare last 5 messages of A with first 5 of B)
    let last_msgs_a = get_last_user_messages(conv_a, 5);
    let first_msgs_b = get_first_user_messages(conv_b, 5);

    if last_msgs_a.is_empty() || first_msgs_b.is_empty() {
        return false;
    }

    // If at least 2 of B's first messages match any of A's last messages, it's likely a continuation
    let mut match_count = 0;
    for msg_b in &first_msgs_b {
        for msg_a in &last_msgs_a {
            if msg_a == msg_b {
                match_count += 1;
                break;
            }
        }
    }

    match_count >= 2
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
                                step: 0,
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
                break;
            }
        }
    }

    turns
}

/// Convert a full conversation to incremental turns
fn conversation_to_turns(messages: &[Value]) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let mut step = 0;
    let mut seen_user_messages: Vec<String> = Vec::new();

    for entry in messages {
        // Extract NEW user messages by comparing with previous
        let new_user_msgs = extract_new_user_messages(entry, &seen_user_messages);

        for user_msg in &new_user_msgs {
            step += 1;
            let timestamp = get_timestamp(entry, "request_created_at");
            let request_id = get_request_id(entry);

            turns.push(ConversationTurn {
                turn_type: "user".to_string(),
                step,
                timestamp,
                request_id,
                content: Some(user_msg.clone()),
                tool: None,
                tool_use_id: None,
                input: None,
                status: None,
            });
        }

        // Update seen messages
        seen_user_messages.extend(new_user_msgs);

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

        // Add to conversation (create if doesn't exist)
        active_conversations
            .entry(session_id)
            .or_insert_with(Vec::new)
            .push(entry);

        if stats.total_lines % 10000 == 0 {
            println!("  ✓ Processed {} lines, {} sessions so far", stats.total_lines, active_conversations.len());
        }
    }

    Ok(())
}

#[derive(Default)]
struct Stats {
    total_lines: usize,
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
    let mut stats = Stats::default();

    for chunk_file in &chunk_files {
        let chunk_path = chunks_dir.join(chunk_file);

        if !chunk_path.exists() {
            println!("⚠️  Warning: {} not found, skipping...", chunk_file);
            continue;
        }

        println!("\n📄 Processing {}...", chunk_file);

        process_chunk_file(&chunk_path, &mut active_conversations, &mut stats)?;
    }

    println!("\n📊 Processing complete!");
    println!("  Total lines processed: {}", stats.total_lines);
    println!("  Unique sessions found: {}", active_conversations.len());

    Ok(active_conversations)
}

/// Merge conversations that are continuations (same user, overlapping messages)
fn merge_continuations(
    conversations: HashMap<String, Vec<Value>>,
) -> HashMap<String, Vec<Value>> {
    println!("\n🔗 Detecting conversation continuations...");

    // Extract user IDs for each conversation
    let mut conv_users: HashMap<String, Option<String>> = HashMap::new();
    let mut conv_timestamps: HashMap<String, String> = HashMap::new();

    for (session_id, messages) in &conversations {
        let user_id = messages.first().and_then(|entry| get_user_id(entry));
        let timestamp = messages.first().map(|entry| get_timestamp(entry, "request_created_at")).unwrap_or_default();
        conv_users.insert(session_id.clone(), user_id);
        conv_timestamps.insert(session_id.clone(), timestamp);
    }

    // Sort conversations by first timestamp
    let mut sorted_convs: Vec<(String, Vec<Value>)> = conversations.into_iter().collect();
    sorted_convs.sort_by_key(|(session_id, _)| {
        conv_timestamps.get(session_id).cloned().unwrap_or_default()
    });

    let mut merged: HashMap<String, Vec<Value>> = HashMap::new();
    let mut merged_into: HashMap<String, String> = HashMap::new(); // Maps session_id -> merged_into_session_id

    for (session_id, messages) in sorted_convs {
        // Check if this was already merged
        if merged_into.contains_key(&session_id) {
            continue;
        }

        // Check if this is a continuation of any existing conversation
        let user_id = conv_users.get(&session_id).unwrap();
        let mut found_parent = false;

        for (parent_id, parent_messages) in merged.iter_mut() {
            let parent_user = conv_users.get(parent_id).unwrap();

            if has_message_overlap(parent_messages, &messages, parent_user, user_id) {
                println!(
                    "  ✓ Merging {} into {} (same user, overlapping messages)",
                    &session_id[..20.min(session_id.len())],
                    &parent_id[..20.min(parent_id.len())]
                );
                parent_messages.extend(messages.clone());
                merged_into.insert(session_id.clone(), parent_id.clone());
                found_parent = true;
                break;
            }
        }

        if !found_parent {
            merged.insert(session_id, messages);
        }
    }

    let original_count = conv_users.len();
    let merged_count = merged.len();
    let continuation_count = original_count - merged_count;

    println!(
        "  Merged {} continuations into existing conversations",
        continuation_count
    );
    println!("  Final conversation count: {}", merged_count);

    merged
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

    // Process all chunks - collect ALL sessions
    let conversations = process_all_chunks(&chunks_dir)?;

    // Merge conversation continuations based on user_id + message overlap
    let merged_conversations = merge_continuations(conversations);

    // Save results
    let count = merged_conversations.len();
    if count > 0 {
        save_conversations(merged_conversations, &output_dir)?;
        println!("\n✅ Done! {} complete conversations constructed.", count);
    } else {
        println!("\n⚠️  No complete conversations found!");
    }

    Ok(())
}
