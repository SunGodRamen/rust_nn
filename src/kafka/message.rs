// src/kafka/message.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct KafkaMessage {
    num_windows: u32,
    cursor_position: CursorPosition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CursorPosition {
    x: u32,
    y: u32,
}

// Ensure to add `mod message;` in your `src/kafka/mod.rs`
