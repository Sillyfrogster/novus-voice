use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One request per line on stdin. 
#[derive(Debug, Deserialize)]
pub struct Envelope {
    pub id: u64,
    #[serde(flatten)]
    pub cmd: Cmd,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Cmd {
    /// Load a voice pack from disk and keep its session resident.
    LoadVoice {
        #[serde(rename = "packDir")]
        pack_dir: PathBuf,
    },
    /// Synthesize one sentence. `voiceId` selects a voice within the loaded pack.
    Synthesize {
        text: String,
        #[serde(rename = "voiceId")]
        voice_id: String,
        speed: f32,
    },
    /// Drop the loaded pack (frees the ONNX session).
    Unload,
    /// Exit cleanly.
    Shutdown,
}

#[derive(Debug, Serialize)]
pub struct Reply {
    pub id: u64,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub body: Option<Body>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Body {
    Loaded {
        #[serde(rename = "packId")]
        pack_id: String,
        voices: Vec<String>,
        #[serde(rename = "wordTimings")]
        word_timings: bool,
    },
    Audio {
        #[serde(rename = "pcmBase64")]
        pcm_base64: String,
        #[serde(rename = "sampleRate")]
        sample_rate: u32,
        #[serde(rename = "durationMs")]
        duration_ms: u32,
        words: Vec<WordTiming>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct WordTiming {
    #[serde(rename = "startMs")]
    pub start_ms: u32,
    #[serde(rename = "endMs")]
    pub end_ms: u32,
    #[serde(rename = "startChar")]
    pub start_char: u32,
    #[serde(rename = "endChar")]
    pub end_char: u32,
}

impl Reply {
    pub fn ok(id: u64, body: Body) -> Self {
        Reply { id, ok: true, error: None, body: Some(body) }
    }
    pub fn ack(id: u64) -> Self {
        Reply { id, ok: true, error: None, body: None }
    }
    pub fn err(id: u64, error: impl ToString) -> Self {
        Reply { id, ok: false, error: Some(error.to_string()), body: None }
    }
}
