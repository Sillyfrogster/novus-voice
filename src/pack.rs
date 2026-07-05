use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `config.json` at the root of every voice pack.
#[derive(Debug, Deserialize)]
pub struct PackConfig {
    pub id: String,
    pub engine: Engine,
    #[serde(rename = "sampleRate")]
    pub sample_rate: u32,
    pub model: String,
    pub tokenizer: Option<String>,
    pub voices: HashMap<String, String>,
    #[serde(rename = "wordTimings", default)]
    pub word_timings: bool,
    /// language tag, e.g. "en-US".
    pub language: String,
}

pub struct Pack {
    pub config: PackConfig,
    pub dir: PathBuf,
}

impl Pack {
    pub fn load(dir: &Path) -> Result<Pack> {
        let config_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: PackConfig = serde_json::from_str(&raw).context("parsing pack config.json")?;
        let pack = Pack { config, dir: dir.to_path_buf() };
        let mut required = vec![pack.model_path()];
        if let Some(t) = pack.tokenizer_path() {
            required.push(t);
        }
        required.extend(pack.config.voices.values().map(|v| pack.dir.join(v)));
        for f in required {
            if !f.is_file() {
                bail!("voice pack {} is missing {}", pack.config.id, f.display());
            }
        }
        Ok(pack)
    }

    pub fn model_path(&self) -> PathBuf {
        self.dir.join(&self.config.model)
    }

    pub fn tokenizer_path(&self) -> Option<PathBuf> {
        self.config.tokenizer.as_ref().map(|t| self.dir.join(t))
    }

    pub fn voice_path(&self, voice_id: &str) -> Result<PathBuf> {
        self.config
            .voices
            .get(voice_id)
            .map(|v| self.dir.join(v))
            .with_context(|| format!("unknown voice '{voice_id}' in pack {}", self.config.id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Kokoro,
    Piper,
}
