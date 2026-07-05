use crate::pack::Pack;
use crate::protocol::WordTiming;
use anyhow::{bail, Context, Result};
use misaki_rs::{language::Language, G2P};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;

/// Kokoro-82M (timestamped ONNX export).
pub struct KokoroEngine {
    session: Session,
    /// Phoneme char -> input id, from the pack's tokenizer.json.
    vocab: HashMap<char, i64>,
    /// voiceId -> raw style table (rows of STYLE_DIM floats).
    styles: HashMap<String, Vec<f32>>,
    g2p: G2P,
    sample_rate: u32,
    audio_output: String,
    duration_output: Option<String>,
}

const STYLE_DIM: usize = 256;
const MAX_PHONEME_IDS: usize = 510;

pub struct Synthesis {
    /// Mono f32 samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub words: Vec<WordTiming>,
}

impl KokoroEngine {
    pub fn load(pack: &Pack) -> Result<Self> {
        // Level3 layout optimizations (NchwcTransformer) segfault onnxruntime
        // on this model's quantized exports; Level2 is stable and near-equal.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .map_err(|e| anyhow::anyhow!("session options: {e}"))?
            .commit_from_file(pack.model_path())
            .context("loading Kokoro ONNX model")?;

        let tokenizer_path = pack
            .tokenizer_path()
            .context("kokoro pack has no tokenizer file")?;
        let vocab = parse_vocab(&std::fs::read_to_string(tokenizer_path)?)?;

        let mut styles = HashMap::new();
        for voice_id in pack.config.voices.keys() {
            let bytes = std::fs::read(pack.voice_path(voice_id)?)?;
            if bytes.len() % (STYLE_DIM * 4) != 0 {
                bail!("style file for '{voice_id}' is not rows of {STYLE_DIM} f32s");
            }
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            styles.insert(voice_id.clone(), floats);
        }

        let lang = if pack.config.language.eq_ignore_ascii_case("en-gb") {
            Language::EnglishGB
        } else {
            Language::EnglishUS
        };

        // Output names differ across exports; detect rather than assume.
        let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        let audio_output = names
            .iter()
            .find(|n| {
                let n = n.to_lowercase();
                n.contains("audio") || n.contains("waveform") || n.contains("wav")
            })
            .cloned()
            .or_else(|| names.first().cloned())
            .context("model has no outputs")?;
        let duration_output = names
            .iter()
            .find(|n| n.to_lowercase().contains("dur"))
            .cloned();
        eprintln!("novus-voice: kokoro outputs {names:?} (audio={audio_output}, durations={duration_output:?})");

        Ok(KokoroEngine {
            session,
            vocab,
            styles,
            g2p: G2P::new(lang),
            sample_rate: pack.config.sample_rate,
            audio_output,
            duration_output,
        })
    }

    pub fn synthesize(&mut self, text: &str, voice_id: &str, speed: f32) -> Result<Synthesis> {
        let speak = speakable_text(text);
        let (phonemes, tokens) = self
            .g2p
            .g2p(&speak)
            .map_err(|e| anyhow::anyhow!("g2p failed: {e:?}"))?;

        // Map each phoneme char to an id
        let mut ids: Vec<i64> = Vec::with_capacity(phonemes.chars().count());
        let mut id_of_char: Vec<Option<usize>> = Vec::with_capacity(ids.capacity());
        for ch in phonemes.chars() {
            match self.vocab.get(&ch) {
                Some(&id) if ids.len() < MAX_PHONEME_IDS => {
                    id_of_char.push(Some(ids.len()));
                    ids.push(id);
                }
                _ => id_of_char.push(None),
            }
        }
        if ids.is_empty() {
            bail!("no speakable content in text");
        }

        let style_table = self
            .styles
            .get(voice_id)
            .with_context(|| format!("voice '{voice_id}' not loaded"))?;
        let rows = style_table.len() / STYLE_DIM;
        let row = ids.len().min(rows - 1);
        let style: Vec<f32> = style_table[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

        // Boundary pads (id 0) on both ends, as the reference pipeline does.
        let mut padded: Vec<i64> = Vec::with_capacity(ids.len() + 2);
        padded.push(0);
        padded.extend_from_slice(&ids);
        padded.push(0);
        let n_ids = padded.len();

        let input_ids = Tensor::from_array(([1, n_ids], padded))?;
        let style_t = Tensor::from_array(([1, STYLE_DIM], style))?;
        let speed_t = Tensor::from_array(([1], vec![speed]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "style" => style_t,
            "speed" => speed_t,
        ])?;

        let (_, audio) = outputs[self.audio_output.as_str()]
            .try_extract_tensor::<f32>()
            .context("extracting audio output")?;
        let samples: Vec<f32> = audio.to_vec();

        let words = match &self.duration_output {
            Some(name) => {
                let durations = extract_durations(&outputs[name.as_str()])?;
                word_timings(
                    &speak,
                    &tokens,
                    &id_of_char,
                    &durations,
                    samples.len(),
                    self.sample_rate,
                )
            }
            None => Vec::new(),
        };

        Ok(Synthesis { samples, sample_rate: self.sample_rate, words })
    }
}

fn speakable_text(text: &str) -> String {
    let mut chars: Vec<char> = text
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' => '\'',
            c => c,
        })
        .collect();

    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_uppercase() {
            i += 1;
            continue;
        }
        // Extend across capitals and in-word apostrophes ("DIDN'T").
        let mut j = i;
        let mut letters = 0;
        while j < chars.len() && (chars[j].is_ascii_uppercase() || chars[j] == '\'') {
            if chars[j] != '\'' {
                letters += 1;
            }
            j += 1;
        }
        let boundary_before = i == 0 || !chars[i - 1].is_alphabetic();
        let boundary_after = j >= chars.len() || !chars[j].is_alphabetic();
        if letters >= 2 && boundary_before && boundary_after {
            for c in &mut chars[i..j] {
                *c = c.to_ascii_lowercase();
            }
        }
        i = j.max(i + 1);
    }
    chars.into_iter().collect()
}

/// tokenizer.json 
fn parse_vocab(raw: &str) -> Result<HashMap<char, i64>> {
    let json: serde_json::Value = serde_json::from_str(raw)?;
    let vocab = json["model"]["vocab"]
        .as_object()
        .context("tokenizer.json has no model.vocab")?;
    let mut map = HashMap::with_capacity(vocab.len());
    for (key, val) in vocab {
        let mut chars = key.chars();
        if let (Some(ch), None, Some(id)) = (chars.next(), chars.next(), val.as_i64()) {
            map.insert(ch, id);
        }
    }
    Ok(map)
}

/// Duration outputs vary in dtype across exports; normalize to f32 per input id.
fn extract_durations(value: &ort::value::DynValue) -> Result<Vec<f32>> {
    if let Ok((_, d)) = value.try_extract_tensor::<i64>() {
        return Ok(d.iter().map(|&v| v as f32).collect());
    }
    if let Ok((_, d)) = value.try_extract_tensor::<f32>() {
        return Ok(d.to_vec());
    }
    bail!("unsupported duration tensor dtype")
}

#[cfg(test)]
mod tests {
    use super::speakable_text;

    #[test]
    fn straightens_curly_apostrophes() {
        assert_eq!(speakable_text("He said don’t worry."), "He said don't worry.");
    }

    #[test]
    fn lowercases_caps_styled_prose() {
        assert_eq!(
            speakable_text("IT WAS A BRIGHT COLD DAY."),
            "it was A bright cold day."
        );
        assert_eq!(speakable_text("“DIDN’T YOU?”"), "“didn't you?”");
    }

    #[test]
    fn leaves_ordinary_case_alone() {
        assert_eq!(speakable_text("I saw McDONALD yesterday."), "I saw McDONALD yesterday.");
        assert_eq!(speakable_text("The iPhone I bought."), "The iPhone I bought.");
    }

    #[test]
    fn preserves_char_count() {
        let input = "“DON’T,” he said — com\u{00AD}pletely.";
        assert_eq!(speakable_text(input).chars().count(), input.chars().count());
    }
}

/// Convert per-id durations into word timings with char offsets into `text`.
fn word_timings(
    text: &str,
    tokens: &[misaki_rs::MToken],
    id_of_char: &[Option<usize>],
    durations: &[f32],
    n_samples: usize,
    sample_rate: u32,
) -> Vec<WordTiming> {
    let total: f32 = durations.iter().sum();
    if total <= 0.0 || n_samples == 0 {
        return Vec::new();
    }
    let ms_per_unit = (n_samples as f64 * 1000.0 / sample_rate as f64) / total as f64;
    let mut starts_ms = Vec::with_capacity(durations.len() + 1);
    let mut acc = 0.0f64;
    for d in durations {
        starts_ms.push(acc * ms_per_unit);
        acc += *d as f64;
    }
    starts_ms.push(acc * ms_per_unit);

    let id_time = |char_idx: usize| -> Option<(f64, f64)> {
        let id = id_of_char.get(char_idx).copied().flatten()?;
        // +1 skips the leading boundary pad.
        let s = *starts_ms.get(id + 1)?;
        let e = *starts_ms.get(id + 2)?;
        Some((s, e))
    };

    // Walk tokens through the phoneme string and the original text in parallel.
    let mut words = Vec::new();
    let mut ph_cursor = 0usize; // char index into the phoneme string
    let mut txt_cursor = 0usize; // byte index into `text`
    for tk in tokens {
        let ph_len = tk.phonemes.as_deref().map_or(0, |p| p.chars().count());
        let span = ph_cursor..ph_cursor + ph_len;
        ph_cursor += ph_len + tk.whitespace.chars().count();

        let found = text[txt_cursor..].find(&tk.text).map(|off| {
            let start = txt_cursor + off;
            (start, start + tk.text.len())
        });
        if let Some((start_char, end_char)) = found {
            txt_cursor = end_char;
            if !tk.text.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            let times: Vec<(f64, f64)> = span.clone().filter_map(id_time).collect();
            if let (Some(&(first, _)), Some(&(_, last))) = (times.first(), times.last()) {
                words.push(WordTiming {
                    start_ms: first as u32,
                    end_ms: last as u32,
                    start_char: text[..start_char].chars().count() as u32,
                    end_char: text[..end_char].chars().count() as u32,
                });
            }
        }
    }
    words
}
