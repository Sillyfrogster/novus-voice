mod kokoro;
mod pack;
mod protocol;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use pack::{Engine, Pack};
use protocol::{Body, Cmd, Envelope, Reply};
use std::io::{BufRead, Write};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--say") {
        if let Err(e) = self_test(&args) {
            eprintln!("novus-voice: {e:#}");
            std::process::exit(1);
        }
        return;
    }
    stdio_loop();
}

fn stdio_loop() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let mut engine: Option<kokoro::KokoroEngine> = None;
    let mut pack_meta: Option<(String, Vec<String>, bool)> = None;

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let envelope: Envelope = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                emit(&mut stdout, &Reply::err(0, format!("bad request: {e}")));
                continue;
            }
        };
        let id = envelope.id;
        match envelope.cmd {
            Cmd::LoadVoice { pack_dir } => match load_pack(&pack_dir) {
                Ok((eng, meta)) => {
                    engine = Some(eng);
                    pack_meta = Some(meta.clone());
                    emit(
                        &mut stdout,
                        &Reply::ok(id, Body::Loaded {
                            pack_id: meta.0,
                            voices: meta.1,
                            word_timings: meta.2,
                        }),
                    );
                }
                Err(e) => emit(&mut stdout, &Reply::err(id, format!("{e:#}"))),
            },
            Cmd::Synthesize { text, voice_id, speed } => match engine.as_mut() {
                Some(eng) => match eng.synthesize(&text, &voice_id, speed.clamp(0.5, 3.0)) {
                    Ok(result) => emit(&mut stdout, &Reply::ok(id, audio_body(result))),
                    Err(e) => emit(&mut stdout, &Reply::err(id, format!("{e:#}"))),
                },
                None => emit(&mut stdout, &Reply::err(id, "no voice pack loaded")),
            },
            Cmd::Unload => {
                engine = None;
                pack_meta = None;
                emit(&mut stdout, &Reply::ack(id));
            }
            Cmd::Shutdown => {
                emit(&mut stdout, &Reply::ack(id));
                break;
            }
        }
        let _ = pack_meta.as_ref(); // TODO: status command?
    }
}

/// pack id, voice ids, word timings supported
type PackMeta = (String, Vec<String>, bool);

fn load_pack(dir: &Path) -> Result<(kokoro::KokoroEngine, PackMeta)> {
    let pack = Pack::load(dir)?;
    if pack.config.engine != Engine::Kokoro {
        bail!("engine {:?} not supported yet", pack.config.engine);
    }
    let mut voices: Vec<String> = pack.config.voices.keys().cloned().collect();
    voices.sort();
    let meta = (pack.config.id.clone(), voices, pack.config.word_timings);
    Ok((kokoro::KokoroEngine::load(&pack)?, meta))
}

fn audio_body(result: kokoro::Synthesis) -> Body {
    let duration_ms = (result.samples.len() as u64 * 1000 / result.sample_rate as u64) as u32;
    let mut pcm = Vec::with_capacity(result.samples.len() * 2);
    for s in &result.samples {
        pcm.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    Body::Audio {
        pcm_base64: base64::engine::general_purpose::STANDARD.encode(&pcm),
        sample_rate: result.sample_rate,
        duration_ms,
        words: result.words,
    }
}

fn emit(out: &mut impl Write, reply: &Reply) {.
    match serde_json::to_string(reply) {
        Ok(json) => {
            if writeln!(out, "{json}").and_then(|_| out.flush()).is_err() {
                eprintln!("novus-voice: stdout gone, exiting");
                std::process::exit(0);
            }
        }
        Err(e) => eprintln!("novus-voice: serialize failed: {e}"),
    }
}

fn self_test(args: &[String]) -> Result<()> {
    let get = |flag: &str| -> Option<&str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };
    let text = get("--say").context("--say TEXT required")?;
    let pack_dir = get("--pack").context("--pack DIR required")?;
    let voice = get("--voice").context("--voice ID required")?;
    let out_path = get("--out").unwrap_or("novus-voice-test.wav");
    let speed: f32 = get("--speed").map_or(Ok(1.0), str::parse)?;

    let started = std::time::Instant::now();
    let (mut engine, _) = load_pack(Path::new(pack_dir))?;
    eprintln!("loaded pack in {:?}", started.elapsed());

    let started = std::time::Instant::now();
    let result = engine.synthesize(text, voice, speed)?;
    let synth_time = started.elapsed();
    let audio_secs = result.samples.len() as f64 / result.sample_rate as f64;
    eprintln!(
        "synthesized {audio_secs:.2}s of audio in {synth_time:?} (RTF {:.3})",
        synth_time.as_secs_f64() / audio_secs
    );

    write_wav(Path::new(out_path), &result.samples, result.sample_rate)?;
    eprintln!("wrote {out_path}");
    println!("{}", serde_json::to_string_pretty(&result.words)?);
    Ok(())
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, wav)?;
    Ok(())
}
