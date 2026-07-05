# novus-voice

Offline neural text-to-speech engine for [Novus](https://github.com/Sillyfrogster/novus), the desktop ebook reader. It turns sentences into natural-sounding speech on-device and reports **word-level timing** for every word it speaks.

- **Fully offline.** Synthesis runs on the local CPU; nothing leaves the machine.
- **Word timings built in.** Each reply maps every spoken word back to exact character offsets in the input text.
- **Faster than real time.** Sentence-level synthesis outpaces playback on a modest modern CPU, so audio streams without gaps.
- **Small.** A single self-contained binary; voices are downloaded separately as [voice packs](https://github.com/Sillyfrogster/novus-voices).

## How it fits into Novus

novus-voice is a *sidecar*: a helper process that the app launches and supervises, communicating over stdin/stdout. It is deliberately not compiled into the app:

- **License isolation.** Novus is MIT; this engine links GPL components. A process boundary keeps the licenses separate.
- **Fault isolation.** A crash in the engine never takes the app down.
- **Memory.** Loaded voice models use RAM; ending the process returns all of it the moment it's stopped.

The current engine is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), an 82-million-parameter TTS model, run via [onnxruntime](https://onnxruntime.ai/) with [misaki-rs](https://crates.io/crates/misaki-rs) for grapheme-to-phoneme conversion and [espeak-ng](https://github.com/espeak-ng/espeak-ng) as the fallback for out-of-vocabulary words.

## Protocol

One JSON request per stdin line; one JSON reply per stdout line, matched by `id`. Logs go to stderr. Closing stdin shuts the process down.

```jsonc
// requests
{"id":1,"cmd":"load_voice","packDir":"/path/to/voice-pack"}
{"id":2,"cmd":"synthesize","text":"One sentence.","voiceId":"af_heart","speed":1.0}
{"id":3,"cmd":"unload"}
{"id":4,"cmd":"shutdown"}

// synthesize reply
{
  "id":2, "ok":true,
  "pcmBase64":"…",
  "sampleRate":24000,
  "durationMs":1234,
  "words":[{"startMs":327,"endMs":389,"startChar":0,"endChar":3}, …]
}
```

`startChar`/`endChar` index into the exact `text` string of the request, so the caller can map timings onto its own document without string matching. `words` is empty for voices whose model has no duration outputs; callers should degrade to sentence-level highlighting.

## Voice packs

Voices ship separately as packs: a zip containing the model, voice style files, and a `config.json`:

```jsonc
{
  "id": "kokoro-en-v1",
  "engine": "kokoro",
  "sampleRate": 24000,
  "model": "model_q8.onnx",
  "tokenizer": "tokenizer.json", // phoneme→id vocab (HF tokenizers format)
  "voices": { "af_heart": "voices/af_heart.bin" },
  "voiceNames": { "af_heart": "Heart — American, female" },
  "wordTimings": true,
  "language": "en-US"
}
```

Packs for Novus are published at [novus-voices](https://github.com/Sillyfrogster/novus-voices).

## Try it

The binary doubles as a command-line tool:

```sh
novus-voice --say "The captain studied the horizon." \
  --pack /path/to/kokoro-en-v1 --voice af_heart --out test.wav
```

Prints load time and synthesis speed, writes a WAV, and dumps the word timings as JSON.

## Building from source

Requires a Rust toolchain, plus `cmake` and `libclang` (used to build the espeak-ng fallback):

```sh
cargo build --release
```

Build without the GPL fallback (unknown words are spelled letter-by-letter):

```sh
cargo build --release --no-default-features
```

## License

[GPL-3.0-or-later](LICENSE). The default build links espeak-ng (GPL-3.0). Individual components carry their own licenses: Kokoro-82M and its ONNX export are Apache-2.0, misaki-rs is MIT, ort is MIT/Apache-2.0.

## Acknowledgments

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, and the timestamped ONNX export by [onnx-community](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped)
- [misaki-rs](https://crates.io/crates/misaki-rs), [ort](https://ort.pyke.io/), and [espeak-ng](https://github.com/espeak-ng/espeak-ng)
