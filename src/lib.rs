//! # kokoro-mlx-rs
//!
//! Fast Kokoro-82M TTS on Apple Silicon via MLX with `mx.compile` kernel fusion.
//!
//! Wraps [`voice-tts`](https://crates.io/crates/voice-tts) 0.2 (mlx-rs backend)
//! and enables MLX compilation for the decoder pass, achieving ~25-30% speedup
//! over the base implementation.
//!
//! ## Performance (Apple Silicon M4)
//!
//! | Backend                    | Medium (164ch) | RTFx  |
//! |----------------------------|----------------|-------|
//! | Python MLX (reference)     | ~1100ms        | 11.1x |
//! | **kokoro-mlx-rs (this)**   | ~1480ms        | 8.5x  |
//! | voice-tts 0.2 (no compile) | ~1990ms        | 6.4x  |
//! | Candle + Metal             | ~5900ms        | 2.1x  |
//! | ONNX Runtime CPU           | ~5700ms        | 2.3x  |

use mlx_rs::Array;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub use voice_tts::{self, KokoroModel};

/// Load Kokoro model from HuggingFace repo or local path.
pub fn load_model(path_or_repo: &str) -> voice_tts::Result<KokoroModel> {
    voice_tts::load_model(path_or_repo)
}

/// Load a voice embedding by name.
pub fn load_voice(voice_name: &str, repo_id: Option<&str>) -> voice_tts::Result<Array> {
    voice_tts::load_voice(voice_name, repo_id)
}

/// Generate audio with MLX compilation enabled for decoder kernel fusion.
///
/// This is the key optimization: `enable_compile()` tells MLX to fuse
/// Metal GPU kernels in the decoder pass, reducing dispatch overhead.
pub fn generate_compiled(
    model: &mut KokoroModel,
    phonemes: &str,
    voice: &Array,
    speed: f32,
) -> voice_tts::Result<Array> {
    mlx_rs::transforms::compile::enable_compile();
    let result = voice_tts::generate(model, phonemes, voice, speed);
    mlx_rs::transforms::compile::disable_compile();
    result
}

/// Convert MLX Array to Vec<f32> samples.
pub fn array_to_samples(audio: &Array) -> Vec<f32> {
    audio.as_slice::<f32>().to_vec()
}

/// Phonemize Italian text using espeak-ng.
pub fn phonemize_it(text: &str) -> String {
    let mut child = Command::new("espeak-ng")
        .args(["-v", "it", "--ipa", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("espeak-ng not found — install with: brew install espeak-ng");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .replace('\n', " ")
}

/// Save audio samples as 16-bit PCM WAV.
pub fn save_wav(samples: &[f32], path: &Path, sample_rate: u32) -> std::io::Result<()> {
    let data_size = samples.len() * 2;
    let mut bytes = Vec::with_capacity(44 + data_size);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_size as u32).to_le_bytes());
    for &s in samples {
        let pcm = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    std::fs::write(path, bytes)
}
