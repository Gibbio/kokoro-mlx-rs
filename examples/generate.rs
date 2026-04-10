use std::path::Path;
use std::time::Instant;

fn main() {
    let text = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ");

    if text.is_empty() {
        eprintln!("Usage: cargo run --release --example generate -- \"testo da sintetizzare\"");
        std::process::exit(1);
    }

    eprintln!("Loading model...");
    let t = Instant::now();
    let mut model = kokoro_mlx_rs::load_model("prince-canuma/Kokoro-82M")
        .expect("failed to load model");
    eprintln!("  Model loaded in {:?}", t.elapsed());

    let voice = kokoro_mlx_rs::load_voice("af_bella", None)
        .expect("failed to load voice");

    eprintln!("Phonemizing...");
    let phonemes = kokoro_mlx_rs::phonemize_it(&text);
    eprintln!("  Phonemes: {phonemes}");

    eprintln!("Generating (with mx.compile)...");
    let t = Instant::now();
    let audio = kokoro_mlx_rs::generate_compiled(&mut model, &phonemes, &voice, 1.0)
        .expect("generation failed");
    let samples = kokoro_mlx_rs::array_to_samples(&audio);
    let gen_time = t.elapsed();

    let duration_s = samples.len() as f64 / 24000.0;
    let rtf = duration_s / gen_time.as_secs_f64();

    eprintln!("  Generated {:.1}s audio in {:?} ({:.1}x realtime)", duration_s, gen_time, rtf);

    let out_path = Path::new("output.wav");
    kokoro_mlx_rs::save_wav(&samples, out_path, 24000).expect("failed to save WAV");
    eprintln!("  Saved to {}", out_path.display());
}
