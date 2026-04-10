use std::time::Instant;

const PHRASES: &[(&str, &str)] = &[
    ("tiny",   "La stanza era silenziosa."),
    ("short",  "La stanza era silenziosa, ma non immobile. Un leggero fruscio proveniva dalla finestra."),
    ("medium", "La stanza era silenziosa, ma non immobile. Un leggero fruscio proveniva dalla finestra socchiusa, dove la brezza notturna faceva ondeggiare la tenda di lino bianco."),
    ("long",   "La stanza era silenziosa, ma non immobile. Un leggero fruscio proveniva dalla finestra socchiusa, dove la brezza notturna faceva ondeggiare la tenda di lino bianco. Sul tavolo, accanto a una tazza di tè ormai freddo, giaceva un libro aperto a metà, le pagine ingiallite illuminate dalla luce calda di una lampada."),
];

const SAMPLE_RATE: f64 = 24000.0;

fn run_bench(label: &str, model: &mut kokoro_mlx_rs::KokoroModel, voice: &mlx_rs::Array, compiled: bool) {
    println!(
        "  {:<8} {:>5} {:>8} {:>8} {:>8} {:>6} {:>5}",
        "Label", "Chars", "Phonem", "Infer", "Total", "Audio", "RTFx"
    );
    println!("  {}", "-".repeat(60));

    for &(phrase_label, text) in PHRASES {
        let t0 = Instant::now();
        let ph = kokoro_mlx_rs::phonemize_it(text);
        let ph_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let audio = if compiled {
            kokoro_mlx_rs::generate_compiled(model, &ph, voice, 1.0)
                .expect("generate failed")
        } else {
            voice_tts::generate(model, &ph, voice, 1.0)
                .expect("generate failed")
        };

        let samples = kokoro_mlx_rs::array_to_samples(&audio);
        let inf_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let total_ms = ph_ms + inf_ms;
        let dur_s = samples.len() as f64 / SAMPLE_RATE;
        let rtf = dur_s / (inf_ms / 1000.0);

        println!(
            "  {:<8} {:>5} {:>7.0}ms {:>7.0}ms {:>7.0}ms {:>5.1}s {:>5.1}x",
            phrase_label, text.len(), ph_ms, inf_ms, total_ms, dur_s, rtf,
        );
    }
}

fn main() {
    let t = Instant::now();
    let mut model = kokoro_mlx_rs::load_model("prince-canuma/Kokoro-82M")
        .expect("failed to load model");
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;

    let voice = kokoro_mlx_rs::load_voice("af_bella", None)
        .expect("failed to load voice");

    // Warmup (uncompiled)
    eprint!("  Warmup...");
    let ph = kokoro_mlx_rs::phonemize_it(PHRASES[0].1);
    let _ = voice_tts::generate(&mut model, &ph, &voice, 1.0);
    eprintln!(" done");

    // Without compile (FIRST — before compile contaminates the graph)
    println!();
    println!("  {}", "=".repeat(65));
    println!("  voice-tts 0.2 mlx-rs (NO compile) — load: {load_ms:.0}ms");
    println!("  {}", "=".repeat(65));
    println!();
    run_bench("no-compile", &mut model, &voice, false);

    // With compile (enable_compile on decoder)
    // Warmup compiled
    let _ = kokoro_mlx_rs::generate_compiled(&mut model, &ph, &voice, 1.0);

    println!();
    println!("  {}", "=".repeat(65));
    println!("  kokoro-mlx-rs (WITH enable_compile decoder) — load: {load_ms:.0}ms");
    println!("  {}", "=".repeat(65));
    println!();
    run_bench("compiled", &mut model, &voice, true);

    println!();
}
