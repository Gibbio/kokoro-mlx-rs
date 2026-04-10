//! # kokoro-mlx-rs
//!
//! Kokoro-82M TTS on Apple Silicon via MLX.
//! Faithful port of [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s Python implementation,
//! using [`mlx-rs`](https://github.com/oxideai/mlx-rs) with `mx.compile` for kernel fusion.
//!
//! Target: match Python MLX performance (~1100ms for medium phrases on M4).

pub mod config;
pub mod modules;
pub mod istftnet;
pub mod albert;
pub mod model;
pub mod weights;
