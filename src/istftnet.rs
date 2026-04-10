//! iSTFTNet vocoder: ConvWeighted, InstanceNorm, AdaIN, Generator, Decoder.
//! Port of mlx_audio/tts/models/kokoro/istftnet.py
//!
//! This is the performance-critical module (~92% of inference time).
//! Must use mx.compile for kernel fusion to match Python MLX speed.

// TODO: implement
// - weight_norm, compute_norm
// - ConvWeighted (Conv1d with weight normalization)
// - InstanceNorm1d, AdaIN1d
// - AdaINResBlock1 (Snake activation + conv + AdaIN)
// - MLXSTFT (STFT transform + inverse)
// - SineGen, SourceModuleHnNSF
// - Generator (upsample + resblocks + iSTFT)
// - Decoder (encode + decode blocks + Generator)
// - AdainResBlk1d, UpSample1d
