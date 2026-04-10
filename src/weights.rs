//! Weight loading and sanitization from safetensors.
//! Handles PyTorch -> MLX weight key remapping (LSTM, Conv, LayerNorm).

// TODO: implement
// - load_safetensors + sanitize
// - LSTM weight remapping (weight_ih_l0 -> Wx_forward etc.)
// - Conv weight transpose where needed
// - HuggingFace Hub download integration
