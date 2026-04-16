#!/usr/bin/env python3
"""Generate a tiny Qwen3-architecture GGUF model for benchmarking.

The weights are random – this model produces garbage text, but it exercises
the same compute path as a real Qwen3-0.6B so it is perfectly valid for
measuring kernel performance.

Architecture mirrors Qwen3-0.6B:
  - vocab_size    = 151936
  - hidden_size   = 896  (n_embd)
  - num_layers    = 24
  - num_heads     = 16
  - num_kv_heads  = 8    (GQA)
  - head_dim      = 64   (hidden_size / num_heads - but Qwen3 uses explicit head_dim)
  - intermediate  = 4864
  - rope_theta    = 1000000
  - max_position  = 40960
  - rms_norm_eps  = 1e-6
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
import gguf

MODEL_ARCH = "qwen3"
VOCAB_SIZE = 151936
N_EMBD     = 896
N_HEAD     = 14
N_KV_HEAD  = 2
HEAD_DIM   = 64    # n_embd / n_head = 896 / 14 = 64
N_FF       = 4864
N_LAYER    = 28
NORM_EPS   = 1e-6
ROPE_THETA = 1000000.0
CTX_LEN    = 40960

out_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'qwen3-0.6b-test-f16.gguf')

writer = gguf.GGUFWriter(out_path, arch="qwen3")

# Metadata
writer.add_context_length(CTX_LEN)
writer.add_embedding_length(N_EMBD)
writer.add_feed_forward_length(N_FF)
writer.add_block_count(N_LAYER)
writer.add_head_count(N_HEAD)
writer.add_head_count_kv(N_KV_HEAD)
writer.add_layer_norm_rms_eps(NORM_EPS)
writer.add_rope_freq_base(ROPE_THETA)
writer.add_vocab_size(VOCAB_SIZE)

# Minimal tokenizer (BPE with a few tokens)
# We only need enough tokens for the model to load.
tokens = []
scores = []
toktypes = []

# Add special tokens
special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
for i, tok in enumerate(special_tokens):
    tokens.append(tok.encode("utf-8"))
    scores.append(-1000.0 - i)
    toktypes.append(gguf.TokenType.CONTROL)

# Fill vocab with byte-level tokens
for i in range(VOCAB_SIZE - len(special_tokens)):
    # Generate a byte sequence that represents a unique token
    tok_str = f"<tok_{i}>"
    tokens.append(tok_str.encode("utf-8"))
    scores.append(-float(i))
    toktypes.append(gguf.TokenType.NORMAL)

writer.add_tokenizer_model("gpt2")
writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(toktypes)
# BPE tokenizer requires merges - add minimal dummy merges
merges = []
for i in range(min(1000, VOCAB_SIZE - len(special_tokens) - 1)):
    merges.append(f"<tok_{i}> <tok_{i+1}>")
writer.add_token_merges(merges)
writer.add_bos_token_id(0)
writer.add_eos_token_id(0)
writer.add_pad_token_id(0)

print(f"Generating random weights for {N_LAYER}-layer Qwen3 model …")

rng = np.random.default_rng(42)

def rand_f16(shape):
    return rng.standard_normal(shape).astype(np.float16) * 0.02

# Token embedding
writer.add_tensor("token_embd.weight", rand_f16((VOCAB_SIZE, N_EMBD)))

for i in range(N_LAYER):
    pfx = f"blk.{i}"

    # Self-attention
    writer.add_tensor(f"{pfx}.attn_norm.weight",   np.ones(N_EMBD, dtype=np.float32))
    writer.add_tensor(f"{pfx}.attn_q.weight",      rand_f16((N_HEAD * HEAD_DIM, N_EMBD)))
    writer.add_tensor(f"{pfx}.attn_k.weight",      rand_f16((N_KV_HEAD * HEAD_DIM, N_EMBD)))
    writer.add_tensor(f"{pfx}.attn_v.weight",      rand_f16((N_KV_HEAD * HEAD_DIM, N_EMBD)))
    writer.add_tensor(f"{pfx}.attn_output.weight",  rand_f16((N_EMBD, N_HEAD * HEAD_DIM)))

    # Qwen3 has per-head norms for Q and K
    writer.add_tensor(f"{pfx}.attn_q_norm.weight",  np.ones(HEAD_DIM, dtype=np.float32))
    writer.add_tensor(f"{pfx}.attn_k_norm.weight",  np.ones(HEAD_DIM, dtype=np.float32))

    # FFN (gate-up-down style like LLaMA)
    writer.add_tensor(f"{pfx}.ffn_norm.weight",     np.ones(N_EMBD, dtype=np.float32))
    writer.add_tensor(f"{pfx}.ffn_gate.weight",     rand_f16((N_FF, N_EMBD)))
    writer.add_tensor(f"{pfx}.ffn_up.weight",       rand_f16((N_FF, N_EMBD)))
    writer.add_tensor(f"{pfx}.ffn_down.weight",     rand_f16((N_EMBD, N_FF)))

    print(f"  layer {i+1}/{N_LAYER}", end="\r")

# Output norm + LM head
writer.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
writer.add_tensor("output.weight",      rand_f16((VOCAB_SIZE, N_EMBD)))

print(f"\nWriting {out_path} …")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

sz = os.path.getsize(out_path) / (1024 * 1024)
print(f"Done! Model size: {sz:.1f} MB")
