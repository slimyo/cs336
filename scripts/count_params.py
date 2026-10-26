import torch
from cs336_basics.model import BasicsTransformerLM
with torch.device("meta"):
    m = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=768,
        num_layers=12, num_heads=12, d_ff=3072, rope_theta=10000.0)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M")
