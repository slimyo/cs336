"""scripts/make_golden.py — 把 M1/M2 冻结成金标准数据，供 M3/M5 的 kernel 逐项对照。

输入先舍入到 bf16（kernel 吃 bf16），参考一律用【同样被舍入过的输入】在 float64 上算，这样 kernel 与参考的
差异只来自 kernel 自己的数值（bf16 的 P、fp32 累加等），不来自输入舍入。

产出 golden/fa_golden.pt，每个配置包含：
  Q, K, V, dO (bf16)；O, L, D, dQ, dK, dV (float64)；
  S00, P00（第一个 Q tile × 第一个 K tile 的分数与概率，float64）；
  m_hist, l_hist（第一个 Q tile 每处理完一个 K tile 之后的 m、l，float64）——3c/3e 逐步对照用

用法：python scripts/make_golden.py
"""
import math
from pathlib import Path

import torch

from res.flash_attn_pytorch import FlashAttnPytorch, NaiveAttentionFn

CONFIGS = [  # (B, N, d, Bq, Bk)：最小调试形状 / 单 tile / 多 tile 且不整除
    (1, 16, 16, 16, 16),
    (1, 64, 64, 64, 64),
    (2, 100, 64, 64, 64),
]


def first_q_tile_history(Q, K, V, causal, Bq, Bk):
    """独立于被测实现的小循环：记录第一个 Q tile 的 S00、P00 以及每个 K tile 之后的 (m, l)。"""
    B, N, d = Q.shape
    qe = min(Bq, N)
    Qt = Q[:, :qe]
    m = torch.full((B, qe), float("-inf"), dtype=Q.dtype)
    l = torch.zeros((B, qe), dtype=Q.dtype)
    m_hist, l_hist, S00, P00 = [], [], None, None
    for ks in range(0, N, Bk):
        if causal and ks > qe - 1:
            break
        ke = min(ks + Bk, N)
        S = (Qt @ K[:, ks:ke].transpose(-2, -1)) / math.sqrt(d)
        if causal:
            qi = torch.arange(0, qe)[:, None]; kj = torch.arange(ks, ke)[None, :]
            S = S.masked_fill(qi < kj, float("-inf"))
        m_new = torch.maximum(m, S.amax(-1))
        P = torch.exp(S - m_new[..., None])
        l = torch.exp(m - m_new) * l + P.sum(-1)
        m = m_new
        m_hist.append(m.clone()); l_hist.append(l.clone())
        if ks == 0:
            S00, P00 = S.clone(), P.clone()
    return S00, P00, torch.stack(m_hist), torch.stack(l_hist)


def main():
    torch.manual_seed(1234)
    out = {}
    for B, N, d, Bq, Bk in CONFIGS:
        for causal in (False, True):
            Qb, Kb, Vb, dOb = (torch.randn(B, N, d).bfloat16() for _ in range(4))
            Q, K, V, dO = (t.double() for t in (Qb, Kb, Vb, dOb))       # 同样被舍入过的输入，升到 float64
            Qg, Kg, Vg = (t.clone().requires_grad_() for t in (Q, K, V))
            O = NaiveAttentionFn.apply(Qg, Kg, Vg, causal)               # M0 作为最终金标准
            O.backward(dO)
            S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
            if causal:
                S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1), float("-inf"))
            L = torch.logsumexp(S, -1)
            D = (dO * O.detach()).sum(-1)

            # 与你的 M1/M2 实现对账（L 按契约是 fp32，容差放宽）
            Qf, Kf, Vf = (t.clone().requires_grad_() for t in (Q, K, V))
            Of = FlashAttnPytorch.apply(Qf, Kf, Vf, causal, Bq, Bk)
            Of.backward(dO)
            for a, b, n in ((Of, O.detach(), "O"), (Qf.grad, Qg.grad, "dQ"), (Kf.grad, Kg.grad, "dK"), (Vf.grad, Vg.grad, "dV")):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6, msg=lambda m, x=n: f"M1/M2 与 M0 不一致 {x}: {m}")

            S00, P00, m_hist, l_hist = first_q_tile_history(Q, K, V, causal, Bq, Bk)
            out[f"B{B}_N{N}_d{d}_causal{int(causal)}"] = dict(
                Q=Qb, K=Kb, V=Vb, dO=dOb, O=O.detach(), L=L, D=D, dQ=Qg.grad, dK=Kg.grad, dV=Vg.grad,
                S00=S00, P00=P00, m_hist=m_hist, l_hist=l_hist, Bq=Bq, Bk=Bk)
    Path("golden").mkdir(exist_ok=True)
    torch.save(out, "golden/fa_golden.pt")
    print(f"已保存 {len(out)} 个配置 -> golden/fa_golden.pt：{list(out)}")


if __name__ == "__main__":
    main()