"""
Effective conditioning of a real transformer layer, and whether the section-06 gap survives.

The MEMIT "key" for an MLP is the input to the down-projection (post-GELU, dim 3072 in GPT-2).
The matrix MEMIT writes as K0 K0^T is the uncentered second moment of those keys over a corpus.
We measure its spectrum, then run two experiments on real activations:

  A  exact-constrained (the section-06 setup): min tr(D M D^T) s.t. D K = R, for M = I and M = C.
  B  MEMIT as published vs the same solve with its covariance term replaced by identity.

Leakage is evaluated on keys from documents that contributed to NEITHER C nor the constraints.
"""
import json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
LAYERS = [3, 6, 9]
NP_, NF_ = 8, 2          # protected keys, new facts  (matches the toy in section 06)
NHELD = 6000
rng = np.random.default_rng(20260927)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.train(False)                     # inference mode
D_IN = model.config.n_inner or 4 * model.config.n_embd
D_OUT = model.config.n_embd
print(f"gpt2: {model.config.n_layer} layers, key dim {D_IN}, value dim {D_OUT}", flush=True)

text = open("corpus.txt").read()
ids = tok(text, return_tensors="pt").input_ids[0]
CH = 512
chunks = [ids[i:i+CH] for i in range(0, len(ids) - CH, CH)]
split = int(len(chunks) * 0.72)
fit_chunks, held_chunks = chunks[:split], chunks[split:]
print(f"tokens {len(ids)} | chunks {len(chunks)} -> fit {len(fit_chunks)} / held {len(held_chunks)}", flush=True)

grab = {}
def hook(name):
    def f(mod, inp, out): grab[name] = inp[0].detach()[0].double().numpy()
    return f
handles = [model.transformer.h[L].mlp.c_proj.register_forward_hook(hook(L)) for L in LAYERS]

def sweep(chs):
    C = {L: np.zeros((D_IN, D_IN)) for L in LAYERS}
    keep = {L: [] for L in LAYERS}
    n = 0
    for ci, ch in enumerate(chs):
        model(ch.unsqueeze(0))
        for L in LAYERS:
            K = grab[L]
            C[L] += K.T @ K
            idx = rng.choice(K.shape[0], size=min(24, K.shape[0]), replace=False)
            keep[L].append(K[idx].astype(np.float32))
        n += ch.shape[0]
        if ci % 20 == 0: print(f"  chunk {ci}/{len(chs)}", flush=True)
    for L in LAYERS: C[L] /= n
    return C, {L: np.concatenate(keep[L]) for L in LAYERS}, n

print("pass 1 - fitting C on the preserved corpus", flush=True)
C, fitkeys, nfit = sweep(fit_chunks)
print("pass 2 - held-out keys", flush=True)
_, heldkeys, nheld = sweep(held_chunks)
for h in handles: h.remove()

def solve_exact(K, R, apply_minv, ridge=1e-10):
    """min tr(D M D^T) s.t. D K = R   ->   D = R (K^T M^-1 K)^-1 K^T M^-1"""
    WK = apply_minv(K)
    G = K.T @ WK
    G = G + ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    Y = np.linalg.solve(G, R.T)
    return (WK @ Y).T

def leak(Dm, H):
    v = H @ Dm.T
    s = np.sqrt((v * v).sum(1))
    return float(s.max()), float(np.sqrt((s * s).mean()))

report = {}
for L in LAYERS:
    print(f"\n=== layer {L} ===", flush=True)
    Cl = C[L]
    w, V = np.linalg.eigh(Cl)
    w = np.clip(w[::-1], 0, None); V = V[:, ::-1]
    tot = w.sum(); cum = np.cumsum(w) / tot
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    k999 = int(np.searchsorted(cum, 0.999) + 1)
    eff_rank = float(tot**2 / (w**2).sum())
    kap_raw = float(w[0] / max(w[-1], 1e-300))
    kap_99 = float(w[0] / w[k99 - 1])
    kap_tail = float(w[1] / w[k99 - 1])

    Wm = model.transformer.h[L].mlp.c_proj.weight.double().numpy().T
    FK = fitkeys[L].astype(np.float64)
    HK = heldkeys[L].astype(np.float64)
    HK = HK[rng.choice(len(HK), size=min(NHELD, len(HK)), replace=False)]

    pick = rng.choice(len(FK), size=NP_ + 2 * NF_, replace=False)
    Kp, Knew, Kdon = FK[pick[:NP_]], FK[pick[NP_:NP_+NF_]], FK[pick[NP_+NF_:]]
    K = np.concatenate([Kp, Knew]).T
    Rr = np.concatenate([np.zeros((NP_, D_OUT)), (Kdon - Knew) @ Wm.T]).T

    Kn = K / np.linalg.norm(K, axis=0, keepdims=True)
    coh = Kn.T @ Kn
    coh_off = coh[~np.eye(len(coh), dtype=bool)]

    res = {"eig": {"kappa_raw": kap_raw, "kappa_99": kap_99, "kappa_tail": kap_tail,
                   "eff_rank": eff_rank, "k99": k99, "k999": k999, "dim": int(D_IN),
                   "top_share": float(w[0] / tot),
                   "spectrum_sample": [float(x) for x in w[::max(1, D_IN // 60)][:64]]},
           "coherence": {"mean": float(coh_off.mean()), "max": float(coh_off.max())},
           "A_exact": {}, "B_memit": {}}

    Vt = V.T
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        inv_d = 1.0 / (w + lam * float(w.mean()))
        def apply_cinv(X, d=inv_d): return V @ (d[:, None] * (Vt @ X))
        De = solve_exact(K, Rr, lambda X: X)
        Dg = solve_exact(K, Rr, apply_cinv)
        me, re_ = leak(De, HK); mg, rg = leak(Dg, HK)
        fit = max(float(np.abs(De @ K - Rr).max()), float(np.abs(Dg @ K - Rr).max()))
        Qe = np.linalg.qr(De.T)[0]; Qg = np.linalg.qr(Dg.T)[0]
        sv = np.clip(np.linalg.svd(Qe.T @ Qg, compute_uv=False), -1, 1)
        ang = float(np.degrees(np.arccos(sv.min())))
        res["A_exact"][f"{lam:g}"] = {"euclid_max": me, "gram_max": mg, "euclid_rms": re_, "gram_rms": rg,
                                      "gain_max": me / mg, "gain_rms": re_ / rg,
                                      "fit_err": fit, "max_principal_angle_deg": ang}
        print(f"  A lam={lam:<6g} euclid/gram  max {me/mg:6.2f}x   rms {re_/rg:6.2f}x   "
              f"(fit {fit:.1e}, angle {ang:.1f} deg)", flush=True)

    K1, R1 = Knew.T, Rr[:, NP_:]
    sc = float(np.trace(K1 @ K1.T))
    for lam in [1e-2, 1e-1, 1.0, 10.0]:
        A_c = K1 @ K1.T + lam * Cl * (sc / max(float(np.trace(Cl)), 1e-30))
        A_i = K1 @ K1.T + lam * np.eye(D_IN) * (sc / D_IN)
        Dc = R1 @ np.linalg.solve(A_c, K1).T
        Di = R1 @ np.linalg.solve(A_i, K1).T
        mc, rc = leak(Dc, HK); mi, ri = leak(Di, HK)
        den = max(float(np.abs(R1).max()), 1e-30)
        fc = float(np.abs(Dc @ K1 - R1).max()) / den
        fi = float(np.abs(Di @ K1 - R1).max()) / den
        res["B_memit"][f"{lam:g}"] = {"cov_max": mc, "eye_max": mi, "cov_rms": rc, "eye_rms": ri,
                                      "gain_max": mi / mc, "gain_rms": ri / rc,
                                      "cov_relfit": fc, "eye_relfit": fi}
        print(f"  B lam={lam:<6g} identity/covariance  max {mi/mc:6.2f}x   rms {ri/rc:6.2f}x   "
              f"(rel. edit error cov {fc:.3f} / eye {fi:.3f})", flush=True)
    report[str(L)] = res
    print(f"  spectrum: kappa_raw {kap_raw:.3e} | kappa_99 {kap_99:.1f} | eff.rank {eff_rank:.1f} of {D_IN}"
          f" | 99% energy in {k99} dirs | top direction holds {100*w[0]/tot:.1f}%", flush=True)
    print(f"  key coherence: mean |cos| {coh_off.mean():.3f}, max {coh_off.max():.3f}", flush=True)

report["meta"] = {"model": "gpt2", "d_in": int(D_IN), "d_out": int(D_OUT), "tokens_fit": int(nfit),
                  "tokens_held": int(nheld), "n_held_keys": int(NHELD), "NP": NP_, "NF": NF_,
                  "corpus_chars": len(text)}
json.dump(report, open("measure.json", "w"), indent=1)
print("\nwrote measure.json", flush=True)
