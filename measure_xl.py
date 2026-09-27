"""
The section-06 measurement, repeated on GPT-2 XL (1.5B) — the model ROME and MEMIT
were actually developed on, and 12x the size of the GPT-2 small run.

Layers 14 and 17 sit inside MEMIT's published critical range for GPT-2 XL ({13..17});
layer 36 is well outside it, as a control.

Changes from the small run, forced by scale:
  - forward passes on MPS in float32, Gramian accumulated in float64 on CPU
  - C^-1 applied by Cholesky solve rather than eigendecomposition (same answer, far cheaper)
  - eigenvalues only (no eigenvectors) for the spectrum diagnostics
"""
import json, sys, time, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
MODEL  = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
LAYERS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["14", "17", "36"])]
NP_, NF_, NHELD = 8, 2, 6000
rng = np.random.default_rng(20260927)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL, dtype=torch.float32)
model.train(False); model.to(dev)
D_IN  = model.config.n_inner or 4 * model.config.n_embd
D_OUT = model.config.n_embd
print(f"{MODEL} on {dev}: {model.config.n_layer} layers, key dim {D_IN}, value dim {D_OUT}", flush=True)

ids = tok(open("corpus.txt").read(), return_tensors="pt").input_ids[0]
CH = 512
chunks = [ids[i:i+CH] for i in range(0, len(ids) - CH, CH)]
split = int(len(chunks) * 0.72)
print(f"tokens {len(ids)} | chunks {len(chunks)} -> fit {split} / held {len(chunks)-split} "
      f"| samples per key dimension: {split*CH/D_IN:.1f}", flush=True)

grab = {}
handles = [model.transformer.h[L].mlp.c_proj.register_forward_hook(
    (lambda L: (lambda m, i, o: grab.__setitem__(L, i[0].detach()[0].float().cpu().numpy())))(L))
    for L in LAYERS]

def sweep(chs, tag):
    C = {L: np.zeros((D_IN, D_IN)) for L in LAYERS}
    keep = {L: [] for L in LAYERS}
    n, t0 = 0, time.time()
    for ci, ch in enumerate(chs):
        model(ch.unsqueeze(0).to(dev))
        for L in LAYERS:
            K = grab[L]
            C[L] += (K.T @ K).astype(np.float64)      # float32 gemm, float64 accumulator
            idx = rng.choice(K.shape[0], size=min(24, K.shape[0]), replace=False)
            keep[L].append(K[idx])
        n += ch.shape[0]
        if ci % 20 == 0:
            el = time.time() - t0
            print(f"  {tag} {ci}/{len(chs)}  {el:.0f}s elapsed, ~{el/max(ci,1)*(len(chs)-ci):.0f}s left", flush=True)
    for L in LAYERS: C[L] /= n
    return C, {L: np.concatenate(keep[L]).astype(np.float64) for L in LAYERS}, n

C, fitkeys, nfit = sweep(chunks[:split], "fit")
_, heldkeys, nheld = sweep(chunks[split:], "held")
for h in handles: h.remove()
model.to("cpu")

def solve_exact(K, R, apply_minv, ridge=1e-10):
    """min tr(D M D^T) s.t. D K = R  ->  D = R (K^T M^-1 K)^-1 K^T M^-1"""
    WK = apply_minv(K)
    G = K.T @ WK
    G = G + ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    return (WK @ np.linalg.solve(G, R.T)).T

def leak(Dm, H):
    v = H @ Dm.T
    s = np.sqrt((v * v).sum(1))
    return float(s.max()), float(np.sqrt((s * s).mean()))

report = {}
for L in LAYERS:
    print(f"\n=== layer {L} ===", flush=True)
    Cl, t0 = C[L], time.time()
    w = np.clip(np.linalg.eigvalsh(Cl)[::-1], 0, None)
    tot = w.sum(); cum = np.cumsum(w) / tot
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    eff_rank = float(tot**2 / (w**2).sum())
    print(f"  eigvalsh {time.time()-t0:.0f}s", flush=True)

    Wm = model.transformer.h[L].mlp.c_proj.weight.double().numpy().T
    FK, HK = fitkeys[L], heldkeys[L]
    HK = HK[rng.choice(len(HK), size=min(NHELD, len(HK)), replace=False)]
    pick = rng.choice(len(FK), size=NP_ + 2*NF_, replace=False)
    Kp, Knew, Kdon = FK[pick[:NP_]], FK[pick[NP_:NP_+NF_]], FK[pick[NP_+NF_:]]
    K = np.concatenate([Kp, Knew]).T
    Rr = np.concatenate([np.zeros((NP_, D_OUT)), (Kdon - Knew) @ Wm.T]).T
    Kn = K / np.linalg.norm(K, axis=0, keepdims=True)
    coh = (Kn.T @ Kn)[~np.eye(K.shape[1], dtype=bool)]

    res = {"eig": {"kappa_raw": float(w[0]/max(w[-1],1e-300)), "kappa_99": float(w[0]/w[k99-1]),
                   "eff_rank": eff_rank, "k99": k99, "dim": int(D_IN), "top_share": float(w[0]/tot),
                   "cum25": float(cum[24]), "cum100": float(cum[99])},
           "coherence": {"mean": float(coh.mean()), "max": float(coh.max())},
           "A_exact": {}, "B_memit": {}}
    mean_w = float(np.trace(Cl) / D_IN)
    for lam in [1e-3, 1e-2, 1e-1]:
        Creg = Cl + (lam * mean_w) * np.eye(D_IN)
        De = solve_exact(K, Rr, lambda X: X)
        Dg = solve_exact(K, Rr, lambda X: np.linalg.solve(Creg, X))
        me, re_ = leak(De, HK); mg, rg = leak(Dg, HK)
        fit = max(float(np.abs(De @ K - Rr).max()), float(np.abs(Dg @ K - Rr).max()))
        Qe = np.linalg.qr(De.T)[0]; Qg = np.linalg.qr(Dg.T)[0]
        ang = float(np.degrees(np.arccos(np.clip(np.linalg.svd(Qe.T @ Qg, compute_uv=False), -1, 1).min())))
        res["A_exact"][f"{lam:g}"] = {"gain_max": me/mg, "gain_rms": re_/rg, "fit_err": fit,
                                      "max_principal_angle_deg": ang}
        print(f"  A lam={lam:<6g} euclid/gram  max {me/mg:6.2f}x   rms {re_/rg:6.2f}x  (fit {fit:.1e}, angle {ang:.0f} deg)", flush=True)

    K1, R1 = Knew.T, Rr[:, NP_:]
    sc = float(np.trace(K1 @ K1.T))
    for lam in [1e-2, 1e-1, 1.0]:
        Ac = K1 @ K1.T + lam * Cl * (sc / max(float(np.trace(Cl)), 1e-30))
        Ai = K1 @ K1.T + lam * np.eye(D_IN) * (sc / D_IN)
        Dc = R1 @ np.linalg.solve(Ac, K1).T
        Di = R1 @ np.linalg.solve(Ai, K1).T
        mc, rc = leak(Dc, HK); mi, ri = leak(Di, HK)
        den = max(float(np.abs(R1).max()), 1e-30)
        fc = float(np.abs(Dc @ K1 - R1).max())/den; fi = float(np.abs(Di @ K1 - R1).max())/den
        res["B_memit"][f"{lam:g}"] = {"gain_max": mi/mc, "gain_rms": ri/rc, "cov_relfit": fc, "eye_relfit": fi}
        print(f"  B lam={lam:<6g} identity/covariance  max {mi/mc:6.2f}x   rms {ri/rc:6.2f}x  "
              f"(rel. edit error cov {fc:.4f} / eye {fi:.4f})", flush=True)

    if L == LAYERS[1]:
        idx = np.unique(np.round(np.logspace(0, np.log10(D_IN), 150)).astype(int) - 1)
        idx = idx[(idx >= 0) & (idx < D_IN)]
        res["spectrum"] = {"idx": [int(i) for i in idx],
                           "lam": [float(f"{w[i]:.6g}") for i in idx],
                           "cum": [round(float(cum[i]), 5) for i in idx]}
    report[str(L)] = res
    print(f"  spectrum: eff.rank {eff_rank:.1f} of {D_IN} | top {100*w[0]/tot:.1f}% | 25 dirs hold "
          f"{100*cum[24]:.1f}% | kappa_99 {w[0]/w[k99-1]:.0f} | coherence {coh.mean():.3f}", flush=True)
    del Cl

report["meta"] = {"model": MODEL, "d_in": int(D_IN), "d_out": int(D_OUT), "n_layer": model.config.n_layer,
                  "tokens_fit": int(nfit), "tokens_held": int(nheld), "samples_per_dim": round(nfit/D_IN, 1),
                  "NP": NP_, "NF": NF_, "layers": LAYERS, "device": dev,
                  "n_held_keys_actual": int(min(NHELD, len(heldkeys[LAYERS[0]])))}
json.dump(report, open(f"measure_{MODEL}.json", "w"), indent=1)
print(f"\nwrote measure_{MODEL}.json", flush=True)
