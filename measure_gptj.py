"""
The section-06 measurement on GPT-J 6B: 4x GPT-2 XL, key dimension 16384.

Architecture differences that matter and are easy to get wrong:
  - the MLP down-projection is `mlp.fc_out`, an nn.Linear, not GPT-2's Conv1D `mlp.c_proj`
  - nn.Linear stores weight as (out, in); Conv1D stores it as (in, out). Only one needs a transpose.
  - MEMIT's published critical range for GPT-J is {3..8}, so 5 and 8 are in range and 20 is a control.
  - weights are fp16 (36 GB of RAM will not hold 6B params in fp32 alongside 16384^2 Gramians);
    activations are cast to float32 on capture and the Gramian accumulates in float64.
"""
import gc, json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
NAME = "EleutherAI/gpt-j-6b"
LAYERS = [5, 8, 20]
NP_, NF_ = 8, 2
LAM_A, LAM_B = [1e-3, 1e-1], [1e-1, 1.0]
rng = np.random.default_rng(20260927)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(NAME)
try:
    model = AutoModelForCausalLM.from_pretrained(NAME, revision="float16", dtype=torch.float16)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16)
model.train(False); model.to(dev)
cfg = model.config
D_IN = cfg.n_inner or 4 * cfg.n_embd
D_OUT = cfg.n_embd
down = [model.transformer.h[L].mlp.fc_out for L in LAYERS]
assert down[0].weight.shape == (D_OUT, D_IN), f"unexpected weight shape {tuple(down[0].weight.shape)}"
print(f"{NAME} on {dev}: {cfg.n_layer} layers, key dim {D_IN}, value dim {D_OUT}, fp16", flush=True)

ids = tok(open("corpus.txt").read(), return_tensors="pt").input_ids[0]
CH = 512
chunks = [ids[i:i+CH] for i in range(0, len(ids) - CH, CH)]
# each 16384x16384 Gramian update costs ~2s per chunk per layer; cap the sweep so the run stays
# inside an hour while keeping samples-per-dimension above what MEMIT itself uses for GPT-J (~6)
MAXCH = 760
if len(chunks) > MAXCH:
    print(f"capping {len(chunks)} chunks to {MAXCH} to bound runtime", flush=True)
    chunks = chunks[:MAXCH]
split = int(len(chunks) * 0.72)
print(f"tokens {len(ids)} | chunks {len(chunks)} -> fit {split} / held {len(chunks)-split} "
      f"| samples per key dimension: {split*CH/D_IN:.1f}", flush=True)

grab = {}
handles = [d.register_forward_hook(
    (lambda L: (lambda m, i, o: grab.__setitem__(L, i[0].detach()[0].float().cpu().numpy())))(L))
    for L, d in zip(LAYERS, down)]

def sweep(chs, tag):
    C = {L: np.zeros((D_IN, D_IN)) for L in LAYERS}
    keep = {L: [] for L in LAYERS}
    n, t0 = 0, time.time()
    for ci, ch in enumerate(chs):
        model(ch.unsqueeze(0).to(dev))
        for L in LAYERS:
            K = grab[L]
            C[L] += K.T @ K
            keep[L].append(K[rng.choice(K.shape[0], size=min(16, K.shape[0]), replace=False)])
        n += ch.shape[0]
        if ci % 20 == 0:
            el = time.time() - t0
            print(f"  {tag} {ci}/{len(chs)}  {el:.0f}s elapsed, ~{el/max(ci,1)*(len(chs)-ci):.0f}s left", flush=True)
    for L in LAYERS: C[L] /= n
    return C, {L: np.concatenate(keep[L]).astype(np.float64) for L in LAYERS}, n

C, fitkeys, nfit = sweep(chunks[:split], "fit")
_, heldkeys, nheld = sweep(chunks[split:], "held")
for h in handles: h.remove()
Wm = {L: d.weight.detach().float().cpu().numpy().astype(np.float64) for L, d in zip(LAYERS, down)}
del model, down, grab; gc.collect()
print("model released; starting linear algebra", flush=True)

def solve_exact(K, R, apply_minv, ridge=1e-10):
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

    FK, HK = fitkeys[L], heldkeys[L]
    pick = rng.choice(len(FK), size=NP_ + 2*NF_, replace=False)
    Kp, Knew, Kdon = FK[pick[:NP_]], FK[pick[NP_:NP_+NF_]], FK[pick[NP_+NF_:]]
    K = np.concatenate([Kp, Knew]).T
    Rr = np.concatenate([np.zeros((NP_, D_OUT)), (Kdon - Knew) @ Wm[L].T]).T
    Kn = K / np.linalg.norm(K, axis=0, keepdims=True)
    coh = (Kn.T @ Kn)[~np.eye(K.shape[1], dtype=bool)]

    res = {"eig": {"kappa_raw": float(w[0]/max(w[-1],1e-300)), "kappa_99": float(w[0]/w[k99-1]),
                   "eff_rank": eff_rank, "k99": k99, "dim": int(D_IN), "top_share": float(w[0]/tot),
                   "cum25": float(cum[24])},
           "coherence": {"mean": float(coh.mean()), "max": float(coh.max())},
           "A_exact": {}, "B_memit": {}}
    mean_w = float(np.trace(Cl) / D_IN)
    for lam in LAM_A:
        t1 = time.time()
        Ci = np.linalg.solve(Cl + (lam * mean_w) * np.eye(D_IN), K)
        De = solve_exact(K, Rr, lambda X: X)
        Dg = solve_exact(K, Rr, lambda X: Ci)
        me, re_ = leak(De, HK); mg, rg = leak(Dg, HK)
        fit = max(float(np.abs(De @ K - Rr).max()), float(np.abs(Dg @ K - Rr).max()))
        Qe = np.linalg.qr(De.T)[0]; Qg = np.linalg.qr(Dg.T)[0]
        ang = float(np.degrees(np.arccos(np.clip(np.linalg.svd(Qe.T @ Qg, compute_uv=False), -1, 1).min())))
        res["A_exact"][f"{lam:g}"] = {"gain_max": me/mg, "gain_rms": re_/rg, "fit_err": fit,
                                      "max_principal_angle_deg": ang}
        print(f"  A lam={lam:<6g} euclid/gram  max {me/mg:6.2f}x   rms {re_/rg:6.2f}x  "
              f"(fit {fit:.1e}, angle {ang:.0f} deg, {time.time()-t1:.0f}s)", flush=True)
        del Ci; gc.collect()

    K1, R1 = Knew.T, Rr[:, NP_:]
    sc = float(np.trace(K1 @ K1.T))
    for lam in LAM_B:
        t1 = time.time()
        Dc = R1 @ np.linalg.solve(K1 @ K1.T + lam * Cl * (sc / float(np.trace(Cl))), K1).T
        Di = R1 @ np.linalg.solve(K1 @ K1.T + lam * np.eye(D_IN) * (sc / D_IN), K1).T
        mc, rc = leak(Dc, HK); mi, ri = leak(Di, HK)
        den = max(float(np.abs(R1).max()), 1e-30)
        fc = float(np.abs(Dc @ K1 - R1).max())/den; fi = float(np.abs(Di @ K1 - R1).max())/den
        res["B_memit"][f"{lam:g}"] = {"gain_max": mi/mc, "gain_rms": ri/rc, "cov_relfit": fc, "eye_relfit": fi}
        print(f"  B lam={lam:<6g} identity/covariance  max {mi/mc:6.2f}x   rms {ri/rc:6.2f}x  "
              f"(rel. edit error cov {fc:.4f} / eye {fi:.4f}, {time.time()-t1:.0f}s)", flush=True)
        del Dc, Di; gc.collect()

    if L == LAYERS[1]:
        idx = np.unique(np.round(np.logspace(0, np.log10(D_IN), 150)).astype(int) - 1)
        idx = idx[(idx >= 0) & (idx < D_IN)]
        res["spectrum"] = {"idx": [int(i) for i in idx],
                           "lam": [float(f"{w[i]:.6g}") for i in idx],
                           "cum": [round(float(cum[i]), 5) for i in idx]}
    report[str(L)] = res
    print(f"  spectrum: eff.rank {eff_rank:.1f} of {D_IN} | top {100*w[0]/tot:.1f}% | 25 dirs hold "
          f"{100*cum[24]:.1f}% | kappa_99 {w[0]/w[k99-1]:.0f} | coherence {coh.mean():.3f}", flush=True)
    del C[L], Cl, w; gc.collect()

report["meta"] = {"model": "gpt-j-6b", "d_in": int(D_IN), "d_out": int(D_OUT), "n_layer": cfg.n_layer,
                  "tokens_fit": int(nfit), "tokens_held": int(nheld),
                  "samples_per_dim": round(nfit/D_IN, 1), "NP": NP_, "NF": NF_, "layers": LAYERS,
                  "device": dev, "dtype": "fp16 weights, fp32 activations, fp64 Gramian",
                  "n_held_keys": int(len(heldkeys[LAYERS[0]]))}
json.dump(report, open("measure_gptj.json", "w"), indent=1)
print("\nwrote measure_gptj.json", flush=True)
