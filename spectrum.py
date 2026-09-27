"""Full eigenvalue spectrum of the layer-6 key Gramian, for plotting."""
import json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
torch.set_grad_enabled(False)
L = 6
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2"); model.train(False)
D = model.config.n_inner or 4 * model.config.n_embd
ids = tok(open("corpus.txt").read(), return_tensors="pt").input_ids[0]
chunks = [ids[i:i+512] for i in range(0, len(ids) - 512, 512)]
chunks = chunks[:int(len(chunks) * 0.72)]
grab = {}
h = model.transformer.h[L].mlp.c_proj.register_forward_hook(
    lambda m, i, o: grab.__setitem__("k", i[0].detach()[0].double().numpy()))
C = np.zeros((D, D)); n = 0
for ci, ch in enumerate(chunks):
    model(ch.unsqueeze(0)); K = grab["k"]; C += K.T @ K; n += ch.shape[0]
    if ci % 40 == 0: print(f"  {ci}/{len(chunks)}", flush=True)
h.remove(); C /= n
w = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
tot = w.sum()
json.dump({"layer": L, "dim": int(D), "tokens": int(n),
           "eig": [float(x) for x in w],
           "eff_rank": float(tot**2 / (w**2).sum()),
           "cum": [float(x) for x in np.cumsum(w) / tot]},
          open("spectrum6.json", "w"))
print("eff rank", tot**2/(w**2).sum(), "| top", w[0]/tot, "| wrote spectrum6.json", flush=True)
