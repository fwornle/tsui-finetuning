"""Merge the GPT-2 small, GPT-2 XL and GPT-J measurements into the page's MEASURED blob."""
import json
small = json.load(open('measure.json'))
big   = json.load(open('measure_gpt2-xl.json'))
gptj  = json.load(open('measure_gptj.json'))
sp6   = json.load(open('spectrum6.json'))

def rows(rep, layers, lamA, lamB):
    out = []
    for L in layers:
        r = rep[str(L)]
        out.append({"layer": int(L),
                    "Amax": round(r["A_exact"][lamA]["gain_max"], 2),
                    "Arms": round(r["A_exact"][lamA]["gain_rms"], 2),
                    "Bmax": round(r["B_memit"][lamB]["gain_max"], 2),
                    "Brms": round(r["B_memit"][lamB]["gain_rms"], 2),
                    "effRank": round(r["eig"]["eff_rank"], 1),
                    "top": round(100 * r["eig"]["top_share"], 1)})
    return out

import numpy as np
w = np.array(sp6["eig"]); tot = w.sum(); cum = np.cumsum(w) / tot
idx = np.unique(np.round(np.logspace(0, np.log10(len(w)), 150)).astype(int) - 1)
idx = idx[(idx >= 0) & (idx < len(w))]
small_spec = {"idx": [int(i) for i in idx],
              "lam": [float(f"{w[i]:.6g}") for i in idx],
              "cum": [round(float(cum[i]), 5) for i in idx]}

bl = big["meta"]["layers"]
blob = {
 "small": {"name": "GPT-2 small", "params": "124M", "proj": "mlp.c_proj", "dim": 3072, "nlayer": 12,
           "effRank": round(small["6"]["eig"]["eff_rank"], 1),
           "top": round(100 * small["6"]["eig"]["top_share"], 1),
           "specLayer": 6, "spec": small_spec,
           "tokens": small["meta"]["tokens_fit"],
           "perDim": round(small["meta"]["tokens_fit"] / 3072, 1),
           "rows": rows(small, [3, 6, 9], "0.001", "0.1")},
 "xl":    {"name": "GPT-2 XL", "params": "1.5B", "proj": "mlp.c_proj", "dim": big["meta"]["d_in"], "nlayer": big["meta"]["n_layer"],
           "effRank": round(big[str(bl[1])]["eig"]["eff_rank"], 1),
           "top": round(100 * big[str(bl[1])]["eig"]["top_share"], 1),
           "specLayer": bl[1], "spec": big[str(bl[1])]["spectrum"],
           "tokens": big["meta"]["tokens_fit"],
           "perDim": big["meta"]["samples_per_dim"],
           "rows": rows(big, bl, "0.001", "0.1")},
 "gptj":  {"name": "GPT-J", "params": "6B", "proj": "mlp.fc_out", "dim": gptj["meta"]["d_in"], "nlayer": gptj["meta"]["n_layer"],
           "effRank": round(gptj[str(gptj["meta"]["layers"][1])]["eig"]["eff_rank"], 1),
           "top": round(100 * gptj[str(gptj["meta"]["layers"][1])]["eig"]["top_share"], 1),
           "specLayer": gptj["meta"]["layers"][1],
           "spec": gptj[str(gptj["meta"]["layers"][1])]["spectrum"],
           "tokens": gptj["meta"]["tokens_fit"],
           "perDim": gptj["meta"]["samples_per_dim"],
           "rows": rows(gptj, gptj["meta"]["layers"], "0.001", "0.1")}
}
open("measured.js", "w").write("const MEASURED = " + json.dumps(blob, separators=(',', ':')) + ";\n")
print("small:", blob["small"]["effRank"], "of", blob["small"]["dim"], "| rows", blob["small"]["rows"])
print("xl   :", blob["xl"]["effRank"], "of", blob["xl"]["dim"], "| rows", blob["xl"]["rows"])
print("gptj :", blob["gptj"]["effRank"], "of", blob["gptj"]["dim"], "| rows", blob["gptj"]["rows"])
print("bytes", len(open("measured.js").read()))
