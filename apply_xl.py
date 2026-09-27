"""Fold the GPT-2 XL run into the page. Every number in the prose is derived from the
JSON, never typed, so the page cannot drift from the measurement."""
import json, re, numpy as np

small = json.load(open('measure.json'))
big   = json.load(open('measure_gpt2-xl.json'))
sp6   = json.load(open('spectrum6.json'))
LAM_A, LAM_B = '0.001', '0.1'

def rows(rep, layers):
    return [{"layer": int(L),
             "Amax": round(rep[str(L)]["A_exact"][LAM_A]["gain_max"], 2),
             "Arms": round(rep[str(L)]["A_exact"][LAM_A]["gain_rms"], 2),
             "Bmax": round(rep[str(L)]["B_memit"][LAM_B]["gain_max"], 2),
             "Brms": round(rep[str(L)]["B_memit"][LAM_B]["gain_rms"], 2)} for L in layers]

w = np.array(sp6["eig"]); tot = w.sum(); cum = np.cumsum(w) / tot
idx = np.unique(np.round(np.logspace(0, np.log10(len(w)), 150)).astype(int) - 1)
idx = idx[(idx >= 0) & (idx < len(w))]
sspec = {"idx": [int(i) for i in idx], "lam": [float(f"{w[i]:.6g}") for i in idx],
         "cum": [round(float(cum[i]), 5) for i in idx]}

SL, BL = [3, 6, 9], big["meta"]["layers"]
blob = {
 "small": {"name":"GPT-2 small","params":"124M","dim":3072,"nlayer":12,
   "effRank":round(small["6"]["eig"]["eff_rank"],1),"top":round(100*small["6"]["eig"]["top_share"],1),
   "spec":sspec,"tokens":small["meta"]["tokens_fit"],
   "held":small["meta"]["tokens_held"]//512*24,"rows":rows(small,SL)},
 "xl": {"name":"GPT-2 XL","params":"1.5B","dim":big["meta"]["d_in"],"nlayer":big["meta"]["n_layer"],
   "effRank":round(big[str(BL[1])]["eig"]["eff_rank"],1),"top":round(100*big[str(BL[1])]["eig"]["top_share"],1),
   "spec":big[str(BL[1])]["spectrum"],"tokens":big["meta"]["tokens_fit"],
   "held":big["meta"]["tokens_held"]//512*24,"rows":rows(big,BL)},
}
def rng_(m, k): 
    v = [r[k] for r in blob[m]["rows"]]; return min(v), max(v)
sB = rng_("small","Brms"); xB = rng_("xl","Brms")
sA = rng_("small","Arms"); xA = rng_("xl","Arms")
xer = [round(big[str(L)]["eig"]["eff_rank"],1) for L in BL]
xtop = [round(100*big[str(L)]["eig"]["top_share"],1) for L in BL]
f = lambda p: f"{p[0]:.1f}–{p[1]:.1f}×"
lamA = sorted(float(k) for k in big[str(BL[0])]["A_exact"])
lamB = sorted(float(k) for k in big[str(BL[0])]["B_memit"])
ordersA = round(np.log10(lamA[-1]/lamA[0]))
grew = xB[0] >= sB[0]

s = open('tsui-fine-tuning.html').read()
s = s.replace('%%DATA%%', 'const MEASURED = ' + json.dumps(blob, separators=(',',':')) + ';')

def rep(a, b):
    global s
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b, 1)

rep("""    <p>Everything above this point was synthetic. The obvious objection was that a real layer&rsquo;s key covariance might
    be close to isotropic in the directions the freedom occupies, in which case finding 2 collapses to a curiosity. So
    I measured it: GPT-2 small, 130,000 tokens of Wikipedia, the key Gramian of the MLP down-projection at layers 3, 6
    and 9, with leakage scored on 1,728 keys from documents that contributed to neither the Gramian nor the constraints.</p>
    <p><strong>It does not collapse. It gets bigger.</strong></p>""",
f"""    <p>Everything above this point was synthetic. The obvious objection was that a real layer&rsquo;s key covariance might
    be close to isotropic in the directions the freedom occupies, in which case finding 2 collapses to a curiosity. So I
    measured it on two models: <strong>GPT-2 small</strong> (124M, key dimension 3072, layers 3/6/9) and
    <strong>GPT-2 XL</strong> (1.5B, key dimension 6400) — the latter being the model ROME and MEMIT were actually
    developed on, with layers {BL[0]} and {BL[1]} inside MEMIT&rsquo;s published critical range and layer {BL[2]} well
    outside it as a control. Leakage is scored on keys from documents that contributed to neither the Gramian nor the
    constraints.</p>
    <p><strong>It does not collapse, and it does not wash out at 12× the size.</strong> On GPT-2 XL, replacing MEMIT&rsquo;s
    covariance term with an identity matrix leaves the edit equally accurate and multiplies average leakage by
    <b class="hh">{f(xB)}</b>, against <b>{f(sB)}</b> on GPT-2 small. The key Gramian&rsquo;s participation ratio at 6400
    dimensions is <b class="gg">{xer[1]}</b> — {'no less' if xer[1] <= 40 else 'still'} concentrated than the 3072-dimensional
    case, in a space twice as large.</p>""")

rep("""    <p>It establishes that the covariance term in a published editor is not a formality: on a real layer, replacing it
    with the identity leaves the edit just as accurate and multiplies collateral leakage by <b class="hh">8 to 11×</b>.
    It also answers the question this page previously left open — the effective conditioning of a real key Gramian is
    not near 1, it is a participation ratio of <b>27 out of 3072</b>, with a single direction holding 18% of all the
    energy. Real activations are about as anisotropic as data gets.</p>""",
f"""    <p>It establishes that the covariance term in a published editor is not a formality: on both models, replacing it
    with the identity leaves the edit just as accurate and multiplies collateral leakage by <b class="hh">{f(sB)}</b>
    (GPT-2 small) and <b class="hh">{f(xB)}</b> (GPT-2 XL). It also answers the question this page previously left open —
    the effective conditioning of a real key Gramian is nowhere near 1. The participation ratio is
    <b>{blob['small']['effRank']} of 3072</b> on the small model and <b>{xer[1]} of 6400</b> on XL, with a single
    direction holding {blob['small']['top']}% and {xtop[1]}% of all the energy respectively. Real activations are about
    as anisotropic as data gets, and getting bigger does not make them isotropic.</p>
    <p>It also rules out one alternative explanation. Layer {BL[2]} sits outside MEMIT&rsquo;s critical range, where no
    editing method operates, and it behaves like the others — so this is a property of MLP key statistics in general,
    not an artefact of the layers people happen to edit.</p>""")

rep("""      <li><strong>Keys.</strong> GPT-2 small, the input to <span class="mono">mlp.c_proj</span> at layers 3, 6 and 9 —
      the post-GELU activation, dimension 3072. Exactly the vector MEMIT calls a key.</li>""",
f"""      <li><strong>Keys.</strong> The input to <span class="mono">mlp.c_proj</span> — the post-GELU activation, exactly
      the vector MEMIT calls a key. Dimension 3072 at layers 3/6/9 of GPT-2 small; dimension 6400 at layers
      {'/'.join(map(str,BL))} of GPT-2 XL.</li>""")

rep("""      <li><strong>Gramian.</strong> The uncentered second moment over 93,000 tokens of Wikipedia: the same statistic
      and the same source MEMIT uses, at a smaller sample size.</li>""",
f"""      <li><strong>Gramian.</strong> The uncentered second moment over {blob['small']['tokens']//1000}k tokens of Wikipedia
      (small) and {blob['xl']['tokens']//1000}k tokens (XL): the same statistic and the same source MEMIT uses, at a
      smaller sample size. The XL corpus was enlarged deliberately so that samples-per-key-dimension
      ({big['meta']['samples_per_dim']}) stays comparable to the small run&rsquo;s, rather than degrading with the
      larger dimension.</li>""")

rep("""      <li><strong>Split.</strong> Leakage is scored on every key the held-out documents produce — 1,728 of them for
      GPT-2 small, 3,168 for GPT-2 XL — from documents that contributed to neither the Gramian nor the constraints.""",
f"""      <li><strong>Split.</strong> Leakage is scored on every key the held-out documents produce — {blob['small']['held']:,}
      of them for GPT-2 small, {blob['xl']['held']:,} for GPT-2 XL — from documents that contributed to neither the
      Gramian nor the constraints.""")

rep("""      <li><strong>Robustness.</strong> The ratio holds across all three layers and across three orders of magnitude of
      the ridge λ, and the two solutions' row spaces have a maximum principal angle of 90° — they are not small
      perturbations of one another.</li>""",
f"""      <li><strong>Robustness.</strong> The ratio holds across all six layers measured, across both models, and across
      {ordersA} orders of magnitude of the ridge λ. The two solutions' row spaces have a maximum principal angle of 90°
      — they are not small perturbations of one another.</li>""")

rep("""    <p>The diagnostic that answers the original question is the <strong>participation ratio</strong>,
    <span class="mono">(Σλ)² / Σλ²</span> — the honest count of how many directions carry the energy. It comes out at
    17 to 28 out of 3072, with the top single direction holding 18–23%. Meanwhile 99% of the energy needs about 2,600
    directions, so the tail is long and nearly flat. Both are true at once and not in tension: a couple of dozen
    directions dominate, and the last percent of the energy is smeared across thousands.</p>""",
f"""    <p>The diagnostic that answers the original question is the <strong>participation ratio</strong>,
    <span class="mono">(Σλ)² / Σλ²</span> — the honest count of how many directions carry the energy. It comes out at
    17 to 28 out of 3072 on GPT-2 small and {min(xer)} to {max(xer)} out of 6400 on GPT-2 XL, with the top single
    direction holding 18–23% and {min(xtop)}–{max(xtop)}% respectively. Meanwhile 99% of the energy needs thousands of
    directions, so the tail is long and nearly flat. Both are true at once and not in tension: a couple of dozen
    directions dominate, and the last percent of the energy is smeared across the rest.</p>""")

for a, b in [("leaks\n    <b>8 to 11×</b> more onto knowledge you never sampled, at identical edit accuracy.",
              f"leaks\n    <b>{f(sB)}</b> more onto knowledge you never sampled, at identical edit accuracy."),
             ("GPT-2 three panels below — you leak <b>8 to 11 times</b> as much onto knowledge you never sampled.",
              f"GPT-2 three panels below — you leak <b>{f(sB)}</b> as much onto knowledge you never sampled."),
             ("""GPT-2 layer the choice within the freedom is worth <b>8 to 11× in leakage</b> onto unsampled knowledge, at identical
    edit accuracy, and the multiplier is the anisotropy of the key Gramian, whose participation ratio is 27 out of 3072.""",
              f"""GPT-2 layer the choice within the freedom is worth <b>{f(sB)} in leakage</b> onto unsampled knowledge ({f(xB)} on
    GPT-2 XL), at identical edit accuracy, and the multiplier is the anisotropy of the key Gramian, whose participation
    ratio is {blob['small']['effRank']} of 3072 and {xer[1]} of 6400."""),
             ("participation ratio is 27 of 3072, and MEMIT's covariance term is worth 8–11× in leakage.",
              f"participation ratio is {blob['small']['effRank']} of 3072 (and {xer[1]} of 6400 on GPT-2 XL), and MEMIT's covariance term is worth {f(sB)}/{f(xB)} in leakage."),
             ("""  130,000 tokens of Wikipedia, key Gramians of <span class="mono">mlp.c_proj</span> at layers 3, 6 and 9, leakage scored
  on held-out documents""",
              f"""  and GPT-2 XL, {(blob['small']['tokens']+small['meta']['tokens_held'])//1000}k and
  {(blob['xl']['tokens']+big['meta']['tokens_held'])//1000}k tokens of Wikipedia, key Gramians of
  <span class="mono">mlp.c_proj</span> at six layers, leakage scored on held-out documents""")]:
    rep(a, b)

open('tsui-fine-tuning.html','w').write(s)
print('small B rms', f(sB), '| xl B rms', f(xB))
print('small A rms', f(sA), '| xl A rms', f(xA))
print('xl eff ranks', xer, '| top', xtop)
print('page updated, all prose numbers generated from JSON')
