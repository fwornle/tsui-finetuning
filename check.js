
const $ = s => document.querySelector(s);
const RM = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const themeHooks = [];
function runThemeHooks(){
  themeHooks.forEach(f => {
    try { f(); }
    catch (err){ console.warn('theme hook failed to redraw', err); }
  });
}
const svg = (w, h, inner, pad) => '<svg viewBox="' + (pad ? -pad : 0) + ' ' + (pad ? -pad : 0) + ' ' +
  (w + (pad ? pad * 2 : 0)) + ' ' + (h + (pad ? pad * 2 : 0)) + '" role="img">' + inner + '</svg>';
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const gaussFrom = r => () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
/* solve A X = B by Gaussian elimination with partial pivoting; A n×n, B n×m */
function gsolve(Ain, Bin){
  const n = Ain.length, m = Bin[0].length;
  const A = Ain.map(r => Float64Array.from(r)), B = Bin.map(r => Float64Array.from(r));
  for (let c = 0; c < n; c++){
    let piv = c; for (let i = c + 1; i < n; i++) if (Math.abs(A[i][c]) > Math.abs(A[piv][c])) piv = i;
    const ta = A[c]; A[c] = A[piv]; A[piv] = ta;
    const tb = B[c]; B[c] = B[piv]; B[piv] = tb;
    const pv = A[c][c] || 1e-300;
    for (let i = c + 1; i < n; i++){ const f = A[i][c] / pv; if (!f) continue;
      for (let j = c; j < n; j++) A[i][j] -= f * A[c][j];
      for (let j = 0; j < m; j++) B[i][j] -= f * B[c][j]; }
  }
  const X = Array.from({length:n}, () => new Float64Array(m));
  for (let i = n - 1; i >= 0; i--) for (let j = 0; j < m; j++){
    let v = B[i][j]; for (let k = i + 1; k < n; k++) v -= A[i][k] * X[k][j];
    X[i][j] = v / (A[i][i] || 1e-300); }
  return X;
}
/* orthonormalise a list of vectors (modified Gram-Schmidt), dropping degenerates */
function ortho(vs){
  const out = [];
  for (let v of vs){ v = Float64Array.from(v);
    for (const u of out){ const d = dotv(v, u); for (let i = 0; i < v.length; i++) v[i] -= d * u[i]; }
    const n = Math.sqrt(dotv(v, v)); if (n < 1e-9) continue;
    for (let i = 0; i < v.length; i++) v[i] /= n; out.push(v); }
  return out;
}

/* ══ navigation · a published artifact runs sandboxed, so the TOC uses buttons not hrefs ══ */
document.addEventListener('click', e => {
  const b = e.target.closest('.toc button[data-target]'); if (!b) return;
  const t = document.getElementById(b.dataset.target); if (!t) return;
  t.scrollIntoView({behavior: RM ? 'auto' : 'smooth', block: 'start'});
  t.setAttribute('tabindex', '-1'); t.focus({preventScroll: true});
  t.addEventListener('blur', () => t.removeAttribute('tabindex'), {once: true});
});

/* ══ reveal ══ */
if ('IntersectionObserver' in window && !RM){
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting){
    e.target.classList.add('in'); io.unobserve(e.target); } }), {threshold:.05});
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));
} else document.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));

/* ══ theme + detail ══ */
(function(){
  const btn = $('#themeBtn'), root = document.documentElement;
  const hostStamp = root.getAttribute('data-theme');
  const mqDark = window.matchMedia('(prefers-color-scheme: dark)');
  const MODES = [['system','◐ system'], ['light','☀ light'], ['dark','☾ dark']];
  let i = 0;
  function apply(){
    const mode = MODES[i][0];
    if (mode === 'system'){
      if (hostStamp) root.setAttribute('data-theme', hostStamp);
      else root.removeAttribute('data-theme');
    } else root.setAttribute('data-theme', mode);
    btn.textContent = MODES[i][1];
    btn.setAttribute('aria-label', 'colour theme: ' + mode);
    runThemeHooks();
  }
  btn.addEventListener('click', () => { i = (i + 1) % MODES.length; apply(); });
  mqDark.addEventListener('change', () => { if (MODES[i][0] === 'system') apply(); });
  apply();
})();

(function(){
  const btns = [$('#d1'), $('#d2'), $('#d3')], hint = $('#hintBtn');
  let depth = 1;
  function apply(scrollTo){
    document.body.classList.remove('depth-1', 'depth-2', 'depth-3');
    document.body.classList.add('depth-' + depth);
    btns.forEach((b, i) => { const on = i + 1 === depth;
      b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
    document.querySelectorAll('.lvl2, .detail').forEach(el => el.classList.add('in'));
    if (scrollTo){ const f = document.querySelector(scrollTo);
      if (f) f.scrollIntoView({behavior: RM ? 'auto' : 'smooth', block: 'start'}); }
    runThemeHooks();
  }
  btns.forEach((b, i) => b.addEventListener('click', () => { depth = i + 1; apply(null); }));
  hint.addEventListener('click', () => { depth = 2; apply('#primer'); });
  apply(null);
})();

/* ══════════ hero · the keys arrive and trace an ellipse ══════════ */
(function(){
  const cv = $('#heroCanvas'); if (!cv) return;
  const ctx = cv.getContext('2d'), out = $('#heroReadout');
  const W = 1024, H = 230;
  const rnd = mulberry(4211), gs = gaussFrom(rnd);
  const TH = 0.42, ct = Math.cos(TH), st = Math.sin(TH);
  const pts = [];
  for (let i = 0; i < 420; i++){ const a = gs() * 1.0, b = gs() * 0.17;
    pts.push([a * ct - b * st, a * st + b * ct]); }
  let shown = 260, timer = null;
  function readCols(){ const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
    return {pro:g('--pro')||'#5ea2f7', ok:g('--ok')||'#4ade80', obs:g('--obs')||'#f5a524',
            dim:g('--dim')||'#8b9ab8', faint:g('--faint')||'#55648a', ink:g('--ink')||'#e3eaf8',
            line:g('--line')||'#1a2438', grid:g('--svgGrid')||'#131c2c'}; }
  let col = readCols();
  function resize(){ const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr,0,0,dpr,0,0); draw(); }
  function cov(n){ let sxx=0,sxy=0,syy=0; for (let i=0;i<n;i++){ const p=pts[i];
      sxx+=p[0]*p[0]; sxy+=p[0]*p[1]; syy+=p[1]*p[1]; }
    const m = Math.max(1,n); return [sxx/m, sxy/m, syy/m]; }
  function draw(){
    ctx.clearRect(0,0,W,H);
    const cx = 268, cy = H/2, S = 74;
    const c2 = cov(shown), a = c2[0], b = c2[1], d = c2[2];
    const tr = a + d, det = a*d - b*b, disc = Math.sqrt(Math.max(0, tr*tr/4 - det));
    const l1 = tr/2 + disc, l2 = Math.max(1e-9, tr/2 - disc);
    const ang = 0.5 * Math.atan2(2*b, a - d);
    ctx.strokeStyle = col.grid; ctx.lineWidth = 1;
    for (let g = -2; g <= 2; g++){ ctx.beginPath();
      ctx.moveTo(cx + g*S, 18); ctx.lineTo(cx + g*S, H-18);
      ctx.moveTo(cx - 2.6*S, cy + g*S*0.72); ctx.lineTo(cx + 2.6*S, cy + g*S*0.72); ctx.stroke(); }
    ctx.fillStyle = col.pro;
    for (let i = 0; i < shown; i++){ const p = pts[i];
      ctx.globalAlpha = 0.16 + 0.5 * (i / Math.max(1,shown));
      ctx.beginPath(); ctx.arc(cx + p[0]*S, cy + p[1]*S, 1.9, 0, 6.2832); ctx.fill(); }
    ctx.globalAlpha = 1;
    const A1 = 2*Math.sqrt(l1)*S, A2 = 2*Math.sqrt(l2)*S;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang);
    ctx.strokeStyle = col.ink; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.ellipse(0, 0, A1, A2, 0, 0, 6.2832); ctx.stroke();
    ctx.strokeStyle = col.pro; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(-A1,0); ctx.lineTo(A1,0); ctx.stroke();
    ctx.strokeStyle = col.ok; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(0,-A2); ctx.lineTo(0,A2); ctx.stroke();
    ctx.restore();
    ctx.font = '600 11px ui-monospace, monospace';
    ctx.fillStyle = col.pro;
    ctx.fillText('heavily used', cx + Math.cos(ang)*A1 + 8, cy + Math.sin(ang)*A1 + 4);
    ctx.fillStyle = col.ok;
    ctx.fillText('barely used', cx - Math.sin(ang)*A2 - 74, cy + Math.cos(ang)*A2 + 20);
    ctx.fillStyle = col.faint; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(shown + ' keys observed', 26, 26);
    const bx = 618, bw = 340;
    const lg = v => Math.max(0.03, 1 + Math.log10(v + 1e-12) / 4);
    ctx.fillStyle = col.faint;
    ctx.fillText('ENERGY PER DIRECTION   ·   log scale', bx, 42);
    [['λ₁  major axis', l1, col.pro], ['λ₂  minor axis', l2, col.ok]].forEach((r, i) => {
      const y = 74 + i*52;
      ctx.fillStyle = col.dim; ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(r[0], bx, y - 6);
      ctx.fillStyle = col.line; ctx.fillRect(bx, y, bw, 12);
      ctx.fillStyle = r[2]; ctx.fillRect(bx, y, Math.max(3, bw * lg(r[1])), 12);
      ctx.fillStyle = col.ink; ctx.font = '600 11px ui-monospace, monospace';
      ctx.fillText(r[1].toExponential(2), bx, y + 27);
    });
    ctx.fillStyle = col.obs; ctx.font = '600 12px ui-monospace, monospace';
    ctx.fillText('ratio  λ₁ / λ₂  =  ' + (l1/l2).toFixed(0) + '×', bx, 196);
    out.innerHTML = 'The ellipse is a <b>Gramian</b> — literally the matrix written ' +
      '<span class="mono pp">K₀K₀ᵀ</span> in the model-editing papers. Its long axis is what the old knowledge leans ' +
      'on hardest; its short axis carries <b>' + (l1/l2).toFixed(0) + '× less</b> energy, and is where every ' +
      'null-space method on the shelf puts the adapter. Everything below is an argument about how to choose that ' +
      'short axis, and what to do with the room it leaves.';
  }
  $('#heroPlay').addEventListener('click', () => {
    if (RM){ shown = pts.length; draw(); return; }
    if (timer){ clearInterval(timer); timer = null; $('#heroPlay').textContent = '▶ stream keys'; return; }
    shown = 12; $('#heroPlay').textContent = '■ pause';
    timer = setInterval(() => { shown = Math.min(pts.length, shown + 7); draw();
      if (shown >= pts.length){ clearInterval(timer); timer = null; $('#heroPlay').textContent = '▶ stream keys'; } }, 34);
  });
  themeHooks.push(() => { col = readCols(); draw(); });
  window.addEventListener('resize', resize);
  resize();
})();

/* ══════════ 00a · matrix–vector product as a weighted sum of columns ══════════ */
(function(){
  const host = $('#mvSvg'); if (!host) return;
  const R = 5, Cn = 7, rnd = mulberry(77);
  const M = Array.from({length:R}, () => Array.from({length:Cn}, () => Math.round((rnd()*2-1)*100)/100));
  const k = Array.from({length:Cn}, () => Math.round((rnd()*1.6-0.4)*100)/100);
  let sel = 3;
  const cw = 31, ch = 27, x0 = 46, y0 = 26, gx = 34, gy = 30;
  function cell(x, y, v, hot, w){
    const mag = Math.min(1, Math.abs(v));
    const c = v >= 0 ? 'var(--pro)' : 'var(--obs)';
    return '<rect x="' + x + '" y="' + y + '" width="' + (w||cw) + '" height="' + ch + '" rx="3" fill="' + c +
      '" fill-opacity="' + (0.10 + 0.62*mag).toFixed(3) + '" stroke="' + (hot ? 'var(--new)' : 'var(--line2)') +
      '" stroke-width="' + (hot ? 1.8 : 0.8) + '"/>' +
      '<text x="' + (x + (w||cw)/2) + '" y="' + (y + ch/2 + 3.5) + '" text-anchor="middle" font-size="8.5" fill="var(--ink)">' +
      v.toFixed(2) + '</text>';
  }
  function draw(){
    const v = M.map(row => row.reduce((s, x, j) => s + x * k[j], 0));
    let g = '<text x="4" y="' + (y0 + 14) + '" font-size="12" font-weight="700" fill="var(--ink)">W</text>' +
            '<text x="4" y="' + (y0 + 2.6*gy + 14) + '" font-size="8.5" fill="var(--faint)">5×7</text>' +
            '<text x="4" y="204" font-size="12" font-weight="700" fill="var(--ink)">k</text>' +
            '<text x="298" y="18" font-size="11" font-weight="700" fill="var(--ink)">v = W k</text>';
    for (let i = 0; i < R; i++) for (let j = 0; j < Cn; j++)
      g += cell(x0 + j*gx, y0 + i*gy, M[i][j], j === sel);
    for (let j = 0; j < Cn; j++) g += cell(x0 + j*gx, 186, k[j], j === sel);
    const mx = Math.max.apply(null, v.map(Math.abs)) || 1;
    for (let i = 0; i < R; i++){
      const contrib = M[i][sel] * k[sel], yy = y0 + i*gy;
      g += '<rect x="298" y="' + yy + '" width="58" height="' + ch + '" rx="3" fill="var(--svgBox)" stroke="var(--line2)" stroke-width="0.8"/>';
      g += '<rect x="300" y="' + (yy+3) + '" width="' + Math.max(1, 54*Math.abs(v[i])/mx).toFixed(1) + '" height="' + (ch-6) +
        '" rx="2" fill="' + (v[i] >= 0 ? 'var(--pro)' : 'var(--obs)') + '" fill-opacity=".34"/>';
      g += '<rect x="300" y="' + (yy+3) + '" width="' + Math.max(1, 54*Math.abs(contrib)/mx).toFixed(1) + '" height="' + (ch-6) +
        '" rx="2" fill="var(--new)" fill-opacity=".85"/>';
      g += '<text x="362" y="' + (yy + ch/2 + 3.5) + '" font-size="8.5" fill="var(--dim)">' + v[i].toFixed(2) + '</text>';
    }
    g += '<line x1="' + (x0 + sel*gx + cw/2) + '" y1="' + (y0 + 5*gy - 2) + '" x2="' + (x0 + sel*gx + cw/2) +
         '" y2="184" stroke="var(--new)" stroke-width="1.2" stroke-dasharray="3 3"/>';
    for (let j = 0; j < Cn; j++)
      g += '<rect class="mvHit" data-j="' + j + '" x="' + (x0 + j*gx) + '" y="' + (y0-4) + '" width="' + cw +
           '" height="' + (190 - y0 + ch) + '" fill="transparent" style="cursor:pointer"/>';
    host.innerHTML = svg(390, 220, g, 6);
    host.querySelectorAll('.mvHit').forEach(r => r.addEventListener('click', () => { sel = +r.dataset.j; draw(); }));
    $('#mvNote').innerHTML = 'Coordinate <b>' + (sel+1) + '</b> of the key is <span class="mono">' + k[sel].toFixed(2) +
      '</span>. It scales <b>column ' + (sel+1) + '</b> of W, and that product — the <span class="nn">teal</span> slice ' +
      'of each output bar — is one of seven contributions that sum to <span class="mono">v</span>. ' +
      'That is all a matrix does: pick columns, scale them, add them up.';
  }
  themeHooks.push(draw); draw();
})();

/* ══════════ 00b · the perpendicularity dial ══════════ */
(function(){
  const host = $('#perpSvg'); if (!host) return;
  const sl = $('#perpAng');
  function draw(){
    const deg = +sl.value, a = deg * Math.PI / 180;
    const cx = 132, cy = 116, Rr = 84;
    const dx = cx + Rr*Math.cos(a), dy = cy - Rr*Math.sin(a);
    const cosA = Math.cos(a), sinA = Math.abs(Math.sin(a)), projLen = Rr * cosA;
    let g = '<defs>' +
      '<marker id="ahP" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto">' +
      '<path d="M0,0 L7,3 L0,6 z" fill="var(--pro)"/></marker>' +
      '<marker id="ahN" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto">' +
      '<path d="M0,0 L7,3 L0,6 z" fill="var(--new)"/></marker></defs>';
    g += '<circle cx="' + cx + '" cy="' + cy + '" r="' + Rr + '" fill="none" stroke="var(--svgGrid)" stroke-width="1"/>';
    g += '<line x1="' + cx + '" y1="' + (cy - Rr - 14) + '" x2="' + cx + '" y2="' + (cy + 14) +
         '" stroke="var(--ok)" stroke-width="1.1" stroke-dasharray="4 4"/>';
    g += '<text x="' + (cx + 6) + '" y="' + (cy - Rr - 4) + '" font-size="9.5" fill="var(--ok)">90° · safe</text>';
    if (Math.abs(projLen) > 1.5){
      g += '<line x1="' + cx + '" y1="' + cy + '" x2="' + (cx + projLen) + '" y2="' + cy +
           '" stroke="var(--hot)" stroke-width="6" stroke-linecap="round" opacity=".8"/>';
      g += '<line x1="' + (cx + projLen) + '" y1="' + cy + '" x2="' + dx + '" y2="' + dy +
           '" stroke="var(--ok)" stroke-width="1.4" stroke-dasharray="3 3"/>';
    }
    g += '<line x1="' + cx + '" y1="' + cy + '" x2="' + (cx + Rr) + '" y2="' + cy +
         '" stroke="var(--pro)" stroke-width="2.4" marker-end="url(#ahP)"/>';
    g += '<line x1="' + cx + '" y1="' + cy + '" x2="' + dx + '" y2="' + dy +
         '" stroke="var(--new)" stroke-width="2.4" marker-end="url(#ahN)"/>';
    g += '<text x="' + (cx + Rr + 7) + '" y="' + (cy + 4) + '" font-size="10" font-weight="700" fill="var(--pro)">k</text>';
    g += '<text x="' + (dx + (dx > cx ? 7 : -17)) + '" y="' + (dy - 7) + '" font-size="10" font-weight="700" fill="var(--new)">Δ</text>';
    const bx = 248, bw = 126;
    [['damage  |Δ·k|', Math.abs(cosA), 'var(--hot)'], ['room to learn', sinA, 'var(--ok)']].forEach((r, i) => {
      const y = 76 + i*54;
      g += '<text x="' + bx + '" y="' + (y - 7) + '" font-size="9.5" fill="var(--faint)">' + r[0] + '</text>';
      g += '<rect x="' + bx + '" y="' + y + '" width="' + bw + '" height="11" rx="3" fill="var(--line)"/>';
      g += '<rect x="' + bx + '" y="' + y + '" width="' + Math.max(1, bw*r[1]).toFixed(1) + '" height="11" rx="3" fill="' + r[2] + '"/>';
      g += '<text x="' + bx + '" y="' + (y + 27) + '" font-size="12" font-weight="700" fill="var(--ink)">' + r[1].toFixed(3) + '</text>';
    });
    host.innerHTML = svg(390, 220, g, 6);
    $('#perpAngV').textContent = deg + '°';
    $('#perpNote').innerHTML = Math.abs(cosA) < 0.02
      ? '<b class="gg">Exactly perpendicular.</b> The update is invisible to this key — the model answers it exactly as before, however large Δ grows. All the learning happens in the remaining directions, and in 4096 dimensions there are a great many of those.'
      : 'At ' + deg + '° the update still overlaps the protected key by <b class="hh">' + Math.abs(cosA).toFixed(3) +
        '</b>. That overlap <em>is</em> the forgetting, not a side effect of it. Every method below is a different way of driving it to zero.';
  }
  sl.addEventListener('input', draw); themeHooks.push(draw); draw();
})();

/* ══════════ 01 · the staircase forms, then gets walked down ══════════ */
(function(){
  const host = $('#hessSvg'); if (!host) return;
  const N = 8, MOUT = 3;                 /* 8 states, 3 measured outputs */
  const CLEAR = N - 2;                   /* columns that carry entries below the sub-diagonal */
  const TOTAL = CLEAR + N;               /* 6 reflections, then 8 back-substituted rows */
  let step = CLEAR + N, timer = null;   /* open on the finished state: the readout is the point */
  const cw = 30, chh = 24, gx = 32, gy = 26, x0 = 74, y0 = 34;
  const rnd = mulberry(1987);
  const vals = Array.from({length:N}, () => Array.from({length:N}, () => rnd()));
  function zeroedCols(s){ return Math.min(CLEAR, s); }
  function rowsDone(s){ return Math.max(0, s - CLEAR); }
  function draw(){
    const zc = zeroedCols(step), rd = rowsDone(step);
    let zeros = 0; for (let j = 0; j < zc; j++) zeros += N - (j + 2);
    const nz = N*N - zeros;
    let g = '';
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++){
      const isLower = (i - j) >= 2;
      const killed = isLower && j < zc;
      const determined = (i >= N - rd) && !killed;
      const free = determined && j >= N - MOUT;
      let fill = 'var(--svgCell)', op = (0.18 + 0.5*vals[i][j]).toFixed(3), stroke = 'var(--line)';
      if (killed){ fill = 'var(--svgOff)'; op = '1'; stroke = 'var(--line)'; }
      else if (free){ fill = 'var(--free)'; op = '.85'; stroke = 'var(--free)'; }
      else if (determined){ fill = 'var(--ok)'; op = '.7'; stroke = 'var(--ok)'; }
      else if (isLower && j === zc){ fill = 'var(--hot)'; op = '.8'; stroke = 'var(--hot)'; }
      g += '<rect x="' + (x0 + j*gx) + '" y="' + (y0 + i*gy) + '" width="' + cw + '" height="' + chh +
        '" rx="2.5" fill="' + fill + '" fill-opacity="' + op + '" stroke="' + stroke + '" stroke-width="0.9"/>';
      if (killed) g += '<text x="' + (x0 + j*gx + cw/2) + '" y="' + (y0 + i*gy + chh/2 + 3.5) +
        '" text-anchor="middle" font-size="9" fill="var(--faint)">0</text>';
    }
    /* the staircase edge */
    let path = '';
    for (let j = 0; j < N - 1; j++){
      const yTop = y0 + (j + 2) * gy;
      path += (j === 0 ? 'M' : 'L') + (x0 + j*gx) + ',' + yTop + 'L' + (x0 + (j+1)*gx) + ',' + yTop;
    }
    g += '<path d="' + path + '" fill="none" stroke="var(--ink)" stroke-width="1.5" opacity="' + (zc === CLEAR ? '.9' : '.25') + '"/>';
    g += '<text x="4" y="' + (y0 - 10) + '" font-size="10" fill="var(--faint)">the 8×8 block</text>';
    g += '<text x="4" y="' + (y0 + 14) + '" font-size="11" font-weight="700" fill="var(--ink)">A</text>';
    if (rd > 0) g += '<path d="M' + (x0 - 14) + ',' + (y0 + (N - rd)*gy + 6) + 'L' + (x0 - 14) + ',' + (y0 + N*gy - 4) +
      '" stroke="var(--ok)" stroke-width="2.5"/>';
    /* right column · running commentary */
    const bx = 372;
    const lines = zc < CLEAR
      ? ['PHASE 1 · ORTHOGONAL REFLECTIONS',
         'Reflection ' + (zc + 1) + ' of ' + CLEAR + ' clears column ' + (zc + 1),
         'below the sub-diagonal. The system is',
         'unchanged — only its coordinates are.',
         '',
         'Because reflections are isometries, every',
         'length and angle survives the move. That',
         'is why Tsui uses this form and not a',
         'canonical one: the measure of merit at',
         'the end is a norm.']
      : ['PHASE 2 · BACK-SUBSTITUTION',
         rd === 0 ? 'The staircase is complete. 21 entries are' : 'Row ' + (N - rd + 1) + ' is now determined by the',
         rd === 0 ? 'now exactly zero, and the constraint' : 'rows below it. Pick its ' + MOUT + ' free values',
         rd === 0 ? 'TA − FT = LC inherits the structure.' : '(purple) and the rest follow by arithmetic.',
         '',
         'Each row hands you an m-vector you may',
         'choose freely, m = ' + MOUT + ' being the number of',
         'measured outputs. Those choices are the',
         'design freedom. Chapters 8 and 9 are',
         'about how to spend them.'];
    lines.forEach((L, i) => {
      const bold = i === 0;
      g += '<text x="' + bx + '" y="' + (y0 + 4 + i*19) + '" font-size="' + (bold ? 10 : 11) + '"' +
        (bold ? ' font-weight="700" letter-spacing="1.4" fill="var(--new)"' : ' fill="var(--dim)"') + '>' + L + '</text>';
    });
    host.innerHTML = svg(690, 270, g, 6);
    $('#hsPhase').textContent = zc < CLEAR ? '1 / 2' : '2 / 2';
    $('#hsPhaseD').textContent = zc < CLEAR ? 'reflection ' + (zc + 1) + ' of ' + CLEAR : 'back-substitution';
    $('#hsNz').textContent = nz;
    $('#hsDet').textContent = rd;
    $('#hsFree').textContent = rd * MOUT;
    $('#hessNote').innerHTML = step === 0
      ? 'Press <b>run</b>. Phase 1 is six orthogonal reflections; phase 2 walks eight rows. Watch the last readout.'
      : step >= TOTAL
        ? 'Done. The hard constraint is satisfied for <b class="ff">' + (N*MOUT) + ' distinct solutions&rsquo; worth of free choice</b> — ' +
          'the purple cells. A textbook design picks one of them arbitrarily and reports a single answer. Tsui&rsquo;s ' +
          'objection is not that this is slow, but that a different choice with <em>identical</em> performance can be ' +
          'several times less sensitive to the model being wrong, and you never find out.'
        : (zc < CLEAR
            ? 'Zeroing column ' + (zc + 1) + '. Entries below the sub-diagonal go to exact zero, using reflections that preserve every norm.'
            : 'Row ' + (N - rd + 1) + ' determined. Green cells are forced by the rows below; purple cells are yours to choose.');
  }
  function advance(){ step = Math.min(TOTAL, step + 1); draw();
    if (step >= TOTAL && timer){ clearInterval(timer); timer = null; $('#hessPlay').textContent = '▶ run'; } }
  $('#hessPlay').addEventListener('click', () => {
    if (timer){ clearInterval(timer); timer = null; $('#hessPlay').textContent = '▶ run'; return; }
    if (step >= TOTAL){ step = 0; draw(); }
    if (RM){ step = TOTAL; draw(); return; }
    $('#hessPlay').textContent = '■ pause'; timer = setInterval(advance, 460);
  });
  $('#hessStep').addEventListener('click', () => { if (timer){ clearInterval(timer); timer = null;
    $('#hessPlay').textContent = '▶ run'; } if (step >= TOTAL) step = 0; advance(); });
  $('#hessReset').addEventListener('click', () => { if (timer){ clearInterval(timer); timer = null;
    $('#hessPlay').textContent = '▶ run'; } step = 0; draw(); });
  themeHooks.push(draw); draw();
})();

/* ══════════ 02 · three ways to obey the same constraint ══════════ */
(function(){
  const host = $('#cSvg'); if (!host) return;
  const P = [-0.5, 1], T = [0.78, 0.94], STEPS = 70;
  const pp = P[0]*P[0] + P[1]*P[1], pn = Math.sqrt(pp);
  const U = [P[1]/pn, -P[0]/pn];                    /* unit vector along the constraint line */
  const X0 = 56, X1 = 656, Y0 = 22, Y1 = 316;
  const xr = [-0.2, 1.16], yr = [-0.14, 1.1];
  const sx = v => X0 + (v - xr[0]) / (xr[1] - xr[0]) * (X1 - X0);
  const sy = v => Y1 - (v - yr[0]) / (yr[1] - yr[0]) * (Y1 - Y0);
  let reveal = STEPS, timer = null;
  function paths(){
    const lam = +$('#cLam').value, lr = +$('#cLr').value;
    const etaP = Math.min(lr, 1.7 / (1 + lam * pp));
    const a = [[0,0]], b = [[0,0]], c = [[0,0]];
    let d1 = [0,0], d2 = [0,0], z = 0;
    for (let s = 0; s < STEPS; s++){
      const pd = d1[0]*P[0] + d1[1]*P[1];
      d1 = [d1[0] - etaP*((d1[0]-T[0]) + lam*pd*P[0]), d1[1] - etaP*((d1[1]-T[1]) + lam*pd*P[1])];
      a.push(d1.slice());
      let g = [d2[0] - lr*(d2[0]-T[0]), d2[1] - lr*(d2[1]-T[1])];
      const gd = (g[0]*P[0] + g[1]*P[1]) / pp;
      d2 = [g[0] - gd*P[0], g[1] - gd*P[1]];
      b.push(d2.slice());
      z = z - lr*(z - (U[0]*T[0] + U[1]*T[1]));
      c.push([z*U[0], z*U[1]]);
    }
    return {a, b, c, etaP, lam, lr, clamped: etaP < lr - 1e-9};
  }
  function poly(pts, n, col, w){
    let d = '';
    for (let i = 0; i <= Math.min(n, pts.length-1); i++) d += (i ? 'L' : 'M') + sx(pts[i][0]).toFixed(1) + ',' + sy(pts[i][1]).toFixed(1);
    return '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + w + '" stroke-linejoin="round" opacity=".95"/>';
  }
  function draw(){
    const R = paths();
    let g = '';
    for (let v = 0; v <= 1.0001; v += 0.25){
      g += '<line x1="' + sx(v) + '" y1="' + Y0 + '" x2="' + sx(v) + '" y2="' + Y1 + '" stroke="var(--svgGrid)" stroke-width="1"/>';
      g += '<line x1="' + X0 + '" y1="' + sy(v) + '" x2="' + X1 + '" y2="' + sy(v) + '" stroke="var(--svgGrid)" stroke-width="1"/>';
      g += '<text x="' + sx(v) + '" y="' + (Y1 + 15) + '" text-anchor="middle" font-size="9" fill="var(--faint)">' + v.toFixed(2) + '</text>';
      g += '<text x="' + (X0 - 8) + '" y="' + (sy(v) + 3.5) + '" text-anchor="end" font-size="9" fill="var(--faint)">' + v.toFixed(2) + '</text>';
    }
    /* constraint line, drawn across the plot */
    const t0 = (yr[0] - 0) / U[1], t1 = (yr[1] - 0) / U[1];
    g += '<line x1="' + sx(t0*U[0]) + '" y1="' + sy(t0*U[1]) + '" x2="' + sx(t1*U[0]) + '" y2="' + sy(t1*U[1]) +
      '" stroke="var(--pro)" stroke-width="2" stroke-dasharray="7 5"/>';
    g += '<text x="' + (sx(0.92*U[0]) + 10) + '" y="' + (sy(0.92*U[1]) - 6) + '" font-size="10" font-weight="700" fill="var(--pro)">Δ·k = 0</text>';
    g += '<text x="' + (sx(0.92*U[0]) + 10) + '" y="' + (sy(0.92*U[1]) + 8) + '" font-size="9" fill="var(--faint)">old knowledge untouched</text>';
    g += poly(R.a, reveal, 'var(--hot)', 2);
    g += poly(R.b, reveal, 'var(--obs)', 2);
    g += poly(R.c, reveal, 'var(--ok)', 2.6);
    [[R.a,'var(--hot)','①'],[R.b,'var(--obs)','②'],[R.c,'var(--ok)','③']].forEach(e => {
      const p = e[0][Math.min(reveal, e[0].length-1)];
      g += '<circle cx="' + sx(p[0]) + '" cy="' + sy(p[1]) + '" r="5" fill="' + e[1] + '" stroke="var(--bg)" stroke-width="1.5"/>';
      g += '<text x="' + (sx(p[0]) + 9) + '" y="' + (sy(p[1]) + 4) + '" font-size="11" font-weight="700" fill="' + e[1] + '">' + e[2] + '</text>';
    });
    g += '<circle cx="' + sx(T[0]) + '" cy="' + sy(T[1]) + '" r="9" fill="none" stroke="var(--new)" stroke-width="1.4" stroke-dasharray="3 3"/>';
    g += '<circle cx="' + sx(T[0]) + '" cy="' + sy(T[1]) + '" r="4.5" fill="var(--new)"/>';
    g += '<text x="' + (sx(T[0]) + 14) + '" y="' + (sy(T[1]) + 4) + '" font-size="10" font-weight="700" fill="var(--new)">the new fact</text>';
    g += '<circle cx="' + sx(0) + '" cy="' + sy(0) + '" r="3" fill="var(--faint)"/>';
    g += '<text x="' + (X1 - 4) + '" y="' + (Y1 + 30) + '" text-anchor="end" font-size="10" fill="var(--faint)">Δ component 1  →</text>';
    g += '<text x="' + (X0 - 40) + '" y="' + (Y0 + 4) + '" font-size="10" fill="var(--faint)">Δ comp 2 ↑</text>';
    host.innerHTML = svg(690, 340, g, 8);
    const viol = pts => { const p = pts[Math.min(reveal, pts.length-1)]; return Math.abs(p[0]*P[0] + p[1]*P[1]); };
    const fmt = v => v < 1e-12 ? '≈ 0' : v.toExponential(1).replace('e-', '·10⁻').replace('e+', '·10');
    $('#cV1').textContent = fmt(viol(R.a));
    $('#cV2').textContent = fmt(viol(R.b));
    $('#cV3').textContent = fmt(viol(R.c));
    $('#cLamV').textContent = R.lam.toFixed(R.lam < 10 ? 1 : 0);
    $('#cLrV').textContent = R.lr.toFixed(3);
    const v1 = viol(R.a);
    $('#cNote').innerHTML =
      'At λ = ' + R.lam.toFixed(R.lam < 10 ? 1 : 0) + ' the penalty settles <b class="hh">' + fmt(v1) +
      '</b> away from the constraint — never zero, because the exchange rate between fitting and obeying is finite by ' +
      'construction. Its violation is exactly <span class="mono">(k·t)/(1 + λ‖k‖²)</span>, which decays like 1/λ and ' +
      'arrives nowhere. ' + (R.clamped
        ? '<b class="oo">The step size has been auto-limited to ' + R.etaP.toFixed(4) + '</b> to keep the penalty run stable — which is the second bill λ sends you: raise it to obey more closely and the optimiser must crawl. '
        : '') +
      'Methods ② and ③ both land on the line exactly. They differ in what they cost and in what they do to your optimiser, ' +
      'not in where they end up — that is the subject of the next section.';
  }
  function play(){
    if (timer){ cancelAnimationFrame(timer); timer = null; }
    if (RM){ reveal = STEPS; draw(); return; }
    reveal = 0;
    const tick = () => { reveal += 1; draw(); if (reveal < STEPS) timer = requestAnimationFrame(tick); else timer = null; };
    timer = requestAnimationFrame(tick);
  }
  $('#cPlay').addEventListener('click', play);
  $('#cReset').addEventListener('click', () => { if (timer){ cancelAnimationFrame(timer); timer = null; } reveal = 0; draw(); });
  ['#cLam','#cLr'].forEach(id => $(id).addEventListener('input', () => { reveal = STEPS; draw(); }));
  themeHooks.push(draw); draw();
})();

/* ══════════ 03 · why the basis must be orthogonal ══════════ */
(function(){
  const host = $('#oSvg'); if (!host) return;
  const ZSTAR = [0.72, 0.52], START = [0.05, 0.08], MAXIT = 600, TOL = 0.01;
  const ANG = 25 * Math.PI / 180, ca = Math.cos(ANG), sa = Math.sin(ANG);
  const PW = 318, PH = 268, GAP = 44, PX = [40, 40 + PW + GAP], PY = 24;
  const zr = [-0.1, 1.0], zr2 = [-0.1, 0.95];
  let reveal = MAXIT, timer = null;
  /* G = Rᵀ diag(1, 1/κ) R  ·  eigenvectors at ANG and ANG+90° */
  function gmat(kap){
    const l1 = 1, l2 = 1 / kap;
    const e1 = [ca, sa], e2 = [-sa, ca];
    return [[l1*e1[0]*e1[0] + l2*e2[0]*e2[0], l1*e1[0]*e1[1] + l2*e2[0]*e2[1]],
            [l1*e1[0]*e1[1] + l2*e2[0]*e2[1], l1*e1[1]*e1[1] + l2*e2[1]*e2[1]]];
  }
  function descend(G){
    const eta = 0.6;                                  /* λmax(G) = 1 in both panels */
    const path = [START.slice()]; let z = START.slice(), steps = MAXIT;
    const nz = Math.hypot(ZSTAR[0], ZSTAR[1]);
    for (let i = 0; i < MAXIT; i++){
      const d = [z[0] - ZSTAR[0], z[1] - ZSTAR[1]];
      if (Math.hypot(d[0], d[1]) / nz < TOL){ steps = i; break; }
      z = [z[0] - eta*(G[0][0]*d[0] + G[0][1]*d[1]), z[1] - eta*(G[1][0]*d[0] + G[1][1]*d[1])];
      if (path.length < 420) path.push(z.slice());
    }
    return {path, steps, eta};
  }
  function panel(px, G, kap, label, sub, col, res){
    const sx = v => px + (v - zr[0]) / (zr[1] - zr[0]) * PW;
    const sy = v => PY + PH - (v - zr2[0]) / (zr2[1] - zr2[0]) * PH;
    let g = '<rect x="' + px + '" y="' + PY + '" width="' + PW + '" height="' + PH +
      '" rx="7" fill="var(--svgBox)" stroke="var(--line2)" stroke-width="1"/>';
    g += '<clipPath id="cp' + Math.round(px) + '"><rect x="' + px + '" y="' + PY + '" width="' + PW + '" height="' + PH + '" rx="7"/></clipPath>';
    g += '<g clip-path="url(#cp' + Math.round(px) + ')">';
    const kx = PW / (zr[1] - zr[0]), ky = PH / (zr2[1] - zr2[0]);
    [0.02, 0.07, 0.16, 0.3].forEach(c => {
      const r1 = Math.sqrt(2*c/1), r2 = Math.sqrt(2*c*kap);
      g += '<ellipse cx="' + sx(ZSTAR[0]).toFixed(1) + '" cy="' + sy(ZSTAR[1]).toFixed(1) + '" rx="' + (r1*kx).toFixed(1) +
        '" ry="' + (r2*ky).toFixed(1) + '" transform="rotate(' + (-25) + ' ' + sx(ZSTAR[0]).toFixed(1) + ' ' + sy(ZSTAR[1]).toFixed(1) +
        ')" fill="none" stroke="' + col + '" stroke-width="1" opacity=".33"/>';
    });
    let d = '';
    const np = Math.min(reveal, res.path.length - 1);
    for (let i = 0; i <= np; i++) d += (i ? 'L' : 'M') + sx(res.path[i][0]).toFixed(1) + ',' + sy(res.path[i][1]).toFixed(1);
    g += '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="2.1" stroke-linejoin="round"/>';
    for (let i = 0; i <= np; i += Math.max(1, Math.round(np/14)))
      g += '<circle cx="' + sx(res.path[i][0]).toFixed(1) + '" cy="' + sy(res.path[i][1]).toFixed(1) + '" r="2.1" fill="' + col + '"/>';
    g += '</g>';
    g += '<circle cx="' + sx(ZSTAR[0]) + '" cy="' + sy(ZSTAR[1]) + '" r="5" fill="var(--new)" stroke="var(--bg)" stroke-width="1.5"/>';
    g += '<text x="' + (sx(ZSTAR[0]) + 10) + '" y="' + (sy(ZSTAR[1]) + 4) + '" font-size="9.5" fill="var(--new)">the answer</text>';
    g += '<circle cx="' + sx(START[0]) + '" cy="' + sy(START[1]) + '" r="3.2" fill="var(--faint)"/>';
    g += '<text x="' + px + '" y="' + (PY - 12) + '" font-size="11" font-weight="700" fill="' + col + '">' + label + '</text>';
    g += '<text x="' + (px + PW) + '" y="' + (PY - 12) + '" text-anchor="end" font-size="9.5" fill="var(--faint)">' + sub + '</text>';
    g += '<text x="' + (px + 10) + '" y="' + (PY + PH - 10) + '" font-size="10" font-weight="700" fill="var(--ink)">' +
      (res.steps >= MAXIT ? '> ' + MAXIT : res.steps) + ' steps</text>';
    return g;
  }
  function draw(){
    const kap = +$('#oSkew').value;
    const G1 = gmat(1), G2 = gmat(kap);
    const r1 = descend(G1), r2 = descend(G2);
    let g = panel(PX[0], G1, 1, 'ORTHONORMAL BASIS', 'BᵀB = I  ·  κ = 1', 'var(--ok)', r1);
    g += panel(PX[1], G2, kap, 'SKEWED BASIS', 'BᵀB = G  ·  κ = ' + kap.toFixed(1), 'var(--hot)', r2);
    host.innerHTML = svg(720, 310, g, 8);
    $('#oSkewV').textContent = kap.toFixed(1);
    $('#oS1').textContent = r1.steps >= MAXIT ? '>' + MAXIT : r1.steps;
    $('#oS2').textContent = r2.steps >= MAXIT ? '>' + MAXIT : r2.steps;
    $('#oRatio').textContent = (r2.steps / Math.max(1, r1.steps)).toFixed(1) + '×';
    $('#oLr').textContent = ((1 - r2.eta / kap) * 100).toFixed(1) + '%';
    $('#oNote').innerHTML = kap < 1.05
      ? 'At κ = 1 the two panels are the same picture: circular contours, and the steepest-descent direction points ' +
        'straight at the answer from everywhere. This is the only case in which your learning rate means what you think it means.'
      : 'Both bases span the <em>identical</em> subspace and both satisfy the constraint exactly. The right-hand one ' +
        'needs <b class="hh">' + (r2.steps >= MAXIT ? '>' + MAXIT : r2.steps) + ' steps</b> against ' +
        '<b class="gg">' + r1.steps + '</b>, because the curvature the optimiser sees is <span class="mono">BᵀB</span> — ' +
        'the Gram matrix of the basis you happened to choose, which has nothing to do with the problem. Along the weak ' +
        'direction each step closes only <b class="oo">' + (100 - (1 - r2.eta/kap)*100).toFixed(1) + '%</b> of the remaining gap. ' +
        'Nothing errors, nothing warns you; the run simply converges somewhere else, more slowly, with weight decay ' +
        'pressing unevenly on directions that were supposed to be interchangeable.';
  }
  function play(){
    if (timer){ cancelAnimationFrame(timer); timer = null; }
    if (RM){ reveal = MAXIT; draw(); return; }
    reveal = 0;
    const tick = () => { reveal += 3; draw(); if (reveal < 240) timer = requestAnimationFrame(tick); else { reveal = MAXIT; timer = null; draw(); } };
    timer = requestAnimationFrame(tick);
  }
  $('#oPlay').addEventListener('click', play);
  $('#oSkew').addEventListener('input', () => { reveal = MAXIT; draw(); });
  themeHooks.push(draw); draw();
})();

/* ══════════ 04 · the Gramian spectrum and where you cut it ══════════ */
(function(){
  const host = $('#spSvg'); if (!host) return;
  const N = 40, rnd = mulberry(5150);
  /* output-side sensitivity per direction: mostly modest, with three loud-but-quiet outliers */
  const Q = Array.from({length:N}, () => 0.18 + 0.30 * rnd());
  [26, 31, 36].forEach((i, k) => { Q[i] = 0.97 - 0.06*k; });
  [4, 9].forEach(i => { Q[i] = 0.12 + 0.05*rnd(); });
  let mode = 'one';
  const X0 = 44, X1 = 676, BW = 13, GXs = (X1 - X0) / N;
  const TOPy = 30, TOPh = 62, BOTy = 116, BOTh = 104;
  function draw(){
    const cut = Math.round(+$('#spThr').value / 100 * (N - 2)) + 1;
    const dec = +$('#spDecay').value;
    const lam = Array.from({length:N}, (_, i) => Math.pow(i + 1, -dec));
    const mx = lam[0], tot = lam.reduce((a, b) => a + b, 0);
    const sig = lam.map((l, i) => Math.sqrt(l * Q[i]));
    const byLam = Array.from({length:N}, (_, i) => i);                       /* already descending */
    const bySig = Array.from({length:N}, (_, i) => i).sort((a, b) => sig[b] - sig[a]);
    const protOne = new Set(byLam.slice(0, cut));
    const protTwo = new Set(bySig.slice(0, cut));
    const prot = mode === 'one' ? protOne : protTwo;
    const mis = byLam.filter(i => !protOne.has(i) && protTwo.has(i));
    const misSet = new Set(mis);
    const hLam = l => Math.max(2, (1 + Math.log10(l / mx) / 6) * BOTh);
    let g = '';
    g += '<text x="' + X0 + '" y="' + (TOPy - 10) + '" font-size="9.5" letter-spacing="1.3" fill="var(--obs)">OUTPUT-SIDE SENSITIVITY  ·  how much the rest of the network cares</text>';
    g += '<text x="' + X0 + '" y="' + (BOTy + BOTh + 30) + '" font-size="9.5" letter-spacing="1.3" fill="var(--pro)">INPUT ENERGY OF THE PRESERVED KEYS  ·  log scale  ·  descending</text>';
    for (let i = 0; i < N; i++){
      const x = X0 + i*GXs + (GXs - BW)/2;
      const qh = Math.max(2, Q[i] * TOPh);
      g += '<rect x="' + x.toFixed(1) + '" y="' + TOPy + '" width="' + BW + '" height="' + qh.toFixed(1) +
        '" rx="2" fill="var(--obs)" fill-opacity="' + (0.25 + 0.6*Q[i]).toFixed(2) + '"/>';
      const lh = hLam(lam[i]);
      const isProt = prot.has(i);
      const col = misSet.has(i) && mode === 'two' ? 'var(--hot)' : isProt ? 'var(--pro)' : 'var(--ok)';
      g += '<rect x="' + x.toFixed(1) + '" y="' + (BOTy + BOTh - lh).toFixed(1) + '" width="' + BW + '" height="' + lh.toFixed(1) +
        '" rx="2" fill="' + col + '" fill-opacity="' + (isProt ? .88 : .62) + '"/>';
      if (misSet.has(i) && mode === 'one')
        g += '<rect x="' + x.toFixed(1) + '" y="' + (BOTy + BOTh - lh).toFixed(1) + '" width="' + BW + '" height="' + lh.toFixed(1) +
          '" rx="2" fill="none" stroke="var(--hot)" stroke-width="1.6" stroke-dasharray="2 2"/>';
      if (i % 5 === 0)
        g += '<text x="' + (x + BW/2).toFixed(1) + '" y="' + (BOTy + BOTh + 14) + '" text-anchor="middle" font-size="8.5" fill="var(--faint)">' + i + '</text>';
    }
    if (mode === 'one'){
      const cx = X0 + cut*GXs;
      g += '<line x1="' + cx.toFixed(1) + '" y1="' + (BOTy - 6) + '" x2="' + cx.toFixed(1) + '" y2="' + (BOTy + BOTh + 4) +
        '" stroke="var(--ink)" stroke-width="1.6" stroke-dasharray="4 3"/>';
      g += '<text x="' + (cx + 6).toFixed(1) + '" y="' + (BOTy + 6) + '" font-size="9.5" font-weight="700" fill="var(--ink)">threshold</text>';
    }
    g += '<line x1="' + X0 + '" y1="' + (BOTy + BOTh) + '" x2="' + X1 + '" y2="' + (BOTy + BOTh) + '" stroke="var(--line2)" stroke-width="1"/>';
    g += '<line x1="' + X0 + '" y1="' + TOPy + '" x2="' + X1 + '" y2="' + TOPy + '" stroke="var(--line2)" stroke-width="1"/>';
    host.innerHTML = svg(700, 262, g, 8);
    const risk = byLam.filter(i => !prot.has(i)).reduce((a, i) => a + lam[i], 0) / tot;
    $('#spThrV').textContent = cut + ' of ' + N + ' protected';
    $('#spDecayV').textContent = dec.toFixed(2);
    $('#spDim').textContent = (N - cut);
    $('#spRisk').textContent = (risk * 100).toFixed(risk < 0.01 ? 3 : 2) + '%';
    $('#spMis').textContent = mis.length;
    $('#spCrit').textContent = mode === 'one' ? 'λ  alone' : '√(λ·q)';
    $('#spCritD').textContent = mode === 'one' ? 'input Gramian only' : 'Hankel-style, both Gramians';
    $('#spNote').innerHTML = mode === 'one'
      ? 'The cut keeps the <span class="pp">' + cut + '</span> highest-energy directions protected and hands the ' +
        'remaining <span class="ff">' + (N - cut) + '</span> to the adapter, putting <b>' + (risk*100).toFixed(2) +
        '%</b> of the preserved knowledge&rsquo;s energy at risk. Notice the <b class="hh">' + mis.length + ' dashed bars</b> ' +
        'inside the safe region: those directions are quiet on the input and <em>loud</em> on the output — look at their ' +
        'amber bars above. A one-sided rule cannot see that, because it never looks up.'
      : 'Ranking by <span class="mono">√(λ·q)</span> — the Hankel-style criterion — moves <b class="hh">' + mis.length +
        ' directions</b> out of the safe set and an equal number in. The energy-at-risk figure barely changes; what ' +
        'changes is <em>which</em> knowledge is at risk, and whether the network notices. This is the whole content of ' +
        'balanced truncation: a direction is safe to overwrite only if it is both hard to excite and hard to observe.';
  }
  ['#spThr','#spDecay'].forEach(id => $(id).addEventListener('input', draw));
  [['#spOne','one'],['#spTwo','two']].forEach(e => $(e[0]).addEventListener('click', () => {
    mode = e[1]; $('#spOne').classList.toggle('on', mode === 'one'); $('#spTwo').classList.toggle('on', mode === 'two'); draw(); }));
  themeHooks.push(draw); draw();
})();

/* ══════════ 05 · sequential editing, measured ══════════ */
(function(){
  const host = $('#dSvg'); if (!host) return;
  const D = 12, EDITS = 40, TGT = 0.7;
  const rnd = mulberry(3141), gs = gaussFrom(rnd);
  const rawP = Array.from({length:D}, () => Float64Array.from({length:D}, gs));
  const rawK = Array.from({length:EDITS}, () => { const v = Float64Array.from({length:D}, gs);
    const n = Math.sqrt(dotv(v, v)); for (let i = 0; i < D; i++) v[i] /= n; return v; });
  const X0 = 52, X1 = 664, Y0 = 20, Y1 = 224;
  const LO = -17, HI = 0.6;
  const sx = i => X0 + i / EDITS * (X1 - X0);
  const sy = v => Y1 - (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * (Y1 - Y0);
  let reveal = EDITS, timer = null, cache = null;
  function simulate(){
    const m = +$('#dProt').value, lam = +$('#dLam').value;
    const Kp = ortho(rawP.slice(0, m));
    const eye = Array.from({length:D}, (_, i) => Float64Array.from({length:D}, (_, j) => i === j ? 1 : 0));
    const full = ortho([...Kp, ...eye]);
    const Un = full.slice(Kp.length);
    const KpKp = Array.from({length:D}, () => new Float64Array(D));
    for (const pv of Kp) for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) KpKp[a][b] += pv[a]*pv[b];
    const dPen = new Float64Array(D), dPrj = new Float64Array(D), dRep = new Float64Array(D);
    const v1 = [], v2 = [], v3 = []; let penFit = 0;
    const viol = dd => { let s = 0; for (const pv of Kp){ const c = dotv(dd, pv); s += c*c; } return Math.sqrt(s); };
    for (let e = 0; e < EDITS; e++){
      const k = rawK[e];
      /* ① penalty */
      let rho = TGT - dotv(dPen, k);
      const M = Array.from({length:D}, (_, a) => Float64Array.from({length:D},
        (_, b) => k[a]*k[b] + lam*KpKp[a][b] + (a === b ? 1e-6 : 0)));
      const sol = gsolve(M, Array.from({length:D}, (_, a) => Float64Array.from([k[a]*rho])));
      for (let a = 0; a < D; a++) dPen[a] += sol[a][0];
      penFit = Math.abs(TGT - dotv(dPen, k));
      /* ② project every step */
      rho = TGT - dotv(dPrj, k);
      const Pk = Float64Array.from(k);
      for (const pv of Kp){ const c = dotv(k, pv); for (let a = 0; a < D; a++) Pk[a] -= c*pv[a]; }
      const dn = dotv(k, Pk);
      if (Math.abs(dn) > 1e-12) for (let a = 0; a < D; a++) dPrj[a] += Pk[a]*rho/dn;
      /* ③ change of variables */
      rho = TGT - dotv(dRep, k);
      const c = Un.map(u => dotv(u, k));
      const cc = c.reduce((a, x) => a + x*x, 0);
      if (cc > 1e-12) Un.forEach((u, j) => { const z = c[j]*rho/cc; for (let a = 0; a < D; a++) dRep[a] += z*u[a]; });
      v1.push(viol(dPen)); v2.push(viol(dPrj)); v3.push(viol(dRep));
    }
    return {v1, v2, v3, penFit, m, lam, nullDim: Un.length};
  }
  function line(vs, n, col, w){
    let d = '';
    for (let i = 0; i < Math.min(n, vs.length); i++){
      const y = sy(Math.log10(Math.max(vs[i], 1e-17)));
      d += (i ? 'L' : 'M') + sx(i + 1).toFixed(1) + ',' + y.toFixed(1);
    }
    return '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + w + '" stroke-linejoin="round"/>';
  }
  function draw(){
    if (!cache) cache = simulate();
    const R = cache;
    let g = '';
    for (let p = 0; p >= -16; p -= 4){
      const y = sy(p);
      g += '<line x1="' + X0 + '" y1="' + y.toFixed(1) + '" x2="' + X1 + '" y2="' + y.toFixed(1) + '" stroke="var(--svgGrid)" stroke-width="1"/>';
      g += '<text x="' + (X0 - 8) + '" y="' + (y + 3.5).toFixed(1) + '" text-anchor="end" font-size="9" fill="var(--faint)">10' +
        (p === 0 ? '⁰' : ('⁻' + String(-p).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join(''))) + '</text>';
    }
    for (let i = 0; i <= EDITS; i += 10){
      g += '<line x1="' + sx(i).toFixed(1) + '" y1="' + Y0 + '" x2="' + sx(i).toFixed(1) + '" y2="' + Y1 + '" stroke="var(--svgGrid)" stroke-width="1"/>';
      g += '<text x="' + sx(i).toFixed(1) + '" y="' + (Y1 + 15) + '" text-anchor="middle" font-size="9" fill="var(--faint)">' + i + '</text>';
    }
    g += '<text x="' + X1 + '" y="' + (Y1 + 32) + '" text-anchor="end" font-size="9.5" fill="var(--faint)">edits applied  →</text>';
    g += '<text x="' + X0 + '" y="' + (Y0 - 6) + '" font-size="9.5" letter-spacing="1.2" fill="var(--faint)">DAMAGE TO PRESERVED KEYS  ‖K₀ᵀΔ‖</text>';
    g += line(R.v3, reveal, 'var(--ok)', 2.8);
    g += line(R.v2, reveal, 'var(--obs)', 1.8);
    g += line(R.v1, reveal, 'var(--hot)', 2.4);
    const lab = (vs, col, txt) => { const i = Math.min(reveal, vs.length) - 1; if (i < 0) return '';
      const y = sy(Math.log10(Math.max(vs[i], 1e-17)));
      return '<circle cx="' + sx(i+1).toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.4" fill="' + col + '"/>' +
        '<text x="' + (sx(i+1) - 6).toFixed(1) + '" y="' + (y - 8).toFixed(1) + '" text-anchor="end" font-size="9.5" font-weight="700" fill="' + col + '">' + txt + '</text>'; };
    g += lab(R.v1, 'var(--hot)', '① penalty');
    g += lab(R.v2, 'var(--obs)', '② projected');
    g += lab(R.v3, 'var(--ok)', '③ re-parametrised');
    g += '<text x="' + (X0 + 6) + '" y="' + (sy(-16.4)).toFixed(1) + '" font-size="9" fill="var(--faint)">floating-point floor</text>';
    host.innerHTML = svg(700, 250, g, 8);
    const fmt = v => v < 1e-15 ? '≈ 10⁻¹⁶' : v.toExponential(1).replace('e-', '·10⁻').replace('e+', '·10');
    const last = vs => vs[Math.min(reveal, vs.length) - 1] ?? 0;
    $('#dF1').textContent = fmt(last(R.v1));
    $('#dF2').textContent = fmt(last(R.v2));
    $('#dF3').textContent = fmt(last(R.v3));
    $('#dFit').textContent = fmt(R.penFit);
    $('#dLamV').textContent = R.lam;
    $('#dProtV').textContent = R.m;
    $('#dNote').innerHTML = 'Twelve dimensions, <b class="pp">' + R.m + '</b> protected directions, leaving <b class="ff">' +
      R.nullDim + '</b> for the adapter. After ' + Math.min(reveal, EDITS) + ' edits the penalty has accumulated ' +
      '<b class="hh">' + fmt(last(R.v1)) + '</b> of damage while paying <b>' + fmt(R.penFit) + '</b> in fit error on the ' +
      'edit it just made — both bills at once. The two exact methods agree with each other to the last bit, because ' +
      'they are the same constraint imposed two ways; what separates them is cost and optimiser geometry, not accuracy. ' +
      'Push λ up and watch the red curve fall and the fit error rise: that is the exchange rate, and it never reaches zero.';
  }
  function play(){
    if (timer){ cancelAnimationFrame(timer); timer = null; }
    if (RM){ reveal = EDITS; draw(); return; }
    reveal = 0;
    const tick = () => { reveal += 1; draw(); if (reveal < EDITS) timer = requestAnimationFrame(tick); else timer = null; };
    timer = requestAnimationFrame(tick);
  }
  $('#dPlay').addEventListener('click', play);
  $('#dReset').addEventListener('click', () => { if (timer){ cancelAnimationFrame(timer); timer = null; } reveal = 0; draw(); });
  ['#dLam','#dProt'].forEach(id => $(id).addEventListener('input', () => { cache = null; reveal = EDITS; draw(); }));
  themeHooks.push(draw); draw();
})();

/* ══════════ 06 · walking the solution set ══════════ */
(function(){
  const host = $('#fCurve'); if (!host) return;
  const D = 40, O = 6, NP = 8, NF = 2, HELD = 300, NC = NP + NF;
  const rnd = mulberry(20260926), gs = gaussFrom(rnd);
  const B = ortho(Array.from({length:D}, () => Float64Array.from({length:D}, gs)));
  const rawC = Array.from({length:NC}, () => Float64Array.from({length:D}, gs));
  const rawH = Array.from({length:HELD}, () => Float64Array.from({length:D}, gs));
  const RHS = Array.from({length:O}, () => Float64Array.from({length:NC},
    (_, j) => j < NP ? 0 : gs() * 0.5));
  let model = null;
  function build(kap){
    const lam = Array.from({length:D}, (_, i) => Math.exp(-Math.log(kap) * i / (D - 1)));
    const apply = (x, f) => { const y = new Float64Array(D);
      for (let i = 0; i < D; i++){ const c = f(lam[i]) * dotv(B[i], x);
        for (let j = 0; j < D; j++) y[j] += c * B[i][j]; } return y; };
    const half = x => apply(x, l => Math.sqrt(l)), cinv = x => apply(x, l => 1 / l);
    const K = rawC.map(half), H = rawH.map(half);
    function sol(W){
      const WK = K.map(W);
      const G = Array.from({length:NC}, (_, i) => Float64Array.from({length:NC}, (_, j) => dotv(K[i], WK[j])));
      const Y = gsolve(G, Array.from({length:NC}, (_, i) => Float64Array.from({length:O}, (_, o) => RHS[o][i])));
      return Array.from({length:O}, (_, o) => { const row = new Float64Array(D);
        for (let j = 0; j < NC; j++){ const y = Y[j][o]; for (let a = 0; a < D; a++) row[a] += WK[j][a] * y; }
        return row; });
    }
    const DE = sol(x => x), DM = sol(cinv);
    const aE = H.map(k => Float64Array.from({length:O}, (_, o) => dotv(DE[o], k)));
    const aM = H.map(k => Float64Array.from({length:O}, (_, o) => dotv(DM[o], k)));
    const cE = K.map(k => Float64Array.from({length:O}, (_, o) => dotv(DE[o], k)));
    const cM = K.map(k => Float64Array.from({length:O}, (_, o) => dotv(DM[o], k)));
    const curve = [];
    for (let i = -60; i <= 160; i += 2){ const u = i / 100;
      let mx = 0, ss = 0;
      for (let j = 0; j < HELD; j++){ let s2 = 0;
        for (let o = 0; o < O; o++){ const v = (1-u)*aE[j][o] + u*aM[j][o]; s2 += v*v; }
        ss += s2; if (s2 > mx) mx = s2; }
      curve.push([u, Math.sqrt(mx), Math.sqrt(ss / HELD)]);
    }
    return {aE, aM, cE, cM, curve, kap};
  }
  function at(u){
    const M = model; let mx = 0, ss = 0; const prof = new Float64Array(HELD);
    for (let j = 0; j < HELD; j++){ let s2 = 0;
      for (let o = 0; o < O; o++){ const v = (1-u)*M.aE[j][o] + u*M.aM[j][o]; s2 += v*v; }
      prof[j] = Math.sqrt(s2); ss += s2; if (s2 > mx) mx = s2; }
    let fit = 0, dmg = 0;
    for (let i = 0; i < NC; i++) for (let o = 0; o < O; o++){
      const v = (1-u)*M.cE[i][o] + u*M.cM[i][o], e = Math.abs(v - RHS[o][i]);
      if (i < NP) dmg = Math.max(dmg, e); else fit = Math.max(fit, e); }
    prof.sort((a, b) => b - a);
    return {mx: Math.sqrt(mx), rms: Math.sqrt(ss/HELD), fit, dmg, prof};
  }
  function draw(){
    const kap = Math.pow(10, +$('#fK').value / 100 * 3.5);
    if (!model || Math.abs(model.kap - kap) > 1e-9) model = build(kap);
    const u = +$('#fT').value / 100;
    const r = at(u), r0 = at(0), r1 = at(1);
    const cv = model.curve;
    const ymax = Math.max.apply(null, cv.map(p => p[1])) * 1.08, ymin = 0;
    const X0 = 46, X1 = 322, Y0 = 18, Y1 = 176;
    const sx = v => X0 + (v + 0.6) / 2.2 * (X1 - X0);
    const sy = v => Y1 - (v - ymin) / (ymax - ymin) * (Y1 - Y0);
    let g = '';
    [0, 0.5, 1].forEach(v => { g += '<line x1="' + sx(v) + '" y1="' + Y0 + '" x2="' + sx(v) + '" y2="' + Y1 +
      '" stroke="var(--svgGrid)" stroke-width="1"/>' +
      '<text x="' + sx(v) + '" y="' + (Y1 + 14) + '" text-anchor="middle" font-size="9" fill="var(--faint)">' + v + '</text>'; });
    const pth = (idx, col, w) => { let d = '';
      cv.forEach((p, i) => { d += (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[idx]).toFixed(1); });
      return '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + w + '"/>'; };
    g += pth(2, 'var(--obs)', 1.6) + pth(1, 'var(--hot)', 2.2);
    [[0, 'var(--pro)', 'Euclidean'], [1, 'var(--ok)', 'Gramian']].forEach(e => {
      const rr = e[0] === 0 ? r0 : r1;
      g += '<line x1="' + sx(e[0]) + '" y1="' + Y0 + '" x2="' + sx(e[0]) + '" y2="' + Y1 +
        '" stroke="' + e[1] + '" stroke-width="1.3" stroke-dasharray="4 3"/>';
      g += '<circle cx="' + sx(e[0]) + '" cy="' + sy(rr.mx) + '" r="4.5" fill="' + e[1] + '" stroke="var(--bg)" stroke-width="1.2"/>';
      g += '<text x="' + (sx(e[0]) + (e[0] ? 7 : -7)) + '" y="' + (Y0 + 12) + '" text-anchor="' + (e[0] ? 'start' : 'end') +
        '" font-size="9" font-weight="700" fill="' + e[1] + '">' + e[2] + '</text>';
    });
    g += '<circle cx="' + sx(u) + '" cy="' + sy(r.mx) + '" r="5.5" fill="var(--hot)" stroke="var(--bg)" stroke-width="1.6"/>';
    g += '<text x="' + X0 + '" y="' + (Y1 + 32) + '" font-size="9" fill="var(--faint)">position u along the walk  →</text>';
    g += '<text x="' + (X0 - 6) + '" y="' + (Y0 + 4) + '" text-anchor="end" font-size="9" fill="var(--faint)">leak</text>';
    host.innerHTML = svg(340, 198, g, 8);
    /* sorted leakage profile */
    const p0 = r0.prof, pc = r.prof;
    const BX0 = 40, BX1 = 322, BY0 = 18, BY1 = 176;
    const pmax = Math.max(p0[0], pc[0]) * 1.06;
    const px = i => BX0 + i / (HELD - 1) * (BX1 - BX0);
    const py = v => BY1 - v / pmax * (BY1 - BY0);
    let h = '';
    [[p0, 'var(--pro)', 1.4, '4 3'], [pc, 'var(--hot)', 2.2, '']].forEach(e => {
      let d = ''; for (let i = 0; i < HELD; i += 2) d += (i ? 'L' : 'M') + px(i).toFixed(1) + ',' + py(e[0][i]).toFixed(1);
      h += '<path d="' + d + '" fill="none" stroke="' + e[1] + '" stroke-width="' + e[2] + '"' +
        (e[3] ? ' stroke-dasharray="' + e[3] + '"' : '') + '/>';
    });
    h += '<line x1="' + BX0 + '" y1="' + BY1 + '" x2="' + BX1 + '" y2="' + BY1 + '" stroke="var(--line2)" stroke-width="1"/>';
    h += '<text x="' + BX0 + '" y="' + (BY1 + 15) + '" font-size="9" fill="var(--faint)">worst key</text>';
    h += '<text x="' + BX1 + '" y="' + (BY1 + 15) + '" text-anchor="end" font-size="9" fill="var(--faint)">300th</text>';
    h += '<text x="' + (BX0 + 8) + '" y="' + (BY0 + 10) + '" font-size="9" fill="var(--pro)">— — Euclidean choice</text>';
    h += '<text x="' + (BX0 + 8) + '" y="' + (BY0 + 24) + '" font-size="9" fill="var(--hot)">■ current position</text>';
    $('#fBall').innerHTML = svg(340, 198, h, 8);
    const fmt = v => v < 1e-13 ? '≈ 10⁻¹⁵' : v.toExponential(1).replace('e-', '·10⁻').replace('e+', '·10');
    $('#fTV').textContent = u.toFixed(2);
    $('#fKV').textContent = kap < 10 ? kap.toFixed(1) : Math.round(kap);
    $('#fFit').textContent = fmt(r.fit);
    $('#fDmg').textContent = fmt(r.dmg);
    $('#fSig').textContent = r.mx.toFixed(3);
    const pen = r0.mx / r1.mx;
    $('#fPen').textContent = pen.toFixed(2) + '×';
    $('#fNote').innerHTML = '40-dimensional keys, 6 outputs, ' + NP + ' protected keys and ' + NF + ' new facts — ' +
      '<b>60 constraints, ' + (O * (D - NC)) + ' free parameters</b>. At κ = ' + (kap < 10 ? kap.toFixed(1) : Math.round(kap)) +
      ' the Euclidean choice leaks <b class="hh">' + pen.toFixed(2) + '×</b> more onto the 300 unsampled preserved keys than the ' +
      'Gramian-weighted choice does — at <b class="gg">identical</b> fit (' + fmt(r.fit) + ') and identical exact preservation (' +
      fmt(r.dmg) + ') all the way along. ' + (pen < 1.03
        ? 'At this conditioning the two choices are effectively the same point, which is the content of finding 1 below.'
        : 'Drag κ to the left and the advantage disappears entirely; drag it right and it grows. The conditioning of the Gramian is the exchange rate.');
  }
  ['#fT','#fK'].forEach(id => $(id).addEventListener('input', draw));
  $('#fE').addEventListener('click', () => { $('#fT').value = 0; draw(); });
  $('#fM').addEventListener('click', () => { $('#fT').value = 100; draw(); });
  themeHooks.push(draw); draw();
})();

/* ══════════ 08 · the layer-spreading coefficients ══════════ */
(function(){
  const host = $('#lySvg'); if (!host) return;
  const L = 8, LO = 3, n = L - LO + 1;
  const pub = [], reach = [];
  for (let l = LO; l <= L; l++){ pub.push(1 / (L - l + 1)); reach.push(Math.exp(-(L - l) * 0.62)); }
  const nrm = a => { const s = a.reduce((x, y) => x + y, 0); return a.map(v => v / s); };
  const P = nrm(pub), Rc = nrm(reach);
  let mode = 'pub', hov = -1;
  const X0 = 58, X1 = 660, Y0 = 26, Y1 = 186, gw = (X1 - X0) / n;
  function draw(){
    const act = mode === 'pub' ? P : Rc;
    let g = '';
    for (let i = 0; i <= 4; i++){ const y = Y1 - i/4 * (Y1 - Y0);
      g += '<line x1="' + X0 + '" y1="' + y.toFixed(1) + '" x2="' + X1 + '" y2="' + y.toFixed(1) + '" stroke="var(--svgGrid)" stroke-width="1"/>';
      g += '<text x="' + (X0 - 8) + '" y="' + (y + 3.5).toFixed(1) + '" text-anchor="end" font-size="9" fill="var(--faint)">' +
        (i/4 * 0.4 * 100).toFixed(0) + '%</text>'; }
    for (let i = 0; i < n; i++){
      const l = LO + i, cx = X0 + i*gw, bw = gw * 0.34;
      const hP = P[i] / 0.4 * (Y1 - Y0), hR = Rc[i] / 0.4 * (Y1 - Y0);
      const on = mode === 'pub';
      g += '<rect x="' + (cx + gw*0.12).toFixed(1) + '" y="' + (Y1 - hP).toFixed(1) + '" width="' + bw.toFixed(1) +
        '" height="' + hP.toFixed(1) + '" rx="3" fill="var(--pro)" fill-opacity="' + (on ? .9 : .28) + '"/>';
      g += '<rect x="' + (cx + gw*0.52).toFixed(1) + '" y="' + (Y1 - hR).toFixed(1) + '" width="' + bw.toFixed(1) +
        '" height="' + hR.toFixed(1) + '" rx="3" fill="var(--obs)" fill-opacity="' + (on ? .28 : .9) + '"' +
        (on ? ' stroke="var(--obs)" stroke-width="1" stroke-dasharray="3 2"' : '') + '/>';
      g += '<text x="' + (cx + gw/2).toFixed(1) + '" y="' + (Y1 + 16) + '" text-anchor="middle" font-size="10" font-weight="700" fill="var(--ink)">layer ' + l + '</text>';
      if (l === L) g += '<text x="' + (cx + gw/2).toFixed(1) + '" y="' + (Y1 + 29) + '" text-anchor="middle" font-size="8.5" fill="var(--new)">residual computed here</text>';
      g += '<text x="' + (cx + gw*0.12 + bw/2).toFixed(1) + '" y="' + (Y1 - hP - 5).toFixed(1) + '" text-anchor="middle" font-size="9" fill="var(--pro)">' +
        (P[i]*100).toFixed(0) + '</text>';
      g += '<text x="' + (cx + gw*0.52 + bw/2).toFixed(1) + '" y="' + (Y1 - hR - 5).toFixed(1) + '" text-anchor="middle" font-size="9" fill="var(--obs)">' +
        (Rc[i]*100).toFixed(0) + '</text>';
    }
    g += '<text x="' + X0 + '" y="' + (Y0 - 10) + '" font-size="9.5" letter-spacing="1.2" fill="var(--faint)">SHARE OF THE RESIDUAL EACH LAYER RECEIVES</text>';
    g += '<rect x="' + (X1 - 232) + '" y="' + (Y0 - 2) + '" width="10" height="10" rx="2" fill="var(--pro)"/>' +
         '<text x="' + (X1 - 217) + '" y="' + (Y0 + 7) + '" font-size="9.5" fill="var(--dim)">MEMIT · 1/(L−ℓ+1)</text>';
    g += '<rect x="' + (X1 - 100) + '" y="' + (Y0 - 2) + '" width="10" height="10" rx="2" fill="var(--obs)"/>' +
         '<text x="' + (X1 - 85) + '" y="' + (Y0 + 7) + '" font-size="9.5" fill="var(--dim)">reach · illustrative</text>';
    host.innerHTML = svg(700, 228, g, 8);
    $('#lyTop').textContent = (act[n-1]*100).toFixed(0) + '%';
    $('#lyBot').textContent = (act[0]*100).toFixed(0) + '%';
    $('#lyBasis').textContent = mode === 'pub' ? '1/(L−ℓ+1)' : 'survival';
    $('#lyBasisD').textContent = mode === 'pub' ? 'layers remaining · published' : 'change reaching the output · schematic';
    $('#lyNote').innerHTML = mode === 'pub'
      ? 'The published rule gives layer 8 <b class="pp">' + (P[n-1]*100).toFixed(0) + '%</b> of the residual and layer 3 ' +
        '<b class="pp">' + (P[0]*100).toFixed(0) + '%</b>. The only input to that split is <em>how many layers are left</em>. ' +
        'Nothing about this model, this layer or this edit enters the calculation — which is fine as a first guess and ' +
        'indefensible as a final answer, given that the quantity it stands in for is measurable with forward passes alone.'
      : '<b class="oo">This second series is illustrative, not measured</b> — a plausible exponential decay for how much of ' +
        'an injected change survives composition to the top of the range. Its shape matters less than its existence: it is ' +
        'the quantity the coefficients should be computed <em>from</em>. BLUE&rsquo;s empirical finding — that the ' +
        'contribution drops sharply with distance and the far layers stop paying — is what you would predict if the real ' +
        'reach profile decays faster than 1/(L−ℓ+1) does, which is exactly what the comparison above shows.';
  }
  [['#lyPub','pub'],['#lyReach','reach']].forEach(e => $(e[0]).addEventListener('click', () => {
    mode = e[1]; $('#lyPub').classList.toggle('on', mode === 'pub'); $('#lyReach').classList.toggle('on', mode === 'reach'); draw(); }));
  themeHooks.push(draw); draw();
})();

/* ══════════ primer 1 · the stack, and one block opened up ══════════ */
(function(){
  const host = $('#archSvg'); if (!host) return;
  const MODELS = [
    {n:'GPT-2 small', L:12, d:768,  ff:3072,  kv:768,  mlp:2, vocab:50257, tie:1},
    {n:'GPT-2 XL',    L:48, d:1600, ff:6400,  kv:1600, mlp:2, vocab:50257, tie:1},
    {n:'GPT-J 6B',    L:28, d:4096, ff:16384, kv:4096, mlp:2, vocab:50400, tie:1},
    {n:'Llama-3 8B',  L:32, d:4096, ff:14336, kv:1024, mlp:3, vocab:128256, tie:2}
  ];
  let mi = 0, sel = 'down', selLayer = null;
  const fmt = v => v >= 1e9 ? (v/1e9).toFixed(2) + 'B' : v >= 1e6 ? (v/1e6).toFixed(1) + 'M'
    : v >= 1e3 ? (v/1e3).toFixed(1) + 'K' : String(v);
  function mats(M){
    const a = [
      {id:'q', nm:'W_Q',    r:M.d,  c:M.d, col:'var(--cold)', role:'what each token is looking for'},
      {id:'k', nm:'W_K',    r:M.kv, c:M.d, col:'var(--cold)', role:'what each token advertises'},
      {id:'v', nm:'W_V',    r:M.kv, c:M.d, col:'var(--cold)', role:'what each token offers up'},
      {id:'o', nm:'W_O',    r:M.d,  c:M.d, col:'var(--cold)', role:'mixes the retrieved values back in'},
      {id:'up', nm:'W_up',  r:M.ff, c:M.d, col:'var(--obs)',  role:'expands to the wide inner layer'}
    ];
    if (M.mlp === 3) a.push({id:'gate', nm:'W_gate', r:M.ff, c:M.d, col:'var(--obs)', role:'the SwiGLU gate, elementwise on W_up'});
    a.push({id:'down', nm:'W_down', r:M.d, c:M.ff, col:'var(--pro)',
            role:'the key→value store — the matrix this page is about'});
    return a;
  }
  function draw(){
    const M = MODELS[mi], A = mats(M);
    const per = 2*M.d*M.d + 2*M.kv*M.d + M.mlp*M.d*M.ff;
    const emb = M.vocab * M.d * M.tie;
    const total = M.L * per + emb;
    const SH = 16, nS = Math.min(M.L, 14);
    let g = '';
    /* isometric stack */
    const bx = 66, by = 34, w = 116, dpth = 30;
    for (let i = nS - 1; i >= 0; i--){
      const y = by + i * SH, on = selLayer === i;
      const crit = i >= Math.floor(nS*0.18) && i <= Math.floor(nS*0.5);
      g += '<path d="M' + bx + ',' + y + ' l' + dpth + ',-' + (dpth*0.52) + ' l' + w + ',0 l-' + dpth + ',' + (dpth*0.52) + ' z" fill="' +
        (on ? 'var(--free)' : crit ? 'var(--pro)' : 'var(--svgCell)') + '" fill-opacity="' + (on ? .95 : crit ? .55 : .35) +
        '" stroke="var(--line2)" stroke-width="0.8"/>';
      g += '<rect x="' + bx + '" y="' + y + '" width="' + w + '" height="' + (SH-3) + '" fill="' +
        (on ? 'var(--free)' : crit ? 'var(--pro)' : 'var(--svgBox)') + '" fill-opacity="' + (on ? .75 : crit ? .3 : .85) +
        '" stroke="var(--line2)" stroke-width="0.8"/>';
      g += '<rect class="lyHit" data-i="' + i + '" x="' + bx + '" y="' + (y-6) + '" width="' + (w+dpth) + '" height="' + SH + '" fill="transparent" style="cursor:pointer"/>';
    }
    g += '<text x="' + bx + '" y="' + (by - 18) + '" font-size="10" fill="var(--faint)">' +
      (M.L > nS ? nS + ' of ' + M.L + ' blocks' : M.L + ' identical blocks') + '</text>';
    g += '<text x="' + bx + '" y="' + (by + nS*SH + 20) + '" font-size="9.5" fill="var(--pro)">shaded = where editing work concentrates</text>';
    g += '<text x="' + (bx - 42) + '" y="' + (by + nS*SH/2) + '" font-size="9.5" fill="var(--faint)" transform="rotate(-90 ' +
      (bx - 42) + ' ' + (by + nS*SH/2) + ')" text-anchor="middle">depth</text>';
    /* one block, opened */
    const ox = 300, oy = 26, bw = 372;
    g += '<line x1="' + (ox + 22) + '" y1="' + oy + '" x2="' + (ox + 22) + '" y2="' + (oy + 268) +
      '" stroke="var(--new)" stroke-width="2.4"/>';
    g += '<text x="' + (ox + 4) + '" y="' + (oy + 290) + '" font-size="9.5" fill="var(--new)">residual stream · width ' + M.d + ' · every block adds its correction back into it</text>';
    const groups = [
      {t:'ATTENTION · tokens read from one another', ids:['q','k','v','o'], y:oy + 6},
      {t:'MLP · each token on its own', ids: M.mlp === 3 ? ['up','gate','down'] : ['up','down'], y:oy + 140}
    ];
    groups.forEach(gr => {
      const items = A.filter(m => gr.ids.indexOf(m.id) >= 0);
      const gh = 44 + Math.ceil(items.length / 2) * 42;
      g += '<rect x="' + (ox + 46) + '" y="' + gr.y + '" width="' + (bw - 60) + '" height="' + gh +
        '" rx="8" fill="var(--svgBox)" stroke="var(--line2)" stroke-width="1"/>';
      g += '<text x="' + (ox + 60) + '" y="' + (gr.y + 20) + '" font-size="9.5" letter-spacing="1.1" fill="var(--faint)">' + gr.t + '</text>';
      g += '<line x1="' + (ox + 22) + '" y1="' + (gr.y + gh/2) + '" x2="' + (ox + 46) + '" y2="' + (gr.y + gh/2) +
        '" stroke="var(--svgArrow)" stroke-width="1.2"/>';
      items.forEach((m, i) => {
        const cx = ox + 62 + (i % 2) * 158, cy = gr.y + 32 + Math.floor(i / 2) * 42;
        const on = sel === m.id;
        g += '<rect class="mtHit" data-id="' + m.id + '" x="' + cx + '" y="' + cy + '" width="146" height="34" rx="6" fill="' + m.col +
          '" fill-opacity="' + (on ? .34 : .13) + '" stroke="' + m.col + '" stroke-width="' + (on ? 2.2 : 1) + '" style="cursor:pointer"/>';
        g += '<text x="' + (cx + 10) + '" y="' + (cy + 15) + '" font-size="11" font-weight="700" fill="var(--ink)" pointer-events="none">' + m.nm + '</text>';
        g += '<text x="' + (cx + 10) + '" y="' + (cy + 27) + '" font-size="9" fill="' + (on ? 'var(--txt)' : 'var(--dim)') +
          '" pointer-events="none">' + m.r + ' × ' + m.c + '  ·  ' + fmt(m.r*m.c) + '</text>';
      });
    });
    host.innerHTML = svg(700, 324, g, 10);
    host.querySelectorAll('.mtHit').forEach(r => r.addEventListener('click', () => { sel = r.dataset.id; draw(); }));
    host.querySelectorAll('.lyHit').forEach(r => r.addEventListener('click', () => {
      selLayer = selLayer === +r.dataset.i ? null : +r.dataset.i; draw(); }));
    const m = A.find(x => x.id === sel) || A[A.length-1];
    const pm = m.r * m.c;
    $('#arName').textContent = m.nm;
    $('#arRole').textContent = m.role;
    $('#arShape').textContent = m.r + ' × ' + m.c;
    $('#arParams').textContent = fmt(pm);
    $('#arShare').textContent = (100*pm/per).toFixed(1) + '% of one block';
    $('#arTotal').textContent = fmt(total);
    $('#arLayers').textContent = M.L + ' × ' + fmt(per) + ' + ' + fmt(emb) + ' emb · matrices only';
    $('#arNote').innerHTML = sel === 'down'
      ? '<b class="pp">W_down</b> is the down-projection: it takes the wide post-nonlinearity vector — ' + M.ff +
        ' numbers, the <em>key</em> — and produces the ' + M.d + '-number <em>value</em> that gets added back to the stream. ' +
        'It is linear, so it behaves like a lookup table, and it is the largest single matrix in the block at <b>' + fmt(pm) +
        '</b> parameters. Every closed-form editing method in this field writes into exactly this matrix.'
      : (sel === 'up' || sel === 'gate')
        ? m.nm + ' expands each token from ' + M.d + ' to ' + M.ff + ' dimensions. The nonlinearity after it is what makes ' +
          'the MLP more than one big linear map — and it is why the <em>key</em> that reaches W_down is a nonlinear function ' +
          'of the input, while the step from key to value stays perfectly linear.'
        : m.nm + ' belongs to attention, where the mixing weights are computed from the input itself. That input dependence ' +
          'is exactly what stops the Gramian machinery on this page from applying here — see primer step 8.';
  }
  $('#archModel').addEventListener('change', e => { mi = +e.target.value; selLayer = null; draw(); });
  themeHooks.push(draw); draw();
})();

/* ══════════ primer 2 · one sentence, all the way to a weight change ══════════ */
(function(){
  const host = $('#pipeSvg'); if (!host) return;
  const TOKS = ['The', ' renal', ' cor', 'pus', 'cle', ' filters', ' blood', ' in', ' the'];
  const CANDS = [['ĠKid', .31], ['Ġblood', .19], ['Ġbody', .11], ['Ġneph', .02], ['Ġliver', .02]];
  const STAGES = [
    {t:'1 · a sentence from your corpus', s:'tokenise'},
    {t:'2 · forward pass', s:'predict the next token'},
    {t:'3 · the loss', s:'−log p of the token that actually came next'},
    {t:'4 · backward pass', s:'∂L/∂W for every matrix at once'},
    {t:'5 · the update', s:'W ← W − η ∂L/∂W'}
  ];
  let st = 0, timer = null;
  const rnd = mulberry(909);
  const grad = Array.from({length:6}, () => Array.from({length:10}, () => rnd()));
  function draw(){
    const X0 = 20, W = 660, bw = W / 5;
    let g = '';
    STAGES.forEach((S, i) => {
      const x = X0 + i * bw, on = i === st;
      g += '<rect x="' + (x + 3) + '" y="18" width="' + (bw - 8) + '" height="30" rx="6" fill="' +
        (on ? 'var(--new)' : 'var(--svgBox)') + '" fill-opacity="' + (on ? .9 : 1) + '" stroke="' +
        (on ? 'var(--new)' : 'var(--line2)') + '" stroke-width="1"/>';
      g += '<text x="' + (x + bw/2) + '" y="' + 37 + '" text-anchor="middle" font-size="9.5" font-weight="' + (on ? 700 : 400) +
        '" fill="' + (on ? 'var(--ink)' : 'var(--dim)') + '">' + S.t + '</text>';
      if (i < 4) g += '<path d="M' + (x + bw - 4) + ',33 l5,0" stroke="var(--svgArrow)" stroke-width="1.2"/>';
    });
    const CY = 74, H = 150;
    g += '<rect x="' + X0 + '" y="' + CY + '" width="' + W + '" height="' + H + '" rx="8" fill="var(--inset)" stroke="var(--line)" stroke-width="1"/>';
    if (st === 0){
      let x = X0 + 18;
      TOKS.forEach((t, i) => {
        const w = 12 + t.length * 7.4;
        g += '<rect x="' + x + '" y="' + (CY + 42) + '" width="' + w + '" height="26" rx="5" fill="var(--pro)" fill-opacity=".2" stroke="var(--pro)" stroke-width="0.9"/>';
        g += '<text x="' + (x + w/2) + '" y="' + (CY + 59) + '" text-anchor="middle" font-size="10" fill="var(--ink)">' + t.replace(/ /g, '␣') + '</text>';
        x += w + 5;
      });
      g += '<rect x="' + x + '" y="' + (CY + 42) + '" width="62" height="26" rx="5" fill="var(--new)" fill-opacity=".22" stroke="var(--new)" stroke-width="1.1" stroke-dasharray="3 2"/>';
      g += '<text x="' + (x + 31) + '" y="' + (CY + 59) + '" text-anchor="middle" font-size="10" fill="var(--new)">? ? ?</text>';
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 26) + '" font-size="10.5" fill="var(--dim)">Text becomes integers. Nothing else about it survives.</text>';
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 96) + '" font-size="10.5" fill="var(--faint)">Note the split: "corpuscle" is three tokens. The model never sees words.</text>';
    } else if (st === 1){
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 24) + '" font-size="10.5" fill="var(--dim)">The stack runs once. Out comes a probability for every one of 50,257 tokens.</text>';
      CANDS.forEach((c, i) => {
        const y = CY + 42 + i * 21;
        g += '<text x="' + (X0 + 22) + '" y="' + (y + 9) + '" font-size="10" fill="' + (i === 3 ? 'var(--new)' : 'var(--dim)') + '">' + c[0] + '</text>';
        g += '<rect x="' + (X0 + 110) + '" y="' + y + '" width="' + (360 * c[1]).toFixed(0) + '" height="12" rx="3" fill="' +
          (i === 3 ? 'var(--new)' : 'var(--pro)') + '" fill-opacity="' + (i === 3 ? .9 : .45) + '"/>';
        g += '<text x="' + (X0 + 480) + '" y="' + (y + 10) + '" font-size="9.5" fill="var(--faint)">' + (100*c[1]).toFixed(0) + '%</text>';
        if (i === 3) g += '<text x="' + (X0 + 520) + '" y="' + (y + 10) + '" font-size="9.5" fill="var(--new)">← the true next token</text>';
      });
    } else if (st === 2){
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 30) + '" font-size="13" fill="var(--ink)" font-family="var(--mono)">L  =  − log( 0.02 )  =  3.91</text>';
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 62) + '" font-size="10.5" fill="var(--dim)">One number. It does not know what a nephron is; it knows the model was surprised.</text>';
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 92) + '" font-size="10.5" fill="var(--dim)">Every position in every sentence contributes one of these, and they are averaged.</text>';
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 124) + '" font-size="10.5" fill="var(--obs)">This is the only channel through which your corpus can reach the weights.</text>';
    } else if (st === 3){
      const names = ['W_Q','W_K','W_V','W_O','W_up','W_down'];
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 24) + '" font-size="10.5" fill="var(--dim)">The chain rule hands every matrix its own share of the blame, in one sweep.</text>';
      names.forEach((nm, i) => {
        const y = CY + 40 + i * 18;
        g += '<text x="' + (X0 + 22) + '" y="' + (y + 9) + '" font-size="9.5" fill="' + (i === 5 ? 'var(--pro)' : 'var(--faint)') + '">' + nm + '</text>';
        for (let j = 0; j < 10; j++)
          g += '<rect x="' + (X0 + 92 + j*34) + '" y="' + y + '" width="30" height="11" rx="2" fill="' +
            (i === 5 ? 'var(--pro)' : 'var(--cold)') + '" fill-opacity="' + (0.12 + 0.72*grad[i][j]).toFixed(2) + '"/>';
      });
      g += '<text x="' + (X0 + 452) + '" y="' + (CY + 136) + '" font-size="9.5" fill="var(--faint)">shade ∝ |∂L/∂W| · every matrix moves</text>';
    } else {
      g += '<text x="' + (X0 + 18) + '" y="' + (CY + 24) + '" font-size="10.5" fill="var(--dim)">Each number takes one small step downhill. Repeat a few million times.</text>';
      for (let i = 0; i < 6; i++) for (let j = 0; j < 22; j++){
        const v = grad[i % 6][j % 10];
        g += '<rect x="' + (X0 + 20 + j*29) + '" y="' + (CY + 42 + i*16) + '" width="25" height="13" rx="2" fill="var(--svgCell)" fill-opacity=".5"/>';
        g += '<rect x="' + (X0 + 20 + j*29) + '" y="' + (CY + 42 + i*16) + '" width="' + (25*v).toFixed(1) + '" height="13" rx="2" fill="var(--ok)" fill-opacity=".8"/>';
      }
      g += '<text x="' + (X0 + 20) + '" y="' + (CY + 142) + '" font-size="9.5" fill="var(--ok)">green = the part of each weight that just changed · it is tiny, and there are 124 million of them</text>';
    }
    host.innerHTML = svg(700, 240, g, 8);
    $('#pipeNote').innerHTML = '<b>' + STAGES[st].t + '</b> — ' + STAGES[st].s + '. ' + [
      'Fine-tuning on specialist text starts here and nowhere else: your knowledge enters as token sequences, and the model has no other way to receive it.',
      'Nothing in this step is aware of your subject matter. The same arithmetic runs whether the sentence is about nephrons or football.',
      'This is the bottleneck worth staring at. A fact becomes a training signal only insofar as it changes how surprising some token was.',
      'Note that the gradient arrives at <em>every</em> matrix simultaneously. Nobody decides that your new knowledge should live in W_down — it lands wherever reduces the loss fastest.',
      'And this is why forgetting happens: the step that makes your corpus likely is free to move weights that other, unrelated capabilities depended on. Nothing in the objective forbids it.'
    ][st];
  }
  $('#pipeNext').addEventListener('click', () => { st = (st + 1) % 5; draw(); });
  $('#pipePlay').addEventListener('click', () => {
    if (timer){ clearInterval(timer); timer = null; $('#pipePlay').textContent = '▶ walk it'; return; }
    if (RM){ st = 4; draw(); return; }
    st = 0; draw(); $('#pipePlay').textContent = '■ pause';
    timer = setInterval(() => { st++; if (st > 4){ clearInterval(timer); timer = null;
      st = 4; $('#pipePlay').textContent = '▶ walk it'; } draw(); }, 2100);
  });
  themeHooks.push(draw); draw();
})();

/* ══════════ primer 4 · a weight update as a surface, built rank by rank ══════════ */
(function(){
  const cv = $('#loraCanvas'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const N = 40, RMAX = 24, W = 1000, H = 420;
  /* an orthonormal cosine (DCT) basis: distinct frequencies are exactly orthogonal,
     so the rank-r truncation below is a true SVD truncation, not an approximation */
  const basis = f => { const v = new Float64Array(N + 1), s = Math.sqrt(2 / N);
    for (let x = 0; x <= N; x++) v[x] = s * Math.cos(Math.PI * f * (x + 0.5) / N); return v; };
  const U = [], V = [], SIG = [];
  const perm = [2,1,5,3,9,4,12,7,16,6,20,10,24,8,28,13,31,11,35,15,38,18,21,26];
  for (let i = 0; i < RMAX; i++){
    U.push(basis(i + 1)); V.push(basis(perm[i]));
    SIG.push(Math.exp(-i / 5.2) * (1 + 0.25 * Math.cos(i * 1.7)));
  }
  const TOTE = SIG.reduce((a, s) => a + s*s, 0);
  let mode = 'sum';
  function toRGB(c){
    c = c.trim();
    if (c[0] === '#'){ const h = c.length === 4 ? c.replace(/#(.)(.)(.)/, '#$1$1$2$2$3$3') : c;
      return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]; }
    const m = c.match(/\d+/g); return m ? [+m[0], +m[1], +m[2]] : [120,120,120];
  }
  function draw(){
    const r = +$('#loraR').value, tilt = +$('#loraTilt').value;
    const cs = getComputedStyle(document.documentElement), gv = n => cs.getPropertyValue(n);
    const POS = toRGB(gv('--pro') || '#5ea2f7'), NEG = toRGB(gv('--obs') || '#f5a524');
    const MESH = gv('--line2') || '#26344f', BG = gv('--inset') || '#080d16';
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const lo = mode === 'sum' ? 0 : r - 1, hi = r;
    const Hgt = Array.from({length:N+1}, () => new Float64Array(N+1));
    let peak = 1e-9;
    for (let i = lo; i < hi; i++){ const s = SIG[i], u = U[i], v = V[i];
      for (let x = 0; x <= N; x++){ const su = s * u[x];
        for (let y = 0; y <= N; y++) Hgt[x][y] += su * v[y]; } }
    for (let x = 0; x <= N; x++) for (let y = 0; y <= N; y++) peak = Math.max(peak, Math.abs(Hgt[x][y]));
    const A = 7.0, B = A * Math.sin(tilt * Math.PI / 180) * 1.42, HS = 74 / peak;
    const cx = W / 2, cy = 96;
    const px = (x, y, h) => [cx + (x - y) * A, cy + (x + y) * B - h * HS];
    const cells = [];
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) cells.push([x, y]);
    cells.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
    ctx.lineWidth = 0.45; ctx.strokeStyle = MESH; ctx.globalAlpha = 1;
    for (const c of cells){
      const x = c[0], y = c[1];
      const h00 = Hgt[x][y], h10 = Hgt[x+1][y], h11 = Hgt[x+1][y+1], h01 = Hgt[x][y+1];
      const p0 = px(x, y, h00), p1 = px(x+1, y, h10), p2 = px(x+1, y+1, h11), p3 = px(x, y+1, h01);
      const hm = (h00 + h10 + h11 + h01) / 4, t = Math.min(1, Math.abs(hm) / peak);
      const col = hm >= 0 ? POS : NEG;
      ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]); ctx.lineTo(p3[0], p3[1]); ctx.closePath();
      ctx.fillStyle = 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (0.14 + 0.78 * t).toFixed(3) + ')';
      ctx.fill(); ctx.stroke();
    }
    ctx.font = '600 11px ui-monospace, monospace';
    ctx.fillStyle = gv('--faint') || '#55648a';
    ctx.textAlign = 'left';
    ctx.fillText('rows of ΔW  →', cx + N * A + 12, cy + N * B + 4);
    ctx.textAlign = 'right';
    ctx.fillText('←  columns of ΔW', cx - N * A - 12, cy + N * B + 4);
    ctx.textAlign = 'left';
    ctx.fillStyle = gv('--ink') || '#e3eaf8';
    ctx.font = '700 12px ui-monospace, monospace';
    ctx.fillText(mode === 'sum' ? 'ΔW  =  Σ σᵢ uᵢvᵢᵀ   for i < ' + r : 'the single ridge  σ' + r + ' u' + r + 'v' + r + 'ᵀ', 24, 26);
    const cap = SIG.slice(0, r).reduce((a, s) => a + s*s, 0) / TOTE;
    const pr = r * (768 + 3072);
    $('#loraRV').textContent = r;
    $('#loraTiltV').textContent = tilt + '°';
    $('#loraP').textContent = pr.toLocaleString('en-US');
    $('#loraPct').textContent = (100 * pr / 2359296).toFixed(2) + '% of the matrix';
    $('#loraFull').textContent = '2,359,296';
    $('#loraCap').textContent = (100 * cap).toFixed(1) + '%';
    $('#loraMem').textContent = (pr * 2 / 1024).toFixed(0) + ' KB';
    $('#loraNote').innerHTML = mode === 'sum'
      ? 'At rank <b class="pp">' + r + '</b> the adapter stores <b>' + pr.toLocaleString('en-US') + '</b> numbers instead of 2,359,296 — ' +
        (100 * pr / 2359296).toFixed(2) + '% — and reproduces <b class="gg">' + (100*cap).toFixed(1) + '%</b> of this target surface. ' +
        'Notice what low rank actually forbids: the surface can only be built from full-width ridges. It can never put a bump ' +
        'in one corner and leave the rest flat, because every rank-one term spans the entire matrix. ' +
        (r <= 2 ? 'At this rank the surface is visibly just a few corrugations.' :
         r >= 18 ? 'By this rank the extra ridges are adding detail you can barely see — which is the empirical argument for LoRA.' :
         'Each extra ridge adds finer corrugation and costs 3,840 more numbers.')
      : 'This is ridge <b>' + r + '</b> on its own: one outer product <span class="mono">σ u vᵀ</span>, with singular value <b>' +
        SIG[r-1].toFixed(3) + '</b>. Its value at row i, column j is just <span class="mono">σ·u[i]·v[j]</span> — a single ' +
        'product of two one-dimensional patterns. Switch back to <b>accumulated</b> to see it added to the others.';
  }
  ['#loraR','#loraTilt'].forEach(id => $(id).addEventListener('input', draw));
  [['#loraSum','sum'],['#loraOne','one']].forEach(e => $(e[0]).addEventListener('click', () => {
    mode = e[1]; $('#loraSum').classList.toggle('on', mode === 'sum');
    $('#loraOne').classList.toggle('on', mode === 'one'); draw(); }));
  themeHooks.push(draw); window.addEventListener('resize', draw); draw();
})();

/* ══════════ primer 5 · what MEMIT actually does ══════════ */
(function(){
  const host = $('#memitSvg'); if (!host) return;
  const ST = [
    {t:'1 · LOCATE', s:'causal tracing'},
    {t:'2 · TARGET', s:'optimise an activation'},
    {t:'3 · SOLVE',  s:'closed form, one matrix'},
    {t:'4 · SPREAD', s:'across the layer range'}
  ];
  let st = 0;
  function draw(){
    const X0 = 18, W = 664, bw = W / 4;
    let g = '';
    ST.forEach((S, i) => {
      const x = X0 + i * bw, on = i === st;
      g += '<rect x="' + (x + 4) + '" y="14" width="' + (bw - 10) + '" height="34" rx="6" fill="' +
        (on ? 'var(--new)' : 'var(--svgBox)') + '" fill-opacity="' + (on ? .9 : 1) + '" stroke="' +
        (on ? 'var(--new)' : 'var(--line2)') + '"/>';
      g += '<text x="' + (x + bw/2) + '" y="' + 31 + '" text-anchor="middle" font-size="10.5" font-weight="700" fill="' +
        (on ? 'var(--ink)' : 'var(--dim)') + '">' + S.t + '</text>';
      g += '<text x="' + (x + bw/2) + '" y="' + 43 + '" text-anchor="middle" font-size="8.5" fill="' +
        (on ? 'var(--ink)' : 'var(--faint)') + '">' + S.s + '</text>';
    });
    const CY = 66, HH = 146;
    g += '<rect x="' + X0 + '" y="' + CY + '" width="' + W + '" height="' + HH + '" rx="8" fill="var(--inset)" stroke="var(--line)"/>';
    if (st === 0){
      for (let i = 0; i < 12; i++){
        const y = CY + 18 + i * 9, crit = i >= 2 && i <= 7;
        g += '<rect x="' + (X0 + 24) + '" y="' + y + '" width="150" height="7" rx="2" fill="' +
          (crit ? 'var(--pro)' : 'var(--svgCell)') + '" fill-opacity="' + (crit ? .85 : .4) + '"/>';
      }
      g += '<text x="' + (X0 + 186) + '" y="' + (CY + 52) + '" font-size="10.5" fill="var(--pro)">layers 3–8 mediate the fact</text>';
      g += '<text x="' + (X0 + 186) + '" y="' + (CY + 74) + '" font-size="10" fill="var(--dim)">Corrupt the subject tokens, then restore one</text>';
      g += '<text x="' + (X0 + 186) + '" y="' + (CY + 90) + '" font-size="10" fill="var(--dim)">layer&rsquo;s activations at a time. The layers that</text>';
      g += '<text x="' + (X0 + 186) + '" y="' + (CY + 106) + '" font-size="10" fill="var(--dim)">restore the right answer are the ones that matter.</text>';
    } else if (st === 1){
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 30) + '" font-size="11" font-family="var(--mono)" fill="var(--ink)">minimise over z:   − log p( &ldquo;Seattle&rdquo; | &ldquo;The Space Needle is in&rdquo; )</text>';
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 54) + '" font-size="10.5" fill="var(--dim)">where z replaces the MLP output at the subject&rsquo;s last token.</text>';
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 84) + '" font-size="10.5" fill="var(--obs)">This is the only gradient descent in the method — and it optimises</text>';
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 100) + '" font-size="10.5" fill="var(--obs)">a 768-number activation, never a weight. Seconds, not hours.</text>';
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 126) + '" font-size="10" fill="var(--faint)">Output: z, the value this key ought to retrieve.</text>';
    } else if (st === 2){
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 30) + '" font-size="12.5" font-family="var(--mono)" fill="var(--ink)">Δ  =  R K₁ᵀ ( K₁K₁ᵀ + λ K₀K₀ᵀ )⁻¹</text>';
      const rows = [['R  = z − W k₁', 'the residual: what you want minus what you get', 'var(--new)'],
                    ['K₁ = the new keys', 'a handful of columns', 'var(--new)'],
                    ['K₀ = keys from ordinary text', 'the knowledge to be preserved', 'var(--pro)']];
      rows.forEach((r, i) => {
        const y = CY + 58 + i * 26;
        g += '<text x="' + (X0 + 24) + '" y="' + y + '" font-size="10.5" font-family="var(--mono)" fill="' + r[2] + '">' + r[0] + '</text>';
        g += '<text x="' + (X0 + 238) + '" y="' + y + '" font-size="10" fill="var(--dim)">' + r[1] + '</text>';
      });
      g += '<text x="' + (X0 + 24) + '" y="' + (CY + 138) + '" font-size="9.5" fill="var(--faint)">No iteration. One linear solve, and the layer now answers differently.</text>';
    } else {
      const share = [1, 1/2, 1/3, 1/4, 1/5, 1/6], tot = share.reduce((a,b) => a+b, 0);
      share.forEach((v, i) => {
        const y = CY + 20 + i * 20, w = 300 * v / share[0];
        g += '<text x="' + (X0 + 24) + '" y="' + (y + 10) + '" font-size="9.5" fill="var(--faint)">layer ' + (8 - i) + '</text>';
        g += '<rect x="' + (X0 + 80) + '" y="' + y + '" width="' + w.toFixed(0) + '" height="12" rx="3" fill="var(--pro)" fill-opacity=".8"/>';
        g += '<text x="' + (X0 + 88 + w) + '" y="' + (y + 10) + '" font-size="9.5" fill="var(--dim)">' + (100*v/tot).toFixed(0) + '%</text>';
      });
      g += '<text x="' + (X0 + 430) + '" y="' + (CY + 42) + '" font-size="10.5" fill="var(--dim)">The residual is split across the range</text>';
      g += '<text x="' + (X0 + 430) + '" y="' + (CY + 58) + '" font-size="10.5" fill="var(--dim)">with the coefficient 1/(L−ℓ+1).</text>';
      g += '<text x="' + (X0 + 430) + '" y="' + (CY + 84) + '" font-size="10.5" fill="var(--obs)">Section 08 is about how little</text>';
      g += '<text x="' + (X0 + 430) + '" y="' + (CY + 100) + '" font-size="10.5" fill="var(--obs)">justification that rule has.</text>';
    }
    host.innerHTML = svg(700, 226, g, 8);
    $('#memitNote').innerHTML = [
      'Knowledge is not filed anywhere obvious, so the first job is finding which layers carry it. Causal tracing is an intervention experiment on the model itself, not an inspection of the weights.',
      'This is the step that surprises people: the method never computes a gradient of the loss with respect to a weight. It asks what the layer <em>should have said</em>, then solves for a matrix that says it.',
      'Here is the equation the whole page argues about. Everything to the right of R is a choice about which of the many matrices that install the fact you actually take — see section 06.',
      'Having solved for a change, MEMIT hands out fractions of it to each layer in the range. That coefficient is the weakest link in the method and the subject of section 08.'
    ][st];
  }
  $('#memitNext').addEventListener('click', () => { st = (st + 1) % 4; draw(); });
  $('#memitReset').addEventListener('click', () => { st = 0; draw(); });
  themeHooks.push(draw); draw();
})();

/* ══════════ primer 6 · the same matrix judged from both ends ══════════ */
(function(){
  const host = $('#sideSvg'); if (!host) return;
  const NC = 26, NR = 12, rnd = mulberry(4242);
  const colE = Array.from({length:NC}, (_, i) => Math.pow(i + 1, -0.9));
  const rowS = Array.from({length:NR}, () => 0.2 + 0.8 * rnd());
  rowS[9] = 0.98; rowS[3] = 0.9;
  const mxC = colE[0], mxR = Math.max.apply(null, rowS);
  let side = 'in';
  function draw(){
    const GX = 118, GY = 74, cw = 19, ch = 13, gp = 2;
    let g = '';
    g += '<text x="' + GX + '" y="26" font-size="9.5" letter-spacing="1.2" fill="var(--pro)">INPUT SIDE · how much energy the keys put in each column direction</text>';
    for (let c = 0; c < NC; c++){
      const h = 28 * colE[c] / mxC;
      g += '<rect x="' + (GX + c*(cw+gp)) + '" y="' + (62 - h) + '" width="' + cw + '" height="' + Math.max(1.5,h).toFixed(1) +
        '" rx="2" fill="var(--pro)" fill-opacity=".8"/>';
    }
    g += '<text x="8" y="' + (GY - 8) + '" font-size="9.5" letter-spacing="1.2" fill="var(--obs)">OUTPUT SIDE</text>';
    for (let r = 0; r < NR; r++){
      const w = 66 * rowS[r] / mxR;
      g += '<rect x="' + (104 - w) + '" y="' + (GY + r*(ch+gp)) + '" width="' + Math.max(1.5,w).toFixed(1) + '" height="' + ch +
        '" rx="2" fill="var(--obs)" fill-opacity=".8"/>';
    }
    for (let r = 0; r < NR; r++) for (let c = 0; c < NC; c++){
      const a = colE[c] / mxC, b = rowS[r] / mxR;
      const v = side === 'in' ? a : side === 'out' ? b : Math.sqrt(a * b);
      const col = side === 'in' ? 'var(--pro)' : side === 'out' ? 'var(--obs)' : 'var(--ok)';
      g += '<rect x="' + (GX + c*(cw+gp)) + '" y="' + (GY + r*(ch+gp)) + '" width="' + cw + '" height="' + ch +
        '" rx="2" fill="' + col + '" fill-opacity="' + (0.06 + 0.84 * v).toFixed(3) + '" stroke="var(--line)" stroke-width="0.5"/>';
    }
    const gw = NC*(cw+gp), gh = NR*(ch+gp);
    g += '<text x="' + (GX + gw + 10) + '" y="' + (GY + 12) + '" font-size="11" font-weight="700" fill="var(--ink)">W</text>';
    g += '<text x="' + (GX + gw + 10) + '" y="' + (GY + 26) + '" font-size="9" fill="var(--faint)">768 × 3072</text>';
    g += '<text x="' + GX + '" y="' + (GY + gh + 18) + '" font-size="9.5" fill="var(--faint)">columns ↔ key-space directions (3072 of them)</text>';
    g += '<text x="8" y="' + (GY + gh + 18) + '" font-size="9.5" fill="var(--faint)">rows ↔ value space</text>';
    host.innerHTML = svg(700, 250, g, 8);
    $('#sideNote').innerHTML = side === 'in'
      ? 'The <span class="pp">input side</span> shades <b>columns</b>. It says: these key directions carry the traffic, so a change to the ' +
        'matrix in those columns will be felt. It is a statistic about the data, computed with no reference to what the network does next — ' +
        'and it is the only criterion almost every published method uses.'
      : side === 'out'
        ? 'The <span class="oo">output side</span> shades <b>rows</b>, and it is a completely different cut through the same matrix. It says: ' +
          'the rest of the network is sensitive to <em>these</em> output directions, so a change that lands in those rows will propagate, ' +
          'however rarely it is triggered. Note rows 4 and 10: loud on the output, and the input criterion has nothing to say about them.'
        : 'Weighting both gives <span class="gg">√(input × output)</span> — the Hankel-style criterion from section 04. A cell is genuinely ' +
          'safe only when it is dark in <em>both</em> profiles: rarely excited <b>and</b> barely observed. Rows that are loud downstream stay ' +
          'protected even where the columns are quiet, which is exactly the case a one-sided rule gets wrong.';
  }
  [['#sideIn','in'],['#sideOut','out'],['#sideBoth','both']].forEach(e => $(e[0]).addEventListener('click', () => {
    side = e[1];
    ['#sideIn','#sideOut','#sideBoth'].forEach((id, i) => $(id).classList.toggle('on', ['in','out','both'][i] === side));
    draw(); }));
  themeHooks.push(draw); draw();
})();

/* ══════════ primer 7 · the freedom, to scale ══════════ */
(function(){
  const host = $('#freedomSvg'); if (!host) return;
  function draw(){
    const TOT = 768 * 3072, CON = 10 * 768, X0 = 20, W = 660, Y = 42, H = 34;
    const cw = Math.max(1.2, W * CON / TOT);
    let g = '<rect x="' + X0 + '" y="' + Y + '" width="' + W + '" height="' + H + '" rx="5" fill="var(--free)" fill-opacity=".55"/>';
    g += '<rect x="' + X0 + '" y="' + Y + '" width="' + cw.toFixed(2) + '" height="' + H + '" fill="var(--hot)"/>';
    g += '<text x="' + X0 + '" y="' + (Y - 12) + '" font-size="10" fill="var(--hot)">7,680 equations</text>';
    g += '<text x="' + (X0 + W) + '" y="' + (Y - 12) + '" text-anchor="end" font-size="10" fill="var(--free)">2,351,616 directions left free  ·  99.67%</text>';
    g += '<text x="' + (X0 + W/2) + '" y="' + (Y + 22) + '" text-anchor="middle" font-size="11" font-weight="700" fill="var(--ink)">2,359,296 unknowns in one W_down</text>';
    /* the sliver, magnified 60× so it is visible at all */
    const MY = Y + H + 26;
    g += '<path d="M' + X0 + ',' + (Y + H) + ' L' + X0 + ',' + MY + ' M' + (X0 + cw) + ',' + (Y + H) + ' L' + (X0 + 210) + ',' + MY +
      '" stroke="var(--hot)" stroke-width="0.9" stroke-dasharray="3 3"/>';
    g += '<rect x="' + X0 + '" y="' + MY + '" width="210" height="22" rx="4" fill="var(--hot)" fill-opacity=".8"/>';
    g += '<text x="' + (X0 + 105) + '" y="' + (MY + 15) + '" text-anchor="middle" font-size="9.5" font-weight="700" fill="var(--bg)">the constrained part, magnified 60×</text>';
    g += '<text x="' + (X0 + 224) + '" y="' + (MY + 9) + '" font-size="9.5" fill="var(--dim)">2 new facts × 768 outputs  =  1,536</text>';
    g += '<text x="' + (X0 + 224) + '" y="' + (MY + 22) + '" font-size="9.5" fill="var(--dim)">8 protected keys × 768 outputs  =  6,144</text>';
    host.innerHTML = svg(700, 116, g, 8);
  }
  themeHooks.push(draw); draw();
})();

/* ══════════ 06b · the measured layers ══════════ */
const MEASURED = {"small":{"name":"GPT-2 small","params":"124M","dim":3072,"nlayer":12,"effRank":27.5,"top":18.1,"spec":{"idx":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,26,27,29,30,32,34,36,38,40,42,45,47,50,53,56,59,62,66,70,74,78,82,87,91,97,102,108,114,120,127,134,141,149,158,166,176,185,196,207,218,230,243,256,271,286,302,318,336,355,374,395,417,440,465,491,518,547,577,609,643,678,716,756,798,842,888,938,990,1044,1102,1163,1228,1296,1368,1444,1524,1608,1697,1791,1890,1995,2106,2222,2345,2475,2612,2757,2910,3071],"lam":[19.4999,3.34528,2.30366,1.64037,1.59601,1.32845,1.13227,1.05769,0.999942,0.925133,0.875947,0.784104,0.722254,0.709657,0.663192,0.646091,0.624601,0.591563,0.572374,0.546821,0.495316,0.462957,0.45648,0.447519,0.423714,0.402975,0.381204,0.363127,0.346904,0.326765,0.311913,0.302895,0.292884,0.281152,0.260907,0.251449,0.239866,0.228364,0.221146,0.211447,0.201815,0.193857,0.181913,0.172806,0.164324,0.157625,0.150888,0.144646,0.13568,0.128028,0.121912,0.115483,0.110947,0.104972,0.100401,0.0956787,0.0909402,0.0861834,0.0825944,0.078763,0.0752053,0.0713985,0.0677915,0.0645788,0.061619,0.0586374,0.0556089,0.0529532,0.0502477,0.047645,0.0458854,0.043579,0.0413595,0.0393824,0.0374045,0.0355608,0.0338292,0.0321625,0.0306137,0.029036,0.0275672,0.0262206,0.0248585,0.02358,0.0223314,0.0211388,0.0200075,0.0189473,0.0179645,0.0170322,0.0160956,0.0151986,0.0143744,0.0135616,0.0127806,0.0120577,0.0113625,0.0106734,0.0100219,0.00941011,0.0088105,0.00822886,0.00767293,0.00713972,0.0066195,0.0061063,0.00560226,0.00510099,0.00459961,0.00407586,0.00346938,0.00256858,0.000187234],"cum":[0.18144,0.21256,0.234,0.24926,0.26411,0.27647,0.287,0.29685,0.30615,0.31476,0.32291,0.3302,0.33692,0.34353,0.3497,0.35571,0.36152,0.36702,0.37235,0.37744,0.38686,0.39117,0.39541,0.39958,0.40753,0.41128,0.41842,0.4218,0.42829,0.43445,0.44036,0.44605,0.45155,0.45685,0.46434,0.46903,0.47584,0.48234,0.48858,0.49453,0.50023,0.50759,0.51453,0.5211,0.52733,0.53329,0.54045,0.54589,0.55363,0.55975,0.56672,0.57329,0.57959,0.58657,0.59324,0.5996,0.60652,0.61393,0.6202,0.62766,0.63407,0.64155,0.64863,0.65539,0.6624,0.66963,0.67653,0.68411,0.69132,0.6986,0.70556,0.71304,0.72052,0.72765,0.73515,0.74261,0.75002,0.75768,0.76526,0.77274,0.78036,0.78785,0.79543,0.80307,0.81053,0.81821,0.82587,0.83347,0.84102,0.8485,0.85619,0.86376,0.87118,0.87871,0.88618,0.89369,0.90108,0.90845,0.91576,0.92299,0.9301,0.93715,0.9441,0.95091,0.95763,0.96419,0.9705,0.97662,0.98248,0.98801,0.99311,0.99745,1.0]},"tokens":93184,"held":1728,"rows":[{"layer":3,"Amax":3.1,"Arms":3.98,"Bmax":3.7,"Brms":9.57},{"layer":6,"Amax":3.2,"Arms":6.92,"Bmax":3.55,"Brms":11.39},{"layer":9,"Amax":1.72,"Arms":4.93,"Bmax":1.73,"Brms":7.88}]},"xl":{"name":"GPT-2 XL","params":"1.5B","dim":6400,"nlayer":48,"effRank":66.9,"top":11.8,"spec":{"idx":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,26,28,29,31,33,35,37,40,42,45,48,50,54,57,60,64,68,72,77,81,86,92,97,103,110,116,123,131,139,147,156,166,176,187,198,210,223,237,251,266,282,300,318,337,357,379,402,427,453,480,509,540,573,608,645,684,725,769,816,865,918,973,1032,1095,1161,1232,1307,1386,1470,1559,1653,1754,1860,1973,2092,2219,2354,2496,2648,2808,2978,3159,3350,3553,3768,3997,4239,4496,4768,5057,5364,5689,6033,6399],"lam":[20.7934,2.941,1.63367,1.25006,1.02612,0.899039,0.856638,0.743194,0.664974,0.650233,0.613675,0.567296,0.55025,0.526513,0.498979,0.486337,0.450376,0.442518,0.42954,0.420234,0.414263,0.397985,0.379764,0.376725,0.361976,0.351256,0.347126,0.331083,0.313619,0.30497,0.291277,0.276819,0.266842,0.252513,0.242281,0.237433,0.224168,0.217909,0.208525,0.202274,0.19327,0.186311,0.178269,0.169922,0.163905,0.157032,0.151234,0.144336,0.137735,0.134633,0.12917,0.124427,0.118965,0.115485,0.110406,0.105575,0.101513,0.0979394,0.0942213,0.0906225,0.0872811,0.0839643,0.0809079,0.0778628,0.0748783,0.0718901,0.0692922,0.0667966,0.0643305,0.061871,0.0596851,0.0571327,0.0552223,0.0530601,0.0511218,0.0491295,0.0472633,0.0454518,0.0436735,0.0419746,0.0403672,0.0388591,0.037273,0.035867,0.0344664,0.0331201,0.0318297,0.0305264,0.0292829,0.0281123,0.0269628,0.025814,0.0247532,0.0237208,0.0226861,0.0216928,0.020732,0.0197953,0.0189011,0.0180066,0.017137,0.0162978,0.0154636,0.0146529,0.0138628,0.0130817,0.0123091,0.0115487,0.0107908,0.0100363,0.0092857,0.00853261,0.00776721,0.00697865,0.00614189,0.00520994,0.00401405,0.000290532],"cum":[0.11832,0.13506,0.14435,0.15147,0.15731,0.16242,0.1673,0.17153,0.17531,0.17901,0.1825,0.18573,0.18886,0.19186,0.1947,0.19746,0.20003,0.20254,0.20499,0.20738,0.20974,0.21434,0.2165,0.21864,0.2228,0.22681,0.22879,0.23261,0.23623,0.23973,0.24308,0.2479,0.25096,0.25531,0.2595,0.26222,0.26745,0.2712,0.27483,0.27949,0.28398,0.28827,0.29343,0.29737,0.30211,0.30758,0.31194,0.31697,0.32258,0.32724,0.33247,0.33823,0.34375,0.34906,0.35481,0.36094,0.36681,0.37303,0.37903,0.38532,0.39189,0.39869,0.40524,0.412,0.41895,0.42646,0.43368,0.44103,0.44848,0.45638,0.46432,0.47262,0.48092,0.48923,0.49781,0.50665,0.51569,0.52491,0.53429,0.54378,0.55339,0.5633,0.57348,0.58367,0.59427,0.60483,0.61572,0.62689,0.63811,0.6497,0.66145,0.6733,0.68538,0.69765,0.71005,0.72279,0.73558,0.74861,0.7617,0.77503,0.78852,0.80202,0.81575,0.82946,0.84324,0.85711,0.8709,0.88467,0.89832,0.91189,0.92519,0.93821,0.95082,0.96294,0.9744,0.9849,0.99399,1.0]},"tokens":172032,"held":3168,"rows":[{"layer":14,"Amax":2.28,"Arms":5.9,"Bmax":3.11,"Brms":11.84},{"layer":17,"Amax":3.71,"Arms":3.38,"Bmax":3.87,"Brms":6.56},{"layer":36,"Amax":3.67,"Arms":7.28,"Bmax":5.77,"Brms":11.92}]}};
(function(){
  const host = $('#mSpec'); if (!host) return;
  let mode = 'rms', key = 'xl';
  function spec(M){
    const S = M.spec, X0 = 48, X1 = 322, Y0 = 16, Y1 = 178;
    const lx = i => Math.log10(i + 1), ly = v => Math.log10(v);
    const xmax = lx(M.dim), ymax = ly(S.lam[0]), ymin = ly(S.lam[S.lam.length - 1]);
    const sx = i => X0 + lx(i) / xmax * (X1 - X0);
    const sy = v => Y1 - (ly(v) - ymin) / (ymax - ymin) * (Y1 - Y0);
    const sup = e => (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');
    let g = '';
    for (let e = 0; e <= 4; e++){
      const x = sx(Math.pow(10, e) - 1); if (x > X1 + 1) continue;
      g += '<line x1="' + x.toFixed(1) + '" y1="' + Y0 + '" x2="' + x.toFixed(1) + '" y2="' + Y1 + '" stroke="var(--svgGrid)"/>';
      g += '<text x="' + x.toFixed(1) + '" y="' + (Y1 + 14) + '" text-anchor="middle" font-size="8.5" fill="var(--faint)">' +
        (e === 0 ? '1' : '10' + sup(e)) + '</text>';
    }
    for (let e = 2; e >= -5; e--){
      const y = sy(Math.pow(10, e)); if (y < Y0 || y > Y1) continue;
      g += '<line x1="' + X0 + '" y1="' + y.toFixed(1) + '" x2="' + X1 + '" y2="' + y.toFixed(1) + '" stroke="var(--svgGrid)"/>';
      g += '<text x="' + (X0 - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end" font-size="8.5" fill="var(--faint)">10' + sup(e) + '</text>';
    }
    let d = '';
    S.idx.forEach((i, k) => { d += (k ? 'L' : 'M') + sx(i).toFixed(1) + ',' + sy(S.lam[k]).toFixed(1); });
    g += '<path d="' + d + '" fill="none" stroke="var(--pro)" stroke-width="2"/>';
    const ex = sx(Math.round(M.effRank));
    g += '<line x1="' + ex.toFixed(1) + '" y1="' + Y0 + '" x2="' + ex.toFixed(1) + '" y2="' + Y1 +
      '" stroke="var(--ok)" stroke-width="1.4" stroke-dasharray="4 3"/>';
    g += '<text x="' + (ex + 6).toFixed(1) + '" y="' + (Y0 + 12) + '" font-size="9" font-weight="700" fill="var(--ok)">effective rank ' + M.effRank + '</text>';
    g += '<circle cx="' + sx(0).toFixed(1) + '" cy="' + sy(S.lam[0]).toFixed(1) + '" r="4" fill="var(--hot)"/>';
    g += '<text x="' + (sx(0) + 8).toFixed(1) + '" y="' + (sy(S.lam[0]) + 4).toFixed(1) + '" font-size="9" fill="var(--hot)">one direction holds ' + M.top + '%</text>';
    g += '<text x="' + X0 + '" y="' + (Y1 + 30) + '" font-size="9" fill="var(--faint)">direction, ranked by energy  →  (of ' + M.dim + ')</text>';
    host.innerHTML = svg(340, 200, g, 8);
  }
  function gaps(M){
    const R = M.rows, X0 = 44, X1 = 320, Y0 = 18, Y1 = 176;
    const k = mode === 'rms' ? ['Arms','Brms'] : ['Amax','Bmax'];
    const top = Math.max.apply(null, R.map(r => Math.max(r[k[0]], r[k[1]]))) * 1.16;
    const gw = (X1 - X0) / R.length, step = top > 8 ? 4 : top > 4 ? 2 : 1;
    let g = '';
    for (let v = step; v <= top; v += step){
      const y = Y1 - v / top * (Y1 - Y0);
      g += '<line x1="' + X0 + '" y1="' + y.toFixed(1) + '" x2="' + X1 + '" y2="' + y.toFixed(1) + '" stroke="var(--svgGrid)"/>';
      g += '<text x="' + (X0 - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end" font-size="8.5" fill="var(--faint)">' + v + '×</text>';
    }
    g += '<line x1="' + X0 + '" y1="' + (Y1 - 1/top*(Y1-Y0)).toFixed(1) + '" x2="' + X1 + '" y2="' + (Y1 - 1/top*(Y1-Y0)).toFixed(1) +
      '" stroke="var(--faint)" stroke-width="1" stroke-dasharray="3 3"/>';
    R.forEach((r, i) => {
      const cx = X0 + i * gw, bw = gw * 0.3;
      [[r[k[0]], 'var(--obs)', 0.14], [r[k[1]], 'var(--hot)', 0.54]].forEach(b => {
        const h = Math.max(0, b[0]) / top * (Y1 - Y0);
        g += '<rect x="' + (cx + gw*b[2]).toFixed(1) + '" y="' + (Y1 - h).toFixed(1) + '" width="' + bw.toFixed(1) +
          '" height="' + h.toFixed(1) + '" rx="3" fill="' + b[1] + '" fill-opacity=".85"/>';
        g += '<text x="' + (cx + gw*b[2] + bw/2).toFixed(1) + '" y="' + (Y1 - h - 5).toFixed(1) +
          '" text-anchor="middle" font-size="9" font-weight="700" fill="' + b[1] + '">' + b[0].toFixed(1) + '</text>';
      });
      g += '<text x="' + (cx + gw/2).toFixed(1) + '" y="' + (Y1 + 15) + '" text-anchor="middle" font-size="9.5" fill="var(--ink)">L' + r.layer + '</text>';
    });
    g += '<line x1="' + X0 + '" y1="' + Y1 + '" x2="' + X1 + '" y2="' + Y1 + '" stroke="var(--line2)"/>';
    g += '<rect x="' + X0 + '" y="' + (Y0 - 4) + '" width="9" height="9" rx="2" fill="var(--obs)"/>' +
         '<text x="' + (X0 + 13) + '" y="' + (Y0 + 4) + '" font-size="9" fill="var(--dim)">exact-constrained</text>';
    g += '<rect x="' + (X0 + 128) + '" y="' + (Y0 - 4) + '" width="9" height="9" rx="2" fill="var(--hot)"/>' +
         '<text x="' + (X0 + 141) + '" y="' + (Y0 + 4) + '" font-size="9" fill="var(--dim)">MEMIT ablation</text>';
    $('#mGap').innerHTML = svg(340, 198, g, 8);
  }
  function draw(){
    const M = MEASURED[key], O = MEASURED[key === 'xl' ? 'small' : 'xl'];
    spec(M); gaps(M);
    const bs = M.rows.map(r => mode === 'rms' ? r.Brms : r.Bmax);
    const lo = Math.min.apply(null, bs), hi = Math.max.apply(null, bs);
    const os = O.rows.map(r => mode === 'rms' ? r.Brms : r.Bmax);
    $('#mTag').textContent = M.name + ' · ' + M.params + ' · mlp.c_proj · key dimension ' + M.dim +
      ' · ' + (M.tokens / 1000).toFixed(0) + 'k tokens · measured, not modelled';
    $('#msER').textContent = M.effRank;
    $('#msERd').textContent = 'of ' + M.dim + ' dimensions';
    $('#msTop').textContent = M.top + '%';
    $('#msB').textContent = lo.toFixed(1) + '–' + hi.toFixed(1) + '×';
    $('#msBd').textContent = mode === 'rms' ? 'less average leakage than identity' : 'less worst-case leakage';
    $('#mNote').innerHTML = 'Left: the spectrum falls off a cliff. In ' + M.dim + ' dimensions a single direction carries ' +
      '<b class="hh">' + M.top + '%</b> of the energy, and the participation ratio — the honest count of how many directions ' +
      'actually matter — is <b class="gg">' + M.effRank + '</b>. The synthetic κ slider above cannot reach anything this extreme.' +
      '<br><br>Right: the same experiment on those real keys, layer by layer. <span class="oo">Amber</span> is the exact-constrained ' +
      'comparison from finding 2, Euclidean against Gramian-weighted. <span class="hh">Red</span> is the blunter and more useful one: ' +
      'MEMIT exactly as published against MEMIT with its <span class="mono">K₀K₀ᵀ</span> swapped for an identity matrix. Both install ' +
      'the edit to the same accuracy; the published version leaks <b class="hh">' + lo.toFixed(1) + '–' + hi.toFixed(1) + '×</b> less. ' +
      'Switch to <b>' + O.name + '</b> (' + O.params + ', key dimension ' + O.dim + ') and the same experiment gives ' +
      Math.min.apply(null, os).toFixed(1) + '–' + Math.max.apply(null, os).toFixed(1) + '× — ' +
      (Math.abs(Math.max.apply(null, os) - hi) / Math.max(hi, 1) < 0.45
        ? 'the effect does not wash out with scale.'
        : 'the effect changes in size with scale but not in sign.');
  }
  [['#mRms','rms','mode'],['#mMax','max','mode'],['#mSmall','small','key'],['#mXL','xl','key']].forEach(e =>
    $(e[0]).addEventListener('click', () => {
      if (e[2] === 'mode') mode = e[1]; else key = e[1];
      $('#mRms').classList.toggle('on', mode === 'rms'); $('#mMax').classList.toggle('on', mode === 'max');
      $('#mSmall').classList.toggle('on', key === 'small'); $('#mXL').classList.toggle('on', key === 'xl');
      draw(); }));
  themeHooks.push(draw); draw();
})();
