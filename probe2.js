// The degeneracy above is structural: a Euclidean minimum-norm solution is orthogonal to the
// freedom direction, so any objective that also treats them as orthogonal is minimised at t=0.
// Reality is not Euclidean: the preserved-key distribution has a covariance that COUPLES the
// free direction to the constrained span. That coupling is what moves the optimum.
const k0=[1,0.3,0], k1=[0.2,1,0.4], r=[0.8,-0.5], m=[0.75,0.45,-0.2];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0), add=(a,b,s=1)=>a.map((x,i)=>x+s*b[i]);
const sc=(a,s)=>a.map(x=>x*s), nrm=a=>Math.sqrt(dot(a,a)), unit=a=>sc(a,1/nrm(a));
const A=[[dot(k0,k0),dot(k0,k1)],[dot(k1,k0),dot(k1,k1)]];
const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
const row=ri=>add(sc(k0,(-A[0][1]*ri)/det),sc(k1,(A[0][0]*ri)/det));
const d1=row(r[0]), d2=row(r[1]);
const n=unit([k0[1]*k1[2]-k0[2]*k1[1],k0[2]*k1[0]-k0[0]*k1[2],k0[0]*k1[1]-k0[1]*k1[0]]);
const v=unit(k1), a=[1,0.62];
for (const [g,q2,spread] of [[0,2,0.5],[0.7,2,0.5],[1.1,2,0.5],[1.1,4,0.5],[1.5,4,0.55],[2.0,4,0.6]]){
  let s=987654321; const rnd=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};
  const gs=()=>{let u=0,w=0;while(!u)u=rnd();while(!w)w=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*w);};
  const held=[];
  for(let j=0;j<44;j++){const x=[gs(),gs(),gs()];
    // correlated map M = I + g(n v^T + v n^T): couples the free direction to the new-fact direction
    const y=add(add(x,v,g*dot(n,x)),n,g*dot(v,x));
    held.push(unit(add(m,y,spread)));}
  const Qh=[1,Math.sqrt(q2)];
  const frag=t=>{let mx=0;for(const k of held){const c=dot(n,k);
    const o=[(dot(d1,k)+t*c*a[0])*Qh[0],(dot(d2,k)+t*c*a[1])*Qh[1]];
    const val=Math.hypot(o[0],o[1]); if(val>mx)mx=val;} return mx;};
  let bt=0,bv=1e9,wv=0;
  for(let i=-260;i<=260;i++){const t=i/200,f=frag(t);if(f<bv){bv=f;bt=t;}if(f>wv)wv=f;}
  console.log(`gamma=${g} Q=diag(1,${q2}) spread=${spread}  ->  t*=${bt.toFixed(3)}  frag(0)/frag(t*)=${(frag(0)/bv).toFixed(2)}x  range=${(wv/bv).toFixed(1)}x`);
}
