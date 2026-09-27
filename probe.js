// Does the least-fragile point on the freedom line differ from the minimum-norm point?
const k0=[1,0.3,0], k1=[0.2,1,0.4], r=[0.8,-0.5], m=[0.75,0.45,-0.2];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const add=(a,b,s=1)=>a.map((x,i)=>x+s*b[i]);
const sc=(a,s)=>a.map(x=>x*s);
const nrm=a=>Math.sqrt(dot(a,a));
const unit=a=>sc(a,1/nrm(a));
// min-norm rows: delta in span{k0,k1}, delta.k0=0, delta.k1=r_i
const A=[[dot(k0,k0),dot(k0,k1)],[dot(k1,k0),dot(k1,k1)]];
const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
function row(ri){const c1=( A[1][1]*0 - A[0][1]*ri)/det, c2=(-A[1][0]*0 + A[0][0]*ri)/det;
  return add(sc(k0,c1),sc(k1,c2));}
const d1=row(r[0]), d2=row(r[1]);
const n=unit([k0[1]*k1[2]-k0[2]*k1[1], k0[2]*k1[0]-k0[0]*k1[2], k0[0]*k1[1]-k0[1]*k1[0]]);
const a=[1,0.62];
console.log('d1',d1.map(x=>x.toFixed(4)),'d1.k0',dot(d1,k0).toExponential(2),'d1.k1',dot(d1,k1).toFixed(4));
console.log('n',n.map(x=>x.toFixed(4)));
// seeded held-out keys from the preserved distribution (mean m, not k0)
let s=12345; const rnd=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};
const gauss=()=>{let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
const held=[];for(let j=0;j<40;j++)held.push(unit(add(m,[gauss(),gauss(),gauss()].map(x=>x*0.5))));
function frag(t){let mx=0;for(const k of held){const c=dot(n,k);
  const o=[dot(d1,k)+t*c*a[0], dot(d2,k)+t*c*a[1]];const v=Math.hypot(o[0],o[1]);if(v>mx)mx=v;}return mx;}
function fnorm(t){const e1=add(d1,n,t),e2=add(d2,n,0.62*t);return Math.sqrt(dot(e1,e1)+dot(e2,e2));}
let best=0,bv=1e9,worst=0,wv=0;
for(let i=-300;i<=300;i++){const t=i/200;const f=frag(t);if(f<bv){bv=f;best=t;}if(f>wv){wv=f;worst=t;}}
console.log('frag(0)      =',frag(0).toFixed(4),' <- what the min-norm solver picks');
console.log('min frag     =',bv.toFixed(4),'at t =',best.toFixed(3));
console.log('penalty      =',(frag(0)/bv).toFixed(2)+'x');
console.log('range over slider t in [-1.5,1.5]:',bv.toFixed(3),'..',wv.toFixed(3),'=',(wv/bv).toFixed(1)+'x');
console.log('fnorm(0)=',fnorm(0).toFixed(4),'fnorm(best)=',fnorm(best).toFixed(4),' (norm min is at t=0 by construction)');
console.log('fit err at best:',Math.hypot(dot(add(d1,n,best),k1)-r[0],dot(add(d2,n,0.62*best),k1)-r[1]).toExponential(2));
console.log('k0 damage at best:',Math.hypot(dot(add(d1,n,best),k0),dot(add(d2,n,0.62*best),k0)).toExponential(2));
