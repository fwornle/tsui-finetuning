// Right framing: MEMIT's ridge solve already lands on the COVARIANCE-weighted (H2) optimum.
// What it does not give is the worst-case (Hinf) optimum. That is the actual unspent freedom.
const k0=[1,0.3,0], k1=[0.2,1,0.4], r=[0.8,-0.5], m=[0.62,0.5,-0.35];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0), add=(a,b,s=1)=>a.map((x,i)=>x+s*b[i]);
const sc=(a,s)=>a.map(x=>x*s), unit=a=>sc(a,1/Math.sqrt(dot(a,a)));
const A=[[dot(k0,k0),dot(k0,k1)],[dot(k1,k0),dot(k1,k1)]];
const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
const row=ri=>add(sc(k0,(-A[0][1]*ri)/det),sc(k1,(A[0][0]*ri)/det));
const d=[row(r[0]),row(r[1])];
const n=unit([k0[1]*k1[2]-k0[2]*k1[1],k0[2]*k1[0]-k0[0]*k1[2],k0[0]*k1[1]-k0[1]*k1[0]]);
const v=unit(k1), q=[1,Math.sqrt(3)];
let s=20260926; const rnd=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};
const gs=()=>{let u=0,w=0;while(!u)u=rnd();while(!w)w=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*w);};
const held=[]; const g=1.2;
for(let j=0;j<40;j++){const x=[gs(),gs(),gs()];
  const y=add(add(x,v,g*dot(n,x)),n,g*dot(v,x));
  held.push(unit(add(m,y,0.62)));}
// leakage of row-offsets (t1,t2): image_j = ( (d1.k + t1 c)*q1 , (d2.k + t2 c)*q2 )
const img=(t,k)=>{const c=dot(n,k);return [(dot(d[0],k)+t[0]*c)*q[0],(dot(d[1],k)+t[1]*c)*q[1]];};
const mx=t=>Math.max(...held.map(k=>Math.hypot(...img(t,k))));
const rms=t=>Math.sqrt(held.reduce((a,k)=>a+img(t,k).reduce((b,x)=>b+x*x,0),0)/held.length);
// H2 optimum: separable per row, closed form
const c=held.map(k=>dot(n,k));
const t2opt=[0,1].map(i=>-held.reduce((a,k,j)=>a+dot(d[i],k)*c[j],0)/held.reduce((a,_,j)=>a+c[j]*c[j],0));
// Hinf optimum: dense 2-D grid then refine
let bt=[0,0],bv=1e9;
for(let i=-1200;i<=1200;i++)for(let j=-1200;j<=1200;j+=1){ if(i%3||j%3) continue;
  const t=[i/400,j/400],val=mx(t); if(val<bv){bv=val;bt=t;} }
for(let it=0;it<3;it++){const st=0.0075/(it+1);let imp=true;
  while(imp){imp=false;for(const dv of [[st,0],[-st,0],[0,st],[0,-st],[st,st],[-st,-st],[st,-st],[-st,st]]){
    const t=[bt[0]+dv[0],bt[1]+dv[1]],val=mx(t); if(val<bv-1e-12){bv=val;bt=t;imp=true;}}}}
console.log('H2  optimum (t1,t2) =',t2opt.map(x=>x.toFixed(3)),' maxleak',mx(t2opt).toFixed(4),' rms',rms(t2opt).toFixed(4));
console.log('Hinf optimum (t1,t2) =',bt.map(x=>x.toFixed(3)),' maxleak',bv.toFixed(4),' rms',rms(bt).toFixed(4));
console.log('min-norm  (0,0)      : maxleak',mx([0,0]).toFixed(4),' rms',rms([0,0]).toFixed(4));
console.log();
console.log('Hinf penalty paid by the H2 choice :',(mx(t2opt)/bv).toFixed(3)+'x');
console.log('Hinf penalty paid by min-norm      :',(mx([0,0])/bv).toFixed(3)+'x');
console.log('rms penalty paid by the Hinf choice:',(rms(bt)/rms(t2opt)).toFixed(3)+'x');
// slider: extend the line through H2 -> Hinf
const dir=[bt[0]-t2opt[0],bt[1]-t2opt[1]];
const at=u=>[t2opt[0]+u*dir[0],t2opt[1]+u*dir[1]];
let lo=1e9,hi=0;for(let i=-250;i<=350;i++){const val=mx(at(i/100));lo=Math.min(lo,val);hi=Math.max(hi,val);}
console.log('along the slider u in [-2.5,3.5]: maxleak',lo.toFixed(3),'..',hi.toFixed(3),'=',(hi/lo).toFixed(2)+'x');
console.log('fit err at Hinf:',Math.hypot(dot(add(d[0],n,bt[0]),k1)-r[0],dot(add(d[1],n,bt[1]),k1)-r[1]).toExponential(2));
console.log('k0 dmg at Hinf:',Math.hypot(dot(add(d[0],n,bt[0]),k0),dot(add(d[1],n,bt[1]),k0)).toExponential(2));
