// Higher dimension. Which axis of the leftover freedom actually matters:
//   (a) Euclidean min-norm  vs  Gramian-weighted min-norm (= what MEMIT's covariance term does)
//   (b) average-case (H2)   vs  worst-case (Hinf) within that
const D=40, O=6, NP=8, NF=2;
function mk(seed){let s=seed>>>0;return()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};}
const rnd=mk(20260926);
const gs=()=>{let u=0,w=0;while(!u)u=rnd();while(!w)w=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*w);};
const zeros=(n,m)=>Array.from({length:n},()=>new Float64Array(m));
function solve(Ain,Bin){ // solve A X = B, A n*n, B n*m  (Gauss, partial pivot)
  const n=Ain.length,m=Bin[0].length,A=Ain.map(r=>Float64Array.from(r)),B=Bin.map(r=>Float64Array.from(r));
  for(let c=0;c<n;c++){let p=c;for(let i=c+1;i<n;i++)if(Math.abs(A[i][c])>Math.abs(A[p][c]))p=i;
    [A[c],A[p]]=[A[p],A[c]];[B[c],B[p]]=[B[p],B[c]];
    const pv=A[c][c]||1e-300;
    for(let i=c+1;i<n;i++){const f=A[i][c]/pv;if(!f)continue;
      for(let j=c;j<n;j++)A[i][j]-=f*A[c][j];for(let j=0;j<m;j++)B[i][j]-=f*B[c][j];}}
  const X=zeros(n,m);
  for(let i=n-1;i>=0;i--)for(let j=0;j<m;j++){let v=B[i][j];
    for(let k=i+1;k<n;k++)v-=A[i][k]*X[k][j];X[i][j]=v/(A[i][i]||1e-300);}
  return X;}
// random orthogonal basis (Gram-Schmidt on gaussians)
const R=[];for(let i=0;i<D;i++){let v=Array.from({length:D},gs);
  for(const u of R){const d=v.reduce((s,x,k)=>s+x*u[k],0);v=v.map((x,k)=>x-d*u[k]);}
  const nn=Math.hypot(...v);R.push(v.map(x=>x/nn));}
function run(kappa){
  const lam=Array.from({length:D},(_,i)=>Math.exp(-Math.log(kappa)*i/(D-1)));
  const Cm=zeros(D,D);
  for(let a=0;a<D;a++)for(let b=0;b<D;b++){let s=0;for(let i=0;i<D;i++)s+=lam[i]*R[i][a]*R[i][b];Cm[a][b]=s;}
  const half=x=>{const y=new Float64Array(D);
    for(let i=0;i<D;i++){const c=Math.sqrt(lam[i]);let d=0;for(let k=0;k<D;k++)d+=R[i][k]*x[k];
      for(let k=0;k<D;k++)y[k]+=c*d*R[i][k];}return y;};
  const draw=()=>half(Array.from({length:D},gs));
  const KP=Array.from({length:NP},draw), KF=Array.from({length:NF},draw);
  const held=Array.from({length:300},draw);
  const K=[...KP,...KF], nc=K.length;                      // all constraint keys
  const Rhs=zeros(O,nc);                                   // 0 for protected, target for new facts
  for(let o=0;o<O;o++)for(let j=0;j<NF;j++)Rhs[o][NP+j]=gs()*0.5;
  // weighted min-norm solution: min tr(D W^-1 D^T) s.t. D K = Rhs  ->  D = Rhs (K^T W^-1 K)^-1 K^T W^-1
  function sol(Winv){ // Winv: D*D
    const WK=zeros(D,nc);
    for(let a=0;a<D;a++)for(let j=0;j<nc;j++){let s=0;for(let b=0;b<D;b++)s+=Winv[a][b]*K[j][b];WK[a][j]=s;}
    const G=zeros(nc,nc);
    for(let i=0;i<nc;i++)for(let j=0;j<nc;j++){let s=0;for(let a=0;a<D;a++)s+=K[i][a]*WK[a][j];G[i][j]=s;}
    const Y=solve(G,Array.from({length:nc},(_,i)=>Float64Array.from({length:O},(_,o)=>Rhs[o][i])));
    const Dm=zeros(O,D);
    for(let o=0;o<O;o++)for(let a=0;a<D;a++){let s=0;for(let j=0;j<nc;j++)s+=WK[a][j]*Y[j][o];Dm[o][a]=s;}
    return Dm;}
  const Ci=zeros(D,D);
  for(let a=0;a<D;a++)for(let b=0;b<D;b++){let s=0;for(let i=0;i<D;i++)s+=R[i][a]*R[i][b]/lam[i];Ci[a][b]=s;}
  const I=zeros(D,D);for(let i=0;i<D;i++)I[i][i]=1;
  const DE=sol(I);                 // Euclidean min-norm: min tr(DD^T)   <- plain pseudo-inverse
  const Dm=sol(Ci);                // min tr(D C D^T) = min E||Dk||^2    <- MEMIT's covariance term
  const leak=Dd=>{let mx=0,ss=0;for(const k of held){let s2=0;
      for(let o=0;o<O;o++){let v=0;for(let a=0;a<D;a++)v+=Dd[o][a]*k[a];s2+=v*v;}
      ss+=s2;mx=Math.max(mx,Math.sqrt(s2));}return[mx,Math.sqrt(ss/held.length)];};
  const chk=Dd=>{let f=0,p=0;
    for(let j=0;j<nc;j++)for(let o=0;o<O;o++){let v=0;for(let a=0;a<D;a++)v+=Dd[o][a]*K[j][a];
      const e=v-Rhs[o][j];if(j<NP)p=Math.max(p,Math.abs(e));else f=Math.max(f,Math.abs(e));}return[f,p];};
  const [me,re]=leak(DE),[mm,rm]=leak(Dm);
  return {kappa,euclid_max:me,metric_max:mm,euclid_rms:re,metric_rms:rm,
          gain_max:me/mm,gain_rms:re/rm,chkE:chk(DE),chkM:chk(Dm)};
}
console.log('freedom dimension = O*(D-nc) =',O*(D-NP-NF),'parameters after the constraints\n');
console.log('kappa'.padEnd(9),'euclid max'.padEnd(12),'metric max'.padEnd(12),'worse by'.padEnd(10),'rms worse by');
for(const k of [1,3,10,30,100,300,1000,3000]){const r=run(k);
  console.log(String(k).padEnd(9),r.euclid_max.toFixed(4).padEnd(12),r.metric_max.toFixed(4).padEnd(12),
    (r.gain_max.toFixed(2)+'x').padEnd(10),r.gain_rms.toFixed(2)+'x');}
const r=run(300);
console.log('\nconstraint check at kappa=300 (fit err, preserved-key err):');
console.log('  euclid:',r.chkE.map(x=>x.toExponential(1)).join('  '));
console.log('  metric:',r.chkM.map(x=>x.toExponential(1)).join('  '));
