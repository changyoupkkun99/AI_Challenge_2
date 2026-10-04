from pathlib import Path
from collections import defaultdict
import json,time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix

P=Path(__file__).resolve().parent; z=np.load(P/'inputs.npz'); A=z['A']; K=z['K']; base=z['base']; lock=z['lock']; IDS=z['ids']; n=len(IDS)
ranked=json.loads((P/'ranked_candidates.json').read_text())
# Preserve the previously user-fixed 0047=d; it is not needed for any robust claim.
eligible=[r for r in ranked if r['id']!='test_0047.jpg']; target={r['index']:r for r in eligible}; m=len(eligible)
groups=defaultdict(list)
for i,row in enumerate(A): groups[(row.tobytes(),i if i in target else -1)].append(i)
members=list(groups.values()); sizes=np.array([len(g) for g in members]); G=len(members); v=5*G; M=lil_matrix((G+A.shape[1]+1,v)); b=np.r_[sizes,K,3357]
g_for={i:g for g,ms in enumerate(members) for i in ms}
for g,ms in enumerate(members):
    M[g,5*g:5*g+5]=1
    for j,a in enumerate(A[ms[0]]): M[G+j,5*g+1+int(a)]=1
    M[-1,5*g+1:5*g+5]=1
M=M.tocsr(); upper=np.repeat(sizes,5)
D=np.zeros((m,v))
for j,r in enumerate(eligible):
    g=g_for[r['index']]; assert sizes[g]==1 and not lock[r['index']]
    D[j,5*g+1+ord(r['after'])-97]=1; D[j,5*g+1+ord(r['before'])-97]=-1
logs=[]; witnesses=[]
def solve(c,label,limit=35):
    start=time.time(); res=milp(c,integrality=np.ones(v),bounds=Bounds(np.zeros(v),upper),constraints=LinearConstraint(M,b,b),options={'time_limit':limit,'mip_rel_gap':0})
    x=None
    if res.x is not None:
        x=np.rint(res.x).astype(int); assert np.array_equal(M@x,b); assert np.all(x>=0) and np.all(x<=upper); witnesses.append(x)
    rec=dict(label=label,status=int(res.status),seconds=round(time.time()-start,3),objective=float(c@x) if x is not None else None,bound=float(res.mip_dual_bound) if res.get('mip_dual_bound') is not None else None)
    logs.append(rec); print(json.dumps(rec),flush=True)
    (P/'integer_solver_log.json').write_text(json.dumps(logs,indent=2))
    return res,x

res,x=solve(np.zeros(v),'all_60_scores_feasible'); assert res.status==0
checks=[]
for count in [1,3,5,8,12,m]:
    c=D[:count].sum(0); lo,xlo=solve(c,f'top{count}_min'); hi,xhi=solve(-c,f'top{count}_max')
    checks.append(dict(top=count,minimum=int(round(c@xlo)) if lo.status==0 else None,maximum=int(round(c@xhi)) if hi.status==0 else None,ids=[r['id'] for r in eligible[:count]]))
    (P/'candidate_bounds.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))

# Cutting-plane robust subset selection. Every cut is a fully feasible integer world.
# Lexicographic objective: maximize guaranteed net correct, then model-estimated gain.
scenarios=[D@w for w in witnesses]; expected=np.array([r['minimum_model_estimated_gain'] for r in eligible]); eps=.2/(expected.sum()+1)
rounds=[]; robust=None
for iteration in range(35):
    C=np.column_stack([-np.array(scenarios),np.ones(len(scenarios))]); C=np.vstack([C,np.r_[np.ones(m),0]])
    ub=np.r_[np.zeros(len(scenarios)),12]
    objective=np.r_[-eps*expected,-1.]
    master=milp(objective,integrality=np.ones(m+1),bounds=Bounds(np.zeros(m+1),np.r_[np.ones(m),12]),constraints=LinearConstraint(C,np.full(len(ub),-np.inf),ub),options={'time_limit':15,'mip_rel_gap':0})
    assert master.status==0,master.message
    choose=np.rint(master.x[:m]).astype(int); proposed=int(round(master.x[-1])); c=choose@D
    rr,xx=solve(c,f'robust_round_{iteration}')
    actual=int(round(c@xx)) if xx is not None else None
    rounds.append(dict(iteration=iteration,master_guarantee=proposed,actual_worst=actual,changes=int(choose.sum()),ids=[eligible[j]['id'] for j in np.flatnonzero(choose)]))
    if rr.status!=0: break
    if actual==proposed:
        robust=dict(guaranteed_gain=actual,changes=[eligible[j] for j in np.flatnonzero(choose)],estimated_min_prior_gain=float(expected@choose),optimal_within='28 stable proposed alternatives, max 12 changes, 0047 held; not globally optimal over all answers',iterations=iteration+1); break
    scenarios.append(D@xx)
    (P/'robust_rounds.json').write_text(json.dumps(rounds,ensure_ascii=False,indent=2))
result=dict(robust=robust,rounds=rounds,eligible_count=m,held_0047=True,assumptions='Fixed 3357 public rows, equally weighted single-answer accuracy, five-decimal truncation and exact saved file-score mapping',guarantee_scope='Relative to reference under the listed assumptions and all 60 observed score constraints; not independent Kaggle verification')
(P/'robust_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); np.savez_compressed(P/'integer_witnesses.npz',witnesses=np.array(witnesses),members=np.array([','.join(map(str,ms)) for ms in members]),D=D)
print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
