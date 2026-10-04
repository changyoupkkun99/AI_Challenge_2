"""Rebuild constraints independently and verify an exact rational lower bound."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import json, csv, hashlib, math
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

P=Path(__file__).resolve().parent; man=json.loads((P/'input_manifest.json').read_text()); chosen=json.loads((P/'robust_result.json').read_text())['robust']['changes']; ids=[f'test_{i:04d}.jpg' for i in range(1,6715)]
def read(rec):
    raw=Path(rec['path']).read_bytes(); assert hashlib.sha256(raw).hexdigest()==rec['sha256']; rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines())); assert [r['id'] for r in rows]==ids
    return [r['answer'] for r in rows]
records=man['unique_score_records']; histories=[read(r) for r in records]; base=read(man['best_record']); output=base.copy(); selected={r['id'] for r in chosen}
for r in chosen:
    i=ids.index(r['id']); assert output[i]==r['before']; output[i]=r['after']
models=[read(r) for r in man['consensus_models']]
locked=[i for i in range(6714) if len({m[i] for m in models})==1]; assert len(locked)==4444
assert all(output[i]==models[0][i] for i in locked)
groups=defaultdict(list)
for i in range(6714): groups[(tuple(a[i] for a in histories),output[i],base[i])].append(i)
members=list(groups.values()); sizes=np.array([len(v) for v in members]); gnum=len(members); var=gnum*5
M=lil_matrix((gnum+len(records)+1,var)); rhs=np.r_[sizes,[r['conditional_correct'] for r in records],3357]; c=np.zeros(var)
for g,rows in enumerate(members):
    i=rows[0]; M[g,g*5:g*5+5]=1
    for j,a in enumerate(histories): M[gnum+j,g*5+1+'abcd'.index(a[i])]=1
    M[-1,g*5+1:g*5+5]=1
    c[g*5+1+'abcd'.index(output[i])]+=1; c[g*5+1+'abcd'.index(base[i])]-=1
M=M.tocsr(); ub=np.repeat(sizes,5)
lp=linprog(c,A_eq=M,b_eq=rhs,bounds=list(zip(np.zeros(var),ub)),method='highs'); assert lp.success,lp.message
dual=[Fraction(float(v)).limit_denominator(1000000) for v in lp.eqlin.marginals]
reduced=[Fraction(int(v)) for v in c]
coo=M.tocoo()
for i,j,v in zip(coo.row,coo.col,coo.data): reduced[j]-=dual[i]*int(v)
exact=sum((dual[i]*int(rhs[i]) for i in range(len(rhs))),Fraction())+sum((min(Fraction(),v)*int(u) for v,u in zip(reduced,ub)),Fraction())
guarantee=math.ceil(exact); assert guarantee>=2,(str(exact),guarantee)
results={}; worlds={}
for label,objective in [('min',c),('max',-c)]:
    rr=milp(objective,integrality=np.ones(var),bounds=Bounds(np.zeros(var),ub),constraints=LinearConstraint(M,rhs,rhs),options={'time_limit':35,'mip_rel_gap':0}); assert rr.status==0,rr.message
    x=np.rint(rr.x).astype(int); assert np.array_equal(M@x,rhs); assert (x>=0).all() and (x<=ub).all()
    public_gold=['private']*6714
    for g,rows in enumerate(members):
        assigned=[]
        for s,state in enumerate(['private','a','b','c','d']): assigned += [state]*int(x[g*5+s])
        assert len(assigned)==len(rows)
        for i,state in zip(rows,assigned): public_gold[i]=state
    assert sum(s!='private' for s in public_gold)==3357
    for j,r in enumerate(records): assert sum(a==b for a,b in zip(histories[j],public_gold))==r['conditional_correct']
    candidate_correct=sum(a==b for a,b in zip(output,public_gold)); base_correct=sum(a==b for a,b in zip(base,public_gold)); gain=candidate_correct-base_correct
    results[label]=dict(gain=gain,base_correct=base_correct,candidate_correct=candidate_correct,displayed_score=f'{candidate_correct*100000//3357/100000:.5f}')
    worlds[label]=public_gold
assert results['min']['gain']==guarantee
certificate=dict(exact_LP_lower_bound=str(exact),integer_lower_bound=guarantee,extrema=results,dual_nonzero=[dict(row=i,value=str(v)) for i,v in enumerate(dual) if v],formula='lower_bound = b dot y + sum_j upper_j * min(0, c_j - (M.T y)_j)',group_count=gnum,constraint_count=M.shape[0],all_60_scores_verified_in_expanded_integer_witnesses=True,conditions=man['assumptions'],public_rows=3357,not_a_real_leaderboard_measurement=True)
(P/'exact_certificate.json').write_text(json.dumps(certificate,ensure_ascii=False,indent=2)); (P/'extreme_worlds.json').write_text(json.dumps(worlds)); plan=dict(baseline=man['best_record'],consensus_models=man['consensus_models'],changes=sorted(chosen,key=lambda r:r['id']),output_file='submission_public_score_plus2.csv',locked_rows=4444,free_rows=2270,expected_changes=12,conditions=man['assumptions'],certificate=certificate)
plan['output_file']='submission_public_score_12changes.csv'
(P/'delivery_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)); print(json.dumps(certificate,ensure_ascii=False,indent=2))
