"""Small semantic tests that would catch invalid memory-update experiments."""
import numpy as np
from benchmark import generate,metrics
from memories import explicit,dense,predict,configurations,config_bytes

def test_exact_events():
    # SET A=2, SET B=7, SET A=9, query B/A, delete A, query A/B.
    ops=np.array([0,0,0,2,2,1,2,2],np.int8)
    ids=np.array([0,1,0,1,0,0,0,1])
    vals=np.array([2,7,9,16,16,16,16,16],np.uint8)
    keys=np.eye(2,dtype=np.float32)[ids]
    p,_=explicit(ops,keys,vals,2,.99)
    assert p[[3,4,6,7]].tolist()==[7,9,16,7]
    # Orthogonal sign features make the delta-rule semantic check exact.
    signs=np.array([[1,1],[1,-1]],np.int8)[ids]
    p=dense(ops,signs,vals,1,1.,1.,.5)
    assert p[[3,4,6,7]].tolist()==[7,9,16,7]

def test_packed_signatures():
    ops=np.array([0,0,2,2],np.int8)
    keys=np.array([[0,255,0,255],[255,0,255,0],[0,255,0,255],[255,0,255,0]],np.uint8)
    p,_=explicit(ops,keys,np.array([3,4,16,16],np.uint8),2,.99,True)
    assert p[2:].tolist()==[3,4]

def test_budget_and_causality():
    for budget in [256,1024,4096]:
        for c in configurations(budget):assert config_bytes(c)<=budget
    s=generate(42,1100,'noisy');original=s.keys.copy()
    for family in ['cache','delta','additive','hybrid']:
        c=next(c for c in configurations(1024) if c['family']==family)
        p,_=predict(s,c)
        # Modifying future events must not change past predictions.
        cut=1050;s.ops[cut:]=1
        q,_=predict(s,c)
        assert np.array_equal(p[:cut],q[:cut])
        # Evaluator labels must not affect predictions.
        s.truth[:]=7;s.previous[:]=3;s.entity[:]=127
        r,_=predict(s,c);assert np.array_equal(q,r)
    assert np.array_equal(s.keys,original)

if __name__=='__main__':
    test_exact_events();test_packed_signatures();test_budget_and_causality()
    print('PASS: overwrite, noninterference on orthogonal keys, deletion, packed signatures, budgets, causality, evaluator isolation')
