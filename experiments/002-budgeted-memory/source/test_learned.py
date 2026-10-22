import numpy as np
import torch
from learned import Delta,codes,predict_neural
from benchmark import generate

torch.manual_seed(710)
model=Delta().cuda()
s=generate(711,1100,'noisy')
with torch.no_grad():
    k=model.keys(torch.from_numpy(s.keys).cuda())
    S=torch.zeros(204,5,device='cuda');beta=torch.sigmoid(model.write)
    targets=codes(torch.from_numpy(s.values.astype(np.int64)).cuda())
    predictions=np.full(len(s.ops),16,np.uint8)
    for t in range(len(s.ops)):
        read=k[t]@S
        if s.ops[t]==2:
            if read[4]>.5:
                predictions[t]=sum(int(read[c]>0)*(1<<c) for c in range(4))
        else:S=S+torch.outer(k[t],beta*(targets[t]-read))
    assert S.numel()*S.element_size()==4080
    reference=predict_neural(s,model)
    assert np.array_equal(reference[s.ops==2],predictions[s.ops==2])
    # Generation must not supply evaluator-only labels or entity IDs.
    s.truth[:]=5;s.entity[:]=0;s.previous[:]=4
    assert np.array_equal(reference,predict_neural(s,model))
print('PASS: Torch/Numba query predictions agree, 4080-byte state, evaluator isolation')
