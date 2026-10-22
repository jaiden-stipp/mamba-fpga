"""Standard counting bound and a slow, storage-tight symbolic control.

Not an invention. Dense base-17 coding prioritizes bytes, not execution cost.
"""
import json,math
from pathlib import Path
import numpy as np
from benchmark import generate,REGIMES,metrics
ROOT=Path(__file__).resolve().parents[1]
POWERS=tuple(17**i for i in range(128))

class PackedWorld:
    def __init__(self,N=128):
        self.powers=POWERS if N==128 else tuple(17**i for i in range(N))
        self.payload=bytearray(((17**N-1).bit_length()+7)//8)
        self.payload[:]=(17**N-1).to_bytes(len(self.payload),'little')
    def query(self,key):
        return (int.from_bytes(self.payload,'little')//self.powers[int(key)])%17
    def update(self,key,value):
        code=int.from_bytes(self.payload,'little');power=self.powers[int(key)]
        old=(code//power)%17;code+=(int(value)-old)*power
        self.payload[:]=code.to_bytes(len(self.payload),'little')

def run():
    rows=[]
    for split,seeds,length,stress in [('test',range(200,208),12288,False),('stress',range(300,304),49152,True)]:
        for regime in REGIMES:
            for seed in seeds:
                s=generate(seed,length,regime,stress);m=PackedWorld();p=np.full(length,16,np.uint8)
                for t in range(length):
                    if s.ops[t]==2:p[t]=m.query(s.entity[t])
                    else:m.update(s.entity[t],s.values[t])
                result=metrics(s,p);assert result['all_accuracy']==1.
                rows.append(dict(split=split,regime=regime,seed=seed,persistent_bytes=len(m.payload),**result))
    out=dict(description='Exact-ID diagnostic only; standard base-17 packing, not a new method.',
        lower_bound_bits=(17**128-1).bit_length(),payload_bytes=66,five_bit_packing_bytes=80,byte_table_bytes=128,rows=rows)
    (ROOT/'results/symbolic_bound.json').write_text(json.dumps(out,indent=2));print('PASS: 66-byte symbolic control, all 36 streams exact; counting lower bound 524 bits')
if __name__=='__main__':run()
