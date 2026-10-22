"""Real engine/client/tokenizer smoke; CPU only, excluded from GPU research."""
import json,subprocess,time,os
from pathlib import Path
import requests
from transformers import AutoTokenizer
from serve_benchmark import complete,ENGINE,MODEL,CREATE_NO_WINDOW
R=Path(__file__).resolve().parent;B=R.parent.parent
base='http://127.0.0.1:18674'
try:
    requests.get(base+'/health',timeout=1)
    raise SystemExit('Smoke port already occupied')
except requests.ConnectionError:pass
tok=AutoTokenizer.from_pretrained(Path(os.environ.get('QWEN3_MODEL_DIR', B/'experiments/003-kv-compression/assets/Qwen3-4B')),local_files_only=True)
text=tok.apply_chat_template([{'role':'user','content':'FACT: access code for project ORCHID is 123456. What is that code? Return only a JSON array containing the decimal string.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
ids=tok.encode(text,add_special_tokens=False)
cmd=[str(ENGINE),'-m',str(MODEL),'-ngl','0','--fit','off','-c','1024','-np','1','--host','127.0.0.1','--port','18674','--no-webui','--no-warmup','--cache-ram','0','--ctx-checkpoints','0','-t','8','-b','512','-ub','512']
with (R/'cpu_smoke.log').open('w',encoding='utf8') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=CREATE_NO_WINDOW)
    try:
        start=time.perf_counter()
        while time.perf_counter()-start<180:
            if proc.poll() is not None:raise RuntimeError('CPU smoke server exited')
            try:
                if requests.get(base+'/health',timeout=1).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.25)
        else:raise TimeoutError('CPU startup timed out')
        native=requests.post(base+'/tokenize',json={'content':text,'add_special':False,'parse_special':True},timeout=15).json()['tokens']
        assert native==ids,'Tokenizer ID mismatch'
        row=complete(base,{'prompt':ids,'temperature':0,'n_predict':24,'cache_prompt':False},['123456'])
        assert row.get('correct') and row.get('strict_correct'),row
        # Deliberately make the repeat request cacheable to audit cache_n semantics.
        second=complete(base,{'prompt':ids,'temperature':0,'n_predict':24,'cache_prompt':True,'id_slot':0},['123456'])
        assert second.get('correct'),second
        assert second['final']['timings']['cache_n']>=len(ids)-1,second
        receipt={'purpose':'CPU-only real engine integration smoke, excluded from GPU timing and quality claims','tokenizer_ids_match':True,'input_tokens':len(ids),'first':row,'repeat':second,'command':cmd}
        (R/'cpu_smoke.json').write_text(json.dumps(receipt,indent=2))
        print('CPU engine smoke passed; tokenizer IDs match; answer:',row['text'])
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
