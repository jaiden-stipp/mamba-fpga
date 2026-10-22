"""Actual HTTP/SSE serving pilot. Uses only own server subprocess handles."""
import argparse,concurrent.futures as cf,hashlib,json,os,statistics,subprocess,threading,time,platform
from pathlib import Path
import requests
R=Path(__file__).resolve().parent;B=R.parent.parent
ENGINE=Path(os.environ.get('LLAMA_SERVER', B/'tools/llama.cpp/bin/llama-server.exe'))
MODEL=Path(os.environ.get('QWEN3_GGUF', B/'assets/models/Qwen3-4B-BF16.gguf'))
SLO={'ttft_s':20.0,'mean_tpot_s':.100,'completion_s':45.0}
CREATE_NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)

def write_json(path,value):
    """Publish a complete snapshot and tolerate transient Windows file locks."""
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8')
    for attempt in range(10):
        try:temporary.replace(path);return
        except OSError:
            if attempt==9:raise
            time.sleep(.1*(attempt+1))

def sse_lines(response):
    """Read available bytes without waiting for a full application-sized chunk."""
    pending=b''
    while True:
        chunk=response.raw.read1(65536,decode_content=True)
        if not chunk:break
        pending+=chunk
        lines=pending.split(b'\n');pending=lines.pop()
        for line in lines:yield line.rstrip(b'\r')
    if pending:yield pending.rstrip(b'\r')

def gpu_info(index):
    p=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=CREATE_NO_WINDOW,timeout=10)
    if p.returncode:return {'error':p.stderr[:300]}
    values=p.stdout.strip().split(',')
    return dict(zip(['used_MiB','free_MiB','util_percent','temperature_C','clock_MHz'],[float(x.strip()) for x in values]))

def check_answer(text,expected):
    stripped=text.strip();decoded=None;strict=False
    try:
        decoded=json.loads(stripped);strict=isinstance(decoded,list) and all(isinstance(v,str) for v in decoded)
    except (ValueError,TypeError):
        at=stripped.find('[')
        if at>=0:
            try:decoded=json.JSONDecoder().raw_decode(stripped[at:])[0]
            except ValueError:pass
    good=isinstance(decoded,list) and [str(v) for v in decoded]==expected
    return good,bool(good and strict)

def complete(base,payload,expected=None):
    start=time.perf_counter();first=None;token_times=[];pieces=[];final={}
    row={'slo':SLO}
    try:
        # Avoid returning a second copy of the long prompt in the final SSE event.
        fields=['content','tokens','stop','timings','truncated','stop_type','stopping_word','tokens_evaluated','tokens_predicted','tokens_cached','id_slot']
        res=requests.post(base+'/completion',json={**payload,'stream':True,'response_fields':fields},stream=True,timeout=(10,120))
        res.raise_for_status()
        for line in sse_lines(res):
            if not line.startswith(b'data:'):continue
            raw=line[5:].strip()
            if raw==b'[DONE]':continue
            event=json.loads(raw);now=time.perf_counter()
            if event.get('error'):raise RuntimeError(str(event['error']))
            content=event.get('content','');tokens=event.get('tokens',[])
            if tokens or content:
                if first is None:first=now
                token_times.extend([now]*max(len(tokens),1));pieces.append(content)
            if event.get('stop'):final=event
        end=time.perf_counter();res.close();text=''.join(pieces)
        predicted=final.get('timings',{}).get('predicted_n',len(token_times))
        ttft=first-start if first is not None else None
        mean=(end-first)/(predicted-1) if first is not None and predicted>1 else 0.0
        good,strict=check_answer(text,expected) if expected is not None else (None,None)
        intervals=[b-a for a,b in zip(token_times,token_times[1:])]
        row.update(text=text,correct=good,strict_correct=strict,ttft_s=ttft,completion_s=end-start,mean_tpot_s=mean,p95_observed_token_interval_s=sorted(intervals)[int(.95*(len(intervals)-1))] if intervals else None,output_tokens=predicted,final=final,expected=expected)
        row['deadlines_met']=ttft is not None and ttft<=SLO['ttft_s'] and mean<=SLO['mean_tpot_s'] and end-start<=SLO['completion_s']
        row['qualified']=bool(good and row['deadlines_met'] and not final.get('truncated',False))
        if not final:row.update(error='missing final stream event',qualified=False)
    except Exception as e:
        row.update(error=type(e).__name__+': '+str(e),completion_s=time.perf_counter()-start,correct=False,strict_correct=False,qualified=False,deadlines_met=False,text=''.join(pieces))
    return row

class Server:
    def __init__(self,fmt,slots,args):
        self.fmt=fmt;self.slots=slots;self.args=args;self.base=f'http://127.0.0.1:{args.port}'
        self.logpath=R/'logs'/f'{args.tag}_{fmt}_c{slots}.log';self.logpath.parent.mkdir(exist_ok=True)
    def __enter__(self):
        try:
            requests.get(self.base+'/health',timeout=1)
            raise RuntimeError('Port already occupied; choose another --port')
        except requests.ConnectionError:pass
        cmd=[str(ENGINE),'-m',str(MODEL),'--host','127.0.0.1','--port',str(self.args.port),'-ngl','999','--split-mode','none','--main-gpu',str(self.args.gpu),'--fit','off','-fa','on','-ctk',self.fmt,'-ctv',self.fmt,'-np',str(self.slots),'-c',str(self.slots*17408),'-b','2048','-ub','512','-t','12','--no-kv-unified','--cache-ram','0','--ctx-checkpoints','0','--no-context-shift','--no-webui','--metrics','--slots','--load-mode','none']
        cmd+=['--verbosity','4']
        self.cmd=cmd;self.log=self.logpath.open('w',encoding='utf8');self.proc=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,creationflags=CREATE_NO_WINDOW)
        start=time.perf_counter()
        try:
            while time.perf_counter()-start<180:
                if self.proc.poll() is not None:raise RuntimeError(f'Server exited {self.proc.returncode}; see {self.logpath.name}')
                try:
                    r=requests.get(self.base+'/health',timeout=1)
                    if r.status_code==200:
                        self.setup_s=time.perf_counter()-start;return self
                except requests.RequestException:pass
                time.sleep(.25)
            raise TimeoutError('Server startup exceeded 180 seconds')
        except BaseException:
            self.close();raise
    def close(self):
        if getattr(self,'proc',None) and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=10)
        if getattr(self,'log',None):self.log.close()
    def __exit__(self,*_):self.close()

def wave(server,jobs,config,phase,outfile,workers):
    start=time.perf_counter();samples=[];stop=threading.Event()
    def sample():
        while not stop.is_set():
            samples.append({'elapsed_s':time.perf_counter()-start,**gpu_info(server.args.gpu)})
            stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    result=[]
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(complete,server.base,payload,expected):meta for payload,expected,meta in jobs}
        for fut in cf.as_completed(futures):
            row={**config,'phase':phase,**futures[fut],**fut.result()}
            row['reused_prompt_tokens']=row.get('final',{}).get('timings',{}).get('cache_n',0)
            if phase.startswith('resident_') and phase!='resident_load':
                row['resident_cache_hit']=row['reused_prompt_tokens']>=row['document_tokens']-1
            result.append(row)
    elapsed=time.perf_counter()-start;stop.set();thread.join(timeout=12)
    for row in result:outfile.write(json.dumps(row)+'\n')
    outfile.flush()
    summary={**config,'phase':phase,'requests':len(result),'correct':sum(bool(x.get('correct')) for x in result),'strict_correct':sum(bool(x.get('strict_correct')) for x in result),'qualified':sum(bool(x.get('qualified')) for x in result),'resident_cache_hits':sum(bool(x.get('resident_cache_hit')) for x in result),'makespan_s':elapsed,'qualified_requests_per_s':sum(bool(x.get('qualified')) for x in result)/elapsed,'output_tokens_per_s':sum(x.get('output_tokens',0) for x in result)/elapsed,'gpu_samples':samples}
    print(config['format'],config['slots'],phase,'correct',summary['correct'],'qualified',summary['qualified'],'/',len(result),'seconds',round(elapsed,2),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--port',type=int,default=18673);ap.add_argument('--slots',default='1,4,8');ap.add_argument('--formats',default='f16,q8_0,q4_0');ap.add_argument('--tag',default='pilot');ap.add_argument('--allow-busy-gpu',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if (R/f'raw_{args.tag}.jsonl').exists():raise SystemExit('Tag already has raw evidence; choose a fresh --tag.')
    before=gpu_info(args.gpu)
    (R/f'preflight_{args.tag}.json').write_text(json.dumps({'gpu':before,'arguments':vars(args),'slo':SLO,'python':platform.python_version(),'status':'blocked_competing_allocations' if before.get('used_MiB',1e9)>3000 and not args.allow_busy_gpu else 'ready','note':'No other process unloaded or terminated by this check.'},indent=2))
    if not args.allow_busy_gpu and before.get('used_MiB',1e9)>3000:
        raise SystemExit(f'GPU {args.gpu} has competing allocations: {before}. Unload models with authorization before timing; no services were changed.')
    cases=json.loads((R/'workload.json').read_text());formats=args.formats.split(',');slotslist=[int(x) for x in args.slots.split(',')]
    configlog=[];summaries=[]
    with (R/f'raw_{args.tag}.jsonl').open('w',encoding='utf8') as f:
        for cindex,slots in enumerate(slotslist):
            order=formats[cindex%len(formats):]+formats[:cindex%len(formats)]
            for fmt in order:
                config={'format':fmt,'slots':slots,'gpu':args.gpu,'tag':args.tag}
                # This checkpoint has 36 layers, 8 physical KV heads and d_head=128.
                # Refuse an impossible device-resident allocation rather than risk WDDM spill.
                kv_bytes=2*36*8*128*slots*17408*{'f16':2,'q8_0':34/32,'q4_0':18/32}[fmt]
                minimum_bytes=MODEL.stat().st_size+kv_bytes
                if minimum_bytes>before['free_MiB']*1024**2:
                    configlog.append({**config,'status':'not_run_capacity_guard','estimated_weight_plus_kv_bytes':int(minimum_bytes),'estimated_kv_bytes':int(kv_bytes),'available_device_bytes':int(before['free_MiB']*1024**2),'note':'Analytical allocation guard, not an observed OOM; scratch space excluded.'})
                    write_json(R/f'configurations_{args.tag}.json',configlog)
                    print('CAPACITY GUARD',fmt,slots,flush=True)
                    continue
                try:
                    with Server(fmt,slots,args) as server:
                        configlog.append({**config,'status':'ready','command':server.cmd,'setup_s':server.setup_s,'gpu_after_load':gpu_info(args.gpu)})
                        write_json(R/f'configurations_{args.tag}.json',configlog)
                        # Short engine warmup, excluded from recorded requests.
                        complete(server.base,{'prompt':[151644,872,198,9707,151645,198,151644,77091,198],'n_predict':8,'temperature':0,'cache_prompt':False})
                        cold=[]
                        for i,case in enumerate(cases[:1] if args.smoke else cases):
                            q=case['queries'][i%4]
                            cold.append(({'prompt':q['ids'],'n_predict':256,'temperature':0,'seed':42,'cache_prompt':False},q['expected'],{'document_seed':case['seed'],'task':q['kind'],'input_tokens':q['input_tokens']}))
                        summaries.append(wave(server,cold,config,'cold',f,len(cold)))
                        write_json(R/f'waves_{args.tag}.json',summaries)
                        if args.smoke:continue
                        # Prime a distinct document in each slot; charge/report all loading separately.
                        priming=[]
                        for slot in range(slots):
                            case=cases[slot%len(cases)]
                            priming.append(({'prompt':case['document_ids'],'n_predict':0,'cache_prompt':True,'id_slot':slot},None,{'document_seed':case['seed'],'slot':slot,'document_tokens':case['document_tokens']}))
                        summaries.append(wave(server,priming,config,'resident_load',f,slots))
                        write_json(R/f'waves_{args.tag}.json',summaries)
                        for turn in range(4):
                            jobs=[]
                            for slot in range(slots):
                                case=cases[slot%len(cases)];q=case['queries'][turn]
                                jobs.append(({'prompt':q['ids'],'n_predict':256,'temperature':0,'seed':42,'cache_prompt':True,'id_slot':slot},q['expected'],{'document_seed':case['seed'],'task':q['kind'],'input_tokens':q['input_tokens'],'document_tokens':case['document_tokens'],'slot':slot,'turn':turn}))
                            summaries.append(wave(server,jobs,config,f'resident_{turn}',f,slots))
                            write_json(R/f'waves_{args.tag}.json',summaries)
                except Exception as e:
                    configlog.append({**config,'status':'failed','error':str(e),'gpu_after_failure':gpu_info(args.gpu)})
                    print('CONFIG FAILED',fmt,slots,str(e),flush=True)
                write_json(R/f'configurations_{args.tag}.json',configlog)
                write_json(R/f'waves_{args.tag}.json',summaries)
    print('Finished',args.tag,flush=True)

if __name__=='__main__':main()
