"""Validate experiment measurements without invoking a model or GPU."""
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from serve_benchmark import check_answer,complete

assert check_answer('["123456"]',['123456'])==(True,True)
assert check_answer('[123456]',['123456'])==(True,False)
assert check_answer('Answer: ["123456"]',['123456'])==(True,False)
assert check_answer('["123457"]',['123456'])==(False,False)
assert check_answer('["123456","000000"]',['123456'])==(False,False)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        for text,tokens,delay in [('["',[1],.05),('123',[2],.08),('456',[3],.08),('"]',[4],.08)]:
            time.sleep(delay)
            self.wfile.write(('data: '+json.dumps({'content':text,'tokens':tokens,'stop':False})+'\n\n').encode());self.wfile.flush()
        event={'content':'','tokens':[],'stop':True,'timings':{'predicted_n':4,'cache_n':16000},'truncated':False}
        self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    row=complete(f'http://127.0.0.1:{server.server_port}',{'prompt':'test'},['123456'])
    assert row['correct'] and row['strict_correct'] and row['qualified'],row
    assert row['ttft_s']<row['completion_s']/2,row
    assert .04<row['mean_tpot_s']<.15,row
    assert row['final']['timings']['cache_n']==16000
    print(json.dumps({'answer_checker':'passed','real_HTTP_SSE_client':'passed','first_token_not_buffered':True,'ttft_s':row['ttft_s'],'completion_s':row['completion_s'],'mean_tpot_s':row['mean_tpot_s']},indent=2))
finally:server.shutdown();server.server_close()
