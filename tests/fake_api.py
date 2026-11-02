"""A fake Anthropic Messages API (streaming SSE) to exercise the real SDK code path end to end.

Usage: python fake_api.py PORT TRUTH.musicxml CHUNKS   (e.g. 5,6,6,6,1)
It answers each system from ground truth, injects one rhythm error, and checks
that every request is well formed (images first, tool_result pairing, thinking
blocks passed back with signatures).
"""
import copy
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scorescribe.draft import score_to_systems  # noqa: E402

port, truth, chunks = int(sys.argv[1]), sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
meta, systems = score_to_systems(truth, use_breaks=False)
allm, TRUTH, i = systems[0]["measures"], [], 0
for c in chunks:
    TRUTH.append({"measures": copy.deepcopy(allm[i:i + c])})
    i += c
TRUTH[0]["parts"] = meta["parts"]
for sy in TRUTH:
    for st in sy["measures"][0]["staves"]:
        st.setdefault("clef", "G2" if st["staff"] == 1 else "F4")
    sy["measures"][0].setdefault("key", -4)
PROBLEMS = []
SEEN = {"first": {}, "consensus": 0, "recount": 0, "images": set()}
import threading
LOCK = threading.Lock()


def check(body):
    msgs = body["messages"]
    if body.get("thinking") != {"type": "adaptive"}:
        PROBLEMS.append("thinking not adaptive")
    if body.get("tool_choice", {}).get("type") != "auto":
        PROBLEMS.append("tool_choice must be auto with thinking")
    if not body.get("output_config", {}).get("effort"):
        PROBLEMS.append("no effort")
    first = msgs[0]["content"]
    if first[0]["type"] != "image" or first[0]["source"]["media_type"] != "image/png":
        PROBLEMS.append("first block is not an image")
    SEEN["images"].add(sum(1 for b in first if b["type"] == "image"))
    marks = sum(1 for m in msgs for b in m["content"] if isinstance(b, dict) and "cache_control" in b)
    marks += sum(1 for b in body.get("system", []) if "cache_control" in b)
    if marks > 4:
        PROBLEMS.append(f"{marks} cache breakpoints (max 4)")
    if "cache_control" not in msgs[-1]["content"][-1]:
        PROBLEMS.append("latest user turn not marked for caching")
    for k in range(1, len(msgs)):
        m = msgs[k]
        if m["role"] == "assistant":
            th = [b for b in m["content"] if b["type"] == "thinking"]
            if th and not th[0].get("signature"):
                PROBLEMS.append("thinking block without signature")
        elif msgs[k - 1]["role"] == "assistant":
            prev_tools = [b["id"] for b in msgs[k - 1]["content"] if b["type"] == "tool_use"]
            results = [b for b in m["content"] if b["type"] == "tool_result"]
            if prev_tools and (not results or results[0]["tool_use_id"] != prev_tools[0] or m["content"][0]["type"] != "tool_result"):
                PROBLEMS.append("tool_use not answered by a leading tool_result")
    if msgs[-1]["role"] != "user":
        PROBLEMS.append("last message not user")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.startswith("/__problems"):
            return
        check(body)
        text = body["messages"][0]["content"][-1]["text"]
        idx = int(text.split("This is system ")[1].split(" ")[0])
        last = body["messages"][-1]["content"][-1].get("text", "")
        res = copy.deepcopy(TRUTH[idx - 1])
        first_call = len(body["messages"]) == 1
        if first_call:
            with LOCK:
                n = SEEN["first"][idx] = SEEN["first"].get(idx, 0) + 1
            if idx == 2:   # both readings make the same rhythm slip -> repair round
                v = res["measures"][2]["staves"][1]["voices"][0].split()
                res["measures"][2]["staves"][1]["voices"][0] = " ".join(v[:-1])
            if idx == 3 and n == 2:   # second reading misreads a pitch -> consensus
                v = res["measures"][1]["staves"][0]["voices"][0]
                res["measures"][1]["staves"][0]["voices"][0] = v.replace("A5", "B5", 1)
            if idx == 4:   # both readings skip the last measure -> barline recount
                res["measures"] = res["measures"][:-1]
        if "second, independent reading" in last:
            SEEN["consensus"] += 1
            if "yours:" not in last:
                PROBLEMS.append("consensus message without diffs")
            res["changes"] = "kept A5 after zooming in"
        if "Automatic checks" in last:
            if "MEASURE COUNT" in last:
                SEEN["recount"] += 1
            res["changes"] = "fixed"
        if "Proofreading" in last:
            res["changes"] = "none"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def ev(name, data):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        ev("message_start", {"type": "message_start", "message": {
            "id": "msg_x", "type": "message", "role": "assistant", "model": body["model"], "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1500, "output_tokens": 1,
                                                                  "cache_read_input_tokens": 1200}}})
        ev("content_block_start", {"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "thinking_delta", "thinking": "Counting barlines..."}})
        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "signature_delta", "signature": "sig123"}})
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        tid = f"toolu_{idx}_{len(body['messages'])}"
        ev("content_block_start", {"type": "content_block_start", "index": 1,
                                   "content_block": {"type": "tool_use", "id": tid, "name": "submit_transcription", "input": {}}})
        js = json.dumps(res)
        for k in range(0, len(js), 200):
            ev("content_block_delta", {"type": "content_block_delta", "index": 1,
                                       "delta": {"type": "input_json_delta", "partial_json": js[k:k + 200]}})
        ev("content_block_stop", {"type": "content_block_stop", "index": 1})
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                             "usage": {"output_tokens": 400}})
        ev("message_stop", {"type": "message_stop"})

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"problems": PROBLEMS, "consensus": SEEN["consensus"],
                                     "recount": SEEN["recount"], "image_counts": sorted(SEEN["images"]),
                                     "first_reads": SEEN["first"]}).encode())


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
