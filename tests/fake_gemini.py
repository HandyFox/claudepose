"""Fake Gemini generateContent endpoint for end-to-end tests.

Usage: python fake_gemini.py PORT TRUTH.musicxml CHUNKS [DAILY_LIMIT]
Answers each system from ground truth; system 3's second reading misreads a pitch
(consensus), system 2's first request gets a per-minute 429 (wait + retry), and after
DAILY_LIMIT requests every call gets a per-day 429 (pause). Checks request shape.
"""
import copy
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scorescribe.draft import score_to_systems  # noqa: E402

port, truth, chunks = int(sys.argv[1]), sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
daily = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
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
LOCK = threading.Lock()
S = {"requests": 0, "first": {}, "throttled": False, "problems": [], "daily_hits": 0, "kinds": []}


def problem(m):
    with LOCK:
        if m not in S["problems"]:
            S["problems"].append(m)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if ":generateContent" not in self.path or "/models/gemini-" not in self.path:
            problem(f"unexpected path {self.path}")
        if not self.headers.get("x-goog-api-key"):
            problem("no api key header")
        with LOCK:
            S["requests"] += 1
            n = S["requests"]
        if n > daily:
            S["daily_hits"] += 1
            return self.send(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                             "message": "You exceeded your current quota.",
                                             "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                          "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        gc = body.get("generationConfig", {})
        si = json.dumps(body.get("systemInstruction", {}))
        if "There is no tool to call here" not in si:
            problem("system instruction missing the JSON-output note")
        if gc.get("responseMimeType") != "application/json":
            problem(f"responseMimeType {gc.get('responseMimeType')}")
        if "measures" not in json.dumps(gc.get("responseJsonSchema", {})):
            problem("no responseJsonSchema")
        tc = gc.get("thinkingConfig") or {}
        if str(tc.get("thinkingLevel", tc.get("thinking_level", ""))).lower() != "high":
            problem(f"thinking {gc.get('thinkingConfig')}")
        if "HIGH" not in str(gc.get("mediaResolution", "")):
            problem(f"mediaResolution {gc.get('mediaResolution')}")
        contents = body["contents"]
        first = contents[0]["parts"]
        inl = first[0].get("inlineData") or first[0].get("inline_data") or {}
        if (inl.get("mimeType") or inl.get("mime_type")) != "image/png":
            problem("first part is not a PNG image")
        text = next(p["text"] for p in reversed(first) if "text" in p)
        idx = int(text.split("This is system ")[1].split(" ")[0])
        for k, c in enumerate(contents):
            if c.get("role") == "model":
                sig = [p.get("thoughtSignature") or p.get("thought_signature") for p in c["parts"]
                       if p.get("thoughtSignature") or p.get("thought_signature")]
                resumed = any(str(p.get("text", "")).startswith("Earlier transcription") for p in c["parts"])
                if not sig and not resumed:
                    problem("model turn sent back without its thought signature")
        last = "".join(p.get("text", "") for p in contents[-1]["parts"])
        res = copy.deepcopy(TRUTH[idx - 1])
        kind = "proof"
        if len(contents) == 1:
            kind = "first"
            with LOCK:
                if idx == 2 and not S["throttled"]:
                    S["throttled"] = True
                    return self.send(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                     "message": "Resource has been exhausted (e.g. check quota).",
                                                     "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                                  "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                                                 {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}]}})
                c = S["first"][idx] = S["first"].get(idx, 0) + 1
            if idx == 3 and c == 2:
                v = res["measures"][1]["staves"][0]["voices"][0]
                res["measures"][1]["staves"][0]["voices"][0] = v.replace("A5", "B5", 1)
        elif "second, independent reading" in last:
            kind = "consensus"
            res["changes"] = "kept the first reading"
        elif "Automatic checks" in last:
            kind = "repair"
        with LOCK:
            S["kinds"].append((idx, kind))
        out = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": json.dumps(res), "thoughtSignature": "c2lnLXRlc3Q="}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5000, "candidatesTokenCount": 600, "thoughtsTokenCount": 900,
                              "totalTokenCount": 6500}}
        self.send(200, out)

    def do_GET(self):
        self.send(200, S)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
