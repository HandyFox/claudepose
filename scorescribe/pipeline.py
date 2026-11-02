"""End-to-end pipeline: segment -> Claude reads each system -> check -> repair -> proofread -> MusicXML."""
from __future__ import annotations

import base64
import copy
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts, render
from .build import build_score, write_musicxml
from .imaging import SystemCrop, crop_systems, load_pages
from .gemini import GeminiBackend, QuotaExhausted, is_gemini
from .model import Doc, Issue, assemble, issues_by_system, normalize_parts

DEFAULT_MODEL = "claude-opus-5-5"


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _img_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(png)}}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, u):
        with self.lock:
            self.calls += 1
            self.input += getattr(u, "input_tokens", 0) or 0
            self.output += getattr(u, "output_tokens", 0) or 0
            self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def __str__(self):
        return (f"{self.calls} API calls, {self.input + self.cache_read + self.cache_write:,} input tokens "
                f"({self.cache_read:,} from cache), {self.output:,} output tokens")


class Session:
    """One conversation with Claude about one system image."""

    def __init__(self, crop: SystemCrop, png: bytes, zooms: list[bytes] | None = None):
        self.crop = crop
        self.png = png
        self.zooms = zooms or []
        self.messages: list[dict] = []
        self.last_tool_id: str | None = None
        self.result: dict | None = None
        self.recount_asked = False
        self.fresh = False  # read during this run (eligible for the consensus check)

    def images(self) -> list[dict]:
        return [_img_block(self.png)] + [_img_block(z) for z in self.zooms]


class Transcriber:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, effort: str | None = "high",
                 thinking: bool = True, workers: int = 4, repair_rounds: int = 2, proofread: bool = True,
                 passes: int = 2, zoom: bool = True, mark_uncertain: bool = False, shape_notes: str | None = None,
                 log=print):
        self.gemini = None
        self.client = None
        self.stopped: QuotaExhausted | None = None
        if is_gemini(model):
            safe_http_client(lambda *a: None, make_client=False)  # undo a broken ssl patch, if any
            self.gemini = GeminiBackend(api_key, model, effort, thinking, log)
        else:
            import anthropic
            kwargs = {}
            http_client = safe_http_client(log)
            if http_client is not None:
                kwargs["http_client"] = http_client
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), max_retries=6, **kwargs)
        self.model = model
        self.system_prompt = prompts.SYSTEM_PROMPT  # the song-from-recording mode swaps these
        self.tool = prompts.TOOL
        self.effort = effort
        self.thinking = thinking
        self.workers = max(1, workers)
        self.repair_rounds = repair_rounds
        self.proofread = proofread
        self.passes = max(1, min(2, passes))
        self.zoom = zoom
        self.mark_uncertain = mark_uncertain
        from .shapes import normalize
        self.shape_notes = normalize(shape_notes)
        self.log = log
        self.usage = Usage()
        self._fallback_plain = False

    # ------------------------------------------------------------ Claude I/O
    def _create(self, messages: list[dict]):
        if self.stopped:
            raise self.stopped
        if self.gemini is not None:
            try:
                msg = self.gemini.create(messages, self.system_prompt, self.tool["input_schema"])
            except QuotaExhausted as e:
                self.stopped = e
                raise
            self.usage.add(msg.usage)
            return msg
        kwargs = dict(
            model=self.model,
            max_tokens=32000,
            system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=[self.tool],
            messages=_with_cache_points(messages),
        )
        if self.thinking and not self._fallback_plain:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["tool_choice"] = {"type": "auto"}
            if self.effort:
                kwargs["output_config"] = {"effort": self.effort}
        else:
            kwargs["tool_choice"] = {"type": "tool", "name": self.tool["name"]}
        import anthropic
        try:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e).lower():
                self.stopped = QuotaExhausted("Your Anthropic credit balance is too low. Everything finished so far is "
                                              "saved; add credit (or switch to a Gemini model) and resume.")
                raise self.stopped from e
            if not self._fallback_plain and ("thinking" in str(e) or "effort" in str(e) or "output_config" in str(e)):
                self.log(f"  model rejected thinking/effort settings ({e.message if hasattr(e, 'message') else e}); "
                         "continuing without them")
                self._fallback_plain = True
                return self._create(messages)
            raise
        self.usage.add(msg.usage)
        return msg

    def _ask(self, sess: Session, user_content: list[dict]) -> dict:
        """Send a user turn, return the submitted transcription dict."""
        sess.messages.append({"role": "user", "content": user_content})
        for attempt in range(3):
            msg = self._create(sess.messages)
            content = [b.model_dump(exclude_none=True) for b in msg.content]
            sess.messages.append({"role": "assistant", "content": content})
            tool = next((b for b in msg.content if b.type == "tool_use" and b.name == self.tool["name"]), None)
            if tool is not None and isinstance(tool.input, dict) and tool.input.get("measures") is not None:
                sess.last_tool_id = tool.id
                return tool.input
            # no tool call: try JSON in text, otherwise nudge
            text = "".join(b.text for b in msg.content if b.type == "text")
            parsed = _json_from_text(text)
            if parsed and parsed.get("measures"):
                sess.last_tool_id = None
                return parsed
            if tool is not None:
                sess.last_tool_id = tool.id
                nudge = [{"type": "tool_result", "tool_use_id": tool.id, "is_error": True,
                          "content": "The submission had no 'measures'. Submit the complete transcription."}]
            else:
                sess.last_tool_id = None
                nudge = [{"type": "text", "text": f"Please call {self.tool['name']} with the complete result now."}]
            sess.messages.append({"role": "user", "content": nudge})
        raise RuntimeError(f"system {sess.crop.index}: Claude did not return a transcription")

    def _reply_content(self, sess: Session, text: str, extra: list[dict] | None = None) -> list[dict]:
        blocks: list[dict] = []
        if sess.last_tool_id:
            blocks.append({"type": "tool_result", "tool_use_id": sess.last_tool_id, "content": "Received."})
        blocks.extend(extra or [])
        blocks.append({"type": "text", "text": text})
        return blocks

    # ------------------------------------------------------------- stages
    def first_pass(self, sess: Session, total: int, parts, hint) -> dict:
        hint = dict(hint or {})
        hint["zooms"] = len(sess.zooms)
        hint["measures"] = sess.crop.measure_count
        text = prompts.first_message(sess.crop.index, total, sess.crop.n_staves, parts, hint)
        return self._ask(sess, sess.images() + [{"type": "text", "text": text}])

    def ensure_thread(self, sess: Session, total: int, parts):
        """Resumed from cache: rebuild a minimal conversation so follow-ups have context."""
        if sess.messages:
            return
        text = prompts.first_message(sess.crop.index, total, sess.crop.n_staves, parts,
                                     {"zooms": len(sess.zooms), "measures": sess.crop.measure_count})
        sess.messages = [
            {"role": "user", "content": sess.images() + [{"type": "text", "text": text}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier transcription:\n```json\n"
                                               + json.dumps({k: v for k, v in sess.result.items() if not k.startswith("_")},
                                                            ensure_ascii=False) + "\n```"}]},
        ]
        sess.last_tool_id = None

    def repair(self, sess: Session, errors: list[str], warnings: list[str], recount: str | None = None) -> dict:
        return self._ask(sess, self._reply_content(sess, prompts.repair_message(errors, warnings, recount)))

    def resolve(self, sess: Session, diffs: list[str], other_json: str | None) -> dict:
        return self._ask(sess, self._reply_content(sess, prompts.consensus_message(diffs, other_json)))

    def proof(self, sess: Session, engraving: bytes | None) -> dict:
        if engraving:
            return self._ask(sess, self._reply_content(sess, prompts.PROOFREAD_MESSAGE, [_img_block(engraving)]))
        return self._ask(sess, self._reply_content(sess, prompts.PROOFREAD_TEXT_MESSAGE))

    # ------------------------------------------------------------ driver
    def run(self, inputs: list[str], workdir: str, out_path: str, parts_hint: list[dict] | None = None,
            staves_per_system: int | None = None, dpi: int = 400, whole_pages: bool = False,
            resume: bool = True, draft: str | None = None) -> dict:
        wd = Path(workdir)
        (wd / "systems").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.log("Loading pages ...")
        pages = load_pages(inputs, dpi=dpi)
        self.log(f"  {len(pages)} page(s)")
        self.log("Finding staves and systems ...")
        crops, overviews = crop_systems(pages, staves_per_system=staves_per_system, whole_pages=whole_pages, log=self.log)
        for i, ov in enumerate(overviews, start=1):
            ov.convert("RGB").save(wd / f"page{i:02d}_systems.jpg", quality=80)
        total = len(crops)
        self.log(f"  {total} systems (see page*_systems.jpg in the work folder to check the split)")

        sessions = []
        for c in crops:
            png = c.png_bytes()
            (wd / "systems" / f"sys{c.index:03d}.png").write_bytes(png)
            zooms = c.zoom_pngs() if self.zoom else []
            for k, z in enumerate(zooms, start=1):
                (wd / "systems" / f"sys{c.index:03d}_zoom{k}.png").write_bytes(z)
            s = Session(c, png, zooms)
            cached = wd / "systems" / f"sys{c.index:03d}.json"
            if resume and cached.exists():
                try:
                    s.result = json.loads(cached.read_text())
                except json.JSONDecodeError:
                    s.result = None
            sessions.append(s)

        def save(s: Session):
            (wd / "systems" / f"sys{s.crop.index:03d}.json").write_text(json.dumps(s.result, indent=1, ensure_ascii=False))

        # ---- optional draft from another OMR engine (Audiveris etc.)
        drafts: dict[int, str] = {}
        if draft:
            from .draft import score_to_systems, system_text
            dmeta, dsys = score_to_systems(draft)
            if len(dsys) == total:
                drafts = {i + 1: system_text(d) for i, d in enumerate(dsys)}
                parts_hint = parts_hint or dmeta["parts"]
                self.log(f"  using draft {Path(draft).name}: {sum(len(d['measures']) for d in dsys)} measures, layout "
                         + ", ".join(p['name'] for p in dmeta['parts']))
            else:
                self.log(f"  draft has {len(dsys)} systems but {total} were detected in the images; "
                         "can't line them up, so the draft is ignored (try --staves-per-system)")

        # ---- layout: from the user, the cache, or the first system
        parts = normalize_parts(parts_hint) if parts_hint else None
        meta = {}
        meta_path = wd / "score.json"
        if resume and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            parts = parts or meta.get("parts")
        if parts is None:
            s0 = sessions[0]
            if s0.result is None:
                self.log(f"Reading system 1/{total} (working out the instrument layout) ...")
                s0.result = self.first_pass(s0, total, None, {"draft": drafts.get(1)})
                s0.fresh = True
                save(s0)
            parts = normalize_parts(s0.result.get("parts"))
        meta["parts"] = parts
        for s in sessions:
            if s.result:
                meta.setdefault("title", s.result.get("title"))
                meta.setdefault("composer", s.result.get("composer"))
        meta_path.write_text(json.dumps(meta, indent=1))
        self.log("  layout: " + ", ".join(f"{p['name']} ({p['staves']} {'staves' if p['staves'] > 1 else 'staff'})" for p in parts))

        # ---- first pass on everything not yet done (plus independent second readings).
        # Each system records the stages it has finished in "_done", so a run that was
        # paused (quota, credit, closed window) picks up exactly where it stopped.
        todo = [s for s in sessions if s.result is None]
        seconds: dict[int, Session] = {}
        if self.passes > 1:
            for s in sessions:
                if "settled" in _done(s):
                    continue
                b = Session(s.crop, s.png, s.zooms)
                f2 = wd / "systems" / f"sys{s.crop.index:03d}_second.json"
                if resume and f2.exists():
                    try:
                        b.result = json.loads(f2.read_text())
                    except json.JSONDecodeError:
                        pass
                seconds[s.crop.index] = b
        batch = todo + [b for b in seconds.values() if b.result is None]
        if batch:
            n2 = len(batch) - len(todo)
            what = f"{len(todo)} system(s)" + (f" + {n2} independent second reading(s)" if n2 else "")
            self.log(f"Reading {what} with {self.model} ({self.workers} at a time) ...")
            main_ids = {id(s) for s in todo}

            def save_any(s):
                if id(s) in main_ids:
                    save(s)
                else:
                    (wd / "systems" / f"sys{s.crop.index:03d}_second.json").write_text(
                        json.dumps(s.result, indent=1, ensure_ascii=False))
            self._parallel(batch, lambda s: self.first_pass(s, total, parts, {"time": self._time_hint(sessions, s),
                                                                               "draft": drafts.get(s.crop.index)}),
                           save_any, "read", quiet=lambda s: id(s) not in main_ids)

        # ---- consensus: compare the two readings; the model settles only the disagreements
        if seconds:
            todo_res = []
            agree = 0
            for s in sessions:
                b = seconds.get(s.crop.index)
                if not (s.result and b and b.result):
                    continue
                diffs, other = diff_readings(s.result, b.result)
                if diffs:
                    todo_res.append((s, diffs, other))
                else:
                    agree += 1
                    s.result["_done"] = sorted(set(_done(s)) | {"settled"})
                    save(s)
            self.log(f"Comparing readings: {agree} system(s) agreed exactly, {len(todo_res)} had disagreements")
            if todo_res:
                lookup = {id(s): (d, o) for s, d, o in todo_res}

                def settle(s):
                    d, o = lookup[id(s)]
                    return self.resolve(s, d, o)
                self._parallel([s for s, _, _ in todo_res], settle, save, "settle disagreements", mark="settled")

        doc = self._assemble(sessions, parts, meta)

        # ---- repair loop
        for rnd in range(1, self.repair_rounds + 1):
            by_sys = issues_by_system(doc)
            recounts = {s.crop.index: _recount(s) for s in sessions if not s.recount_asked and _recount(s)}
            bad = [s for s in sessions
                   if any(i.level == "error" for i in by_sys.get(s.crop.index, [])) or s.crop.index in recounts]
            if not bad:
                break
            self.log(f"Repair round {rnd}: {len(bad)} system(s) failed checks ...")

            def fix(s, by_sys=by_sys, recounts=recounts):
                self.ensure_thread(s, total, parts)
                iss = by_sys.get(s.crop.index, [])
                errs = [f"{i.where()}: {i.msg}" for i in iss if i.level == "error"]
                warns = [f"{i.where()}: {i.msg}" for i in iss if i.level == "warn"]
                rc = recounts.get(s.crop.index)
                if rc:
                    s.recount_asked = True
                return self.repair(s, errs, warns, rc)
            self._parallel(bad, fix, save, "repair")
            doc = self._assemble(sessions, parts, meta)

        # ---- proofreading pass (engraving vs original)
        if self.proofread:
            use_img = render.available()
            self.log(f"Proofreading {total} system(s) "
                     + ("against an engraving of the transcription" if use_img
                        else f"(text only - for visual proofreading run: {render.missing_hint()})")
                     + " ...")
            snapshot = doc

            def proof(s, snapshot=snapshot):
                self.ensure_thread(s, total, parts)
                eng = None
                if use_img:
                    try:
                        eng = render_system(snapshot, s.crop.index, s.result)
                    except Exception as e:  # rendering is best effort
                        self.log(f"  system {s.crop.index}: could not engrave ({e}); proofreading from text")
                return self.proof(s, eng)
            changed = self._parallel([s for s in sessions if s.result and "proofread" not in _done(s)],
                                     proof, save, "proofread", mark="proofread")
            self.log(f"  {changed} system(s) changed during proofreading")
            doc = self._assemble(sessions, parts, meta)
            by_sys = issues_by_system(doc)
            bad = [s for s in sessions if any(i.level == "error" for i in by_sys.get(s.crop.index, []))]
            if bad and self.repair_rounds:
                self.log(f"Final repair: {len(bad)} system(s) ...")

                def fix2(s, by_sys=by_sys):
                    iss = by_sys.get(s.crop.index, [])
                    return self.repair(s, [f"{i.where()}: {i.msg}" for i in iss if i.level == "error"],
                                       [f"{i.where()}: {i.msg}" for i in iss if i.level == "warn"])
                self._parallel(bad, fix2, save, "repair")
                doc = self._assemble(sessions, parts, meta)

        for s in sessions:
            rc = _recount(s)
            if rc:
                doc.issues.append(Issue(s.crop.index, 0, None, None, "warn",
                                        f"has {len(s.result.get('measures') or [])} measures but barline detection counted "
                                        f"{s.crop.measure_count} - check this system for a missing or extra measure"))
        return finish(doc, wd, out_path, self.log, t0, str(self.usage), self.mark_uncertain, self.shape_notes)

    # ------------------------------------------------------------ helpers
    def _time_hint(self, sessions: list[Session], s: Session) -> str | None:
        for prev in reversed([p for p in sessions if p.crop.index < s.crop.index]):
            if prev.result:
                for m in reversed(prev.result.get("measures") or []):
                    if m.get("time"):
                        return m["time"]
        return None

    def _parallel(self, sessions: list[Session], fn, save, label: str, quiet=lambda s: False,
                  mark: str | None = None) -> int:
        changed = 0
        lock = threading.Lock()

        def work(s):
            nonlocal changed
            before = json.dumps(s.result, sort_keys=True) if s.result else None
            if self.stopped:
                return
            try:
                res = fn(s)
            except QuotaExhausted:
                return
            except Exception as e:
                self.log(f"  system {s.crop.index}: {label}{' (second reading)' if quiet(s) else ''} failed: {e}")
                return
            if not res.get("parts") and s.result and s.result.get("parts"):
                res["parts"] = s.result["parts"]
            done = set(_done(s)) | ({mark} if mark else set())
            if done:
                res["_done"] = sorted(done)
            s.result = res
            save(s)
            ch = (res.get("changes") or "").strip()
            if quiet(s):
                return
            with lock:
                if before is not None and json.dumps({k: v for k, v in res.items() if k != "changes" and not k.startswith("_")},
                                                     sort_keys=True) != \
                        json.dumps({k: v for k, v in json.loads(before).items() if k != "changes" and not k.startswith("_")},
                                   sort_keys=True):
                    changed += 1
                note = f" - {ch[:140]}" if ch and ch.lower() != "none" else ""
                self.log(f"  system {s.crop.index}: {label} done ({len(res.get('measures') or [])} measures){note}")

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(as_completed([ex.submit(work, s) for s in sessions]))
        if self.stopped:
            raise self.stopped
        return changed

    def _assemble(self, sessions, parts, meta) -> Doc:
        return assemble([s.result or {} for s in sessions], parts, meta.get("title"), meta.get("composer"))


def safe_http_client(log=print, make_client: bool = True):
    """Work around Python's ssl module having been patched by another package.

    Some setups replace ``ssl.SSLContext`` at startup (for example an add-on that
    makes Python use the Windows certificate store via ``truststore``). The
    Anthropic SDK uses its own copy of truststore, and a second copy layered on a
    patched ``ssl.SSLContext`` recurses forever ("RecursionError ... verify_mode"),
    which surfaces as "Connection error". When that patch is detected, restore the
    original class for this process and give the SDK an ordinary TLS context that
    trusts the operating system's certificates (plus certifi's, if installed).
    Returns None when nothing needs fixing, so the SDK's defaults are used.
    """
    import ssl

    orig = next((c for c in ssl.SSLContext.__mro__ if c.__module__ == "ssl" and c.__name__ == "SSLContext"), None)
    if orig is None or ssl.SSLContext is orig:
        return None
    patched_by = ssl.SSLContext.__module__
    ssl.SSLContext = orig
    ctx = orig(ssl.PROTOCOL_TLS_CLIENT)  # verifies certificates and host names by default
    ctx.load_default_certs()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(var) and os.path.isfile(os.environ[var]):
            try:
                ctx.load_verify_locations(os.environ[var])
            except Exception:
                pass
    log(f"  note: Python's SSL module had been replaced by '{patched_by}'; using a standard secure connection instead")
    if not make_client:
        return None
    import anthropic
    return anthropic.DefaultHttpxClient(verify=ctx)


def _with_cache_points(messages: list[dict]) -> list[dict]:
    """Mark the first and the latest user turn for prompt caching.

    Follow-up calls (consensus, repair, proofread) resend the whole conversation,
    images included; with these cache points they are read from cache at a
    fraction of the price instead of being paid for again.
    """
    out = list(messages)
    idxs = [i for i, m in enumerate(out) if m["role"] == "user"]
    for i in {idxs[0], idxs[-1]} if idxs else set():
        m = out[i]
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        if not content:
            continue
        content = list(content)
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        out[i] = {**m, "content": content}
    return out


def _done(s: Session) -> list[str]:
    return list((s.result or {}).get("_done") or [])


def _recount(s: Session) -> str | None:
    det = s.crop.measure_count
    got = len((s.result or {}).get("measures") or [])
    if det and got and det != got:
        return prompts.recount_message(det, got)
    return None


def _norm_voice(v) -> str:
    return " ".join(str(v).split())


def diff_readings(a: dict, b: dict, limit: int = 60) -> tuple[list[str], str | None]:
    """Measure-by-measure differences between two readings of one system."""
    ma, mb = a.get("measures") or [], b.get("measures") or []
    if len(ma) != len(mb):
        return ([f"- MEASURE COUNT: you have {len(ma)} measures, the other reading has {len(mb)}."],
                json.dumps(mb, ensure_ascii=False, separators=(",", ":")))
    out = []
    for k, (x, y) in enumerate(zip(ma, mb), start=1):
        for fld in ("time", "key", "incomplete", "barline", "repeat_start", "volta"):
            vx, vy = x.get(fld), y.get(fld)
            if fld in ("incomplete", "repeat_start"):
                vx, vy = bool(vx), bool(vy)
            if fld == "barline":
                vx, vy = vx or "regular", vy or "regular"
            if vx != vy and not (fld == "key" and (vx is None or vy is None)):
                out.append(f"- measure {k}: `{fld}` yours={vx!r} other={vy!r}")
        sx = {st.get("staff"): st for st in x.get("staves") or []}
        sy = {st.get("staff"): st for st in y.get("staves") or []}
        for staff in sorted(set(sx) | set(sy), key=lambda v: (v is None, v)):
            p, q = sx.get(staff), sy.get(staff)
            if p is None or q is None:
                out.append(f"- measure {k}, staff {staff}: only {'yours' if q is None else 'the other reading'} has this staff")
                continue
            if p.get("clef") and q.get("clef") and p.get("clef") != q.get("clef"):
                out.append(f"- measure {k}, staff {staff}: clef yours={p.get('clef')} other={q.get('clef')}")
            vp = [_norm_voice(v) for v in (p.get("voices") or [])]
            vq = [_norm_voice(v) for v in (q.get("voices") or [])]
            if vp != vq:
                out.append(f"- measure {k}, staff {staff}:\n    yours: " + " || ".join(vp) + "\n    other: " + " || ".join(vq))
    if len(out) > limit:
        return out[:limit] + [f"- ... and {len(out) - limit} more differences"], json.dumps(mb, ensure_ascii=False, separators=(",", ":"))
    return out, None


def _json_from_text(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def render_system(doc: Doc, system_index: int, raw: dict | None = None) -> bytes:
    """Engrave one system exactly as transcribed from its image (measures numbered from 1).

    Uses the raw transcription of that image (not the version with cut-off measures
    joined to the neighbouring image), with the key, time and clefs in effect before it.
    """
    if raw is not None:
        before = [m for m in doc.measures if m.system < system_index]
        raw = copy.deepcopy(raw)
        ms_raw = raw.get("measures") or []
        if not ms_raw:
            raise ValueError("no measures")
        if before:
            ctx = before[-1]
            m0 = ms_raw[0]
            m0.setdefault("time", {"4/4|common": "C", "2/2|cut": "C|"}.get(ctx.eff_time[0], ctx.eff_time[0]))
            if m0.get("key") in (None, ""):
                m0["key"] = ctx.eff_key
            for st in m0.get("staves") or []:
                try:
                    st.setdefault("clef", ctx.eff_clefs.get(int(st.get("staff"))))
                except (TypeError, ValueError):
                    pass
        ms = assemble([raw], doc.parts, stitch=False).measures
    else:
        ms = [copy.deepcopy(m) for m in doc.measures if m.system == system_index]
    if not ms:
        raise ValueError("no measures")
    first = ms[0]
    first.time = first.eff_time
    first.key = first.eff_key
    for st, S in first.staves.items():
        S.clef = first.eff_clefs.get(st, S.clef)
    for i, m in enumerate(ms, start=1):
        m.number = i
        m.flagged = m.uncertain = False
        m.flag_staves, m.unsure_staves = set(), set()
        m.tempo_text = m.tempo_bpm = None
    sub = Doc(None, None, doc.parts, ms, [])
    score = build_score(sub)
    with tempfile.TemporaryDirectory() as td:
        p = score.write("musicxml", fp=os.path.join(td, "s.musicxml"))
        xml = Path(p).read_text()
    xml = xml.replace("<work-title>Transcribed score</work-title>", "")
    pngs = render.musicxml_to_pngs(xml, width_px=2400, page_width=2600, scale=50, one_system_per_page=True)
    return render.stack_pngs(pngs)


def finish(doc: Doc, wd: Path, out_path: str, log, t0: float, usage: str = "", mark_uncertain: bool = False,
           shape_notes: str | None = None) -> dict:
    out = write_musicxml(doc, out_path, mark_uncertain=mark_uncertain, shape_notes=shape_notes)
    errors = [i for i in doc.issues if i.level == "error"]
    unsure = [i for i in doc.issues if i.level == "unsure"]
    warns = [i for i in doc.issues if i.level == "warn"]
    mnum = {(m.system, m.idx): m.number for m in doc.measures}
    lines = ["# Transcription report", "",
             f"Output: {out}", f"Measures: {len(doc.measures)}", "Parts: " + ", ".join(p['name'] for p in doc.parts), ""]
    if usage:
        lines += [f"Usage: {usage}", ""]
    for title, items in (("Measures still failing checks (coloured RED in MuseScore)", errors),
                         ("Measures the reader was unsure about" + (" (coloured ORANGE)" if mark_uncertain else
                                                                 " - worth a listen"), unsure),
                         ("Warnings", warns)):
        lines.append(f"## {title}: {len(items)}")
        for i in items:
            n = mnum.get((i.system, i.measure))
            loc = f"m. {n}" if n is not None else f"system {i.system}"
            lines.append(f"- {loc} (system {i.system})" + (f", staff {i.staff}" if i.staff else "") + f": {i.msg}")
        lines.append("")
    (wd / "report.md").write_text("\n".join(lines))
    log("")
    log(f"Done in {time.time() - t0:.0f}s. {len(doc.measures)} measures written to {out}")
    if usage:
        log(f"  {usage}")
    flagged = sorted({mnum[(i.system, i.measure)] for i in errors if (i.system, i.measure) in mnum})
    unsure_m = sorted({mnum[(i.system, i.measure)] for i in unsure if (i.system, i.measure) in mnum})
    if flagged:
        log(f"  red (failed checks): measures {', '.join(map(str, flagged))}")
    if unsure_m:
        log(f"  Unsure about: measures {', '.join(map(str, unsure_m))}"
            + (" (orange)" if mark_uncertain else " (listed in the report)"))
    log(f"  full report: {wd / 'report.md'}")
    return {"output": out, "measures": len(doc.measures), "red": flagged, "orange": unsure_m,
            "report": str(wd / "report.md")}


def rebuild(workdir: str, out_path: str, log=print, mark_uncertain: bool = False, shape_notes: str | None = None) -> dict:
    """Rebuild MusicXML from the cached (possibly hand-edited) JSON in a work folder; no API calls."""
    wd = Path(workdir)
    files = sorted((wd / "systems").glob("sys[0-9][0-9][0-9].json"))  # not the *_second.json readings
    systems = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if (wd / "score.json").exists():
        meta = json.loads((wd / "score.json").read_text())
    else:  # agent-mode folder: layout from the manifest or from system 1
        man = {}
        if (wd / "scorescribe-manifest.json").exists():
            man = json.loads((wd / "scorescribe-manifest.json").read_text())
            by_num = {int(f.stem[3:]): d for f, d in zip(files, systems)}
            systems = [by_num.get(s["system"], {}) for s in man.get("systems", [])]
        parts = man.get("parts") or next((x.get("parts") for x in systems if x and x.get("parts")), None)
        if not parts:
            raise ValueError("no instrument layout: add `parts` to systems/sys001.json")
        meta = {"parts": parts, "title": next((x.get("title") for x in systems if x and x.get("title")), None),
                "composer": next((x.get("composer") for x in systems if x and x.get("composer")), None)}
    doc = assemble(systems, meta["parts"], meta.get("title"), meta.get("composer"))
    return finish(doc, wd, out_path, log, time.time(), mark_uncertain=mark_uncertain, shape_notes=shape_notes)
