"""Document extractor: structured JSON from invoices and resumes with Gemini or Claude.

  python docextract.py extract samples/invoices/northwind-office.txt --type invoice --model gemini-3.8-flash
  python docextract.py compare samples
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from scoring import field_errors, score

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Party(BaseModel):
    name: Optional[str]
    address: Optional[str] = Field(description="Full address on one line, parts separated by commas")


class LineItem(BaseModel):
    description: str
    quantity: Optional[float]
    unit_price: Optional[float]
    amount: float


class Invoice(BaseModel):
    invoice_number: Optional[str]
    invoice_date: Optional[str] = Field(description="YYYY-MM-DD")
    due_date: Optional[str] = Field(description="YYYY-MM-DD; null if not stated")
    currency: Optional[str] = Field(description="ISO 4217 code, e.g. USD, EUR")
    vendor: Party
    vendor_tax_id: Optional[str] = Field(description="VAT / GST / EIN of the vendor, as written")
    bill_to: Party
    line_items: list[LineItem]
    subtotal: Optional[float]
    tax: Optional[float] = Field(description="Total tax amount, not the rate")
    shipping: Optional[float]
    total: Optional[float] = Field(description="Amount due")
    payment_terms: Optional[str] = Field(description="e.g. 'Net 30'")


class Job(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = Field(description="YYYY-MM")
    end_date: Optional[str] = Field(description="YYYY-MM; null if this is the current role")
    location: Optional[str]


class Education(BaseModel):
    institution: str
    degree: Optional[str] = Field(description="Degree as written, e.g. 'B.S.', 'MBA'")
    field: Optional[str] = Field(description="Field of study / major")
    graduation_year: Optional[int]


class Resume(BaseModel):
    name: str
    email: Optional[str]
    phone: Optional[str]
    location: Optional[str] = Field(description="City, region/country of the candidate")
    skills: list[str] = Field(description="Each skill from the skills section as written, one per item")
    experience: list[Job]
    education: list[Education]


SCHEMAS: dict[str, type[BaseModel]] = {"invoice": Invoice, "resume": Resume}

# --------------------------------------------------------------------------
# Models and list prices (USD per 1M tokens, standard paid tier).
# Gemini output price includes thinking tokens. Free-tier keys pay nothing, but
# the table reports list price so models are comparable.
# --------------------------------------------------------------------------

MODELS: dict[str, dict[str, float]] = {
    "gemini-4-hydrogen": {"in": 1.50, "out": 7.50, "cache_read": 0.075},
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75, "cache_read": 0.075},
    "gemini-3.6-flash": {"in": 0.75, "out": 3.75, "cache_read": 0.075},
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00, "cache_read": 0.05},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50, "cache_read": 0.03},
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.10},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01},
}
# All three work on the Gemini free tier, which allows about 20 requests per model
# per day (a full benchmark uses 6 per model). Others, e.g. gemini-3.8-flash,
# can be passed with --models.
GEMINI_DEFAULT = ["gemini-4-hydrogen", "gemini-3-flash-preview", "gemini-3.5-flash-lite"]
CLAUDE_DEFAULT = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]

SYSTEM = (
    "You extract structured data from business documents. Copy values exactly as they "
    "appear in the document, normalising only dates and numbers to the formats the schema "
    "asks for. Numbers are plain numbers with no currency symbols or thousands separators. "
    "Use null for any field the document does not contain; never guess or compute a value "
    "that is not printed."
)

MEDIA_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def load_doc(path: Path) -> tuple[str | None, str | None, str | None]:
    """(text, media_type, base64 data) — text for text files, bytes otherwise."""
    ext = path.suffix.lower()
    if ext in MEDIA_TYPES:
        return None, MEDIA_TYPES[ext], base64.standard_b64encode(path.read_bytes()).decode()
    return path.read_text(encoding="utf-8"), None, None


def price(model: str) -> dict[str, float]:
    return MODELS.get(model, {"in": float("nan"), "out": float("nan"), "cache_read": float("nan")})


# --------------------------------------------------------------------------
# Gemini (REST, stdlib only)
# --------------------------------------------------------------------------

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
THINKING_LEVEL = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY to use Gemini models.")
    return key


def gemini_schema(model_cls: type[BaseModel]) -> dict:
    """Pydantic JSON Schema -> Gemini responseSchema (OpenAPI subset: no $ref, nullable flag)."""
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})

    def conv(s: dict) -> dict:
        if "$ref" in s:
            return conv(defs[s["$ref"].rsplit("/", 1)[-1]])
        if "anyOf" in s:
            options = [x for x in s["anyOf"] if x.get("type") != "null"]
            out = conv(options[0])
            if len(options) < len(s["anyOf"]):
                out["nullable"] = True
            if "description" in s:
                out["description"] = s["description"]
            return out
        out: dict = {"type": s["type"]}
        if "description" in s:
            out["description"] = s["description"]
        if s["type"] == "object":
            props = s.get("properties", {})
            out["properties"] = {k: conv(v) for k, v in props.items()}
            out["required"] = list(props)
            out["propertyOrdering"] = list(props)
        elif s["type"] == "array":
            out["items"] = conv(s["items"])
        return out

    return conv(schema)


def _post(url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def _retry_delay(err_body: str, attempt: int) -> float:
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', err_body)
    return min(float(m.group(1)) + 1 if m else 2 ** attempt * 5, 65)


def extract_gemini(model: str, path: Path, doc_type: str, effort: str) -> dict:
    text, media_type, data = load_doc(path)
    ask = {"text": f"Extract the {doc_type} fields from this document."}
    doc_part = {"text": f"<document>\n{text}\n</document>"} if text is not None else \
        {"inlineData": {"mimeType": media_type, "data": data}}
    gen_cfg = {
        "responseMimeType": "application/json",
        "responseSchema": gemini_schema(SCHEMAS[doc_type]),
        "thinkingConfig": {"thinkingLevel": THINKING_LEVEL[effort]},
    }
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [doc_part, ask]}],
            "generationConfig": gen_cfg}
    url, key = GEMINI_URL.format(model=model), gemini_key()

    # Network/server failures are retried; if they persist the result is flagged
    # "transport" so it counts as an error but isn't scored as a wrong extraction.
    notes, attempts = [], 7
    for attempt in range(attempts):
        t0 = time.perf_counter()
        try:
            resp = _post(url, body, key)
            latency = time.perf_counter() - t0  # only the successful attempt counts
            break
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            if e.code == 400 and "thinking" in err.lower() and "thinkingConfig" in gen_cfg:
                gen_cfg.pop("thinkingConfig")  # model doesn't accept this thinking level
                notes.append("thinking level not supported; ran with model default")
                continue
            msg = re.search(r'"message":\s*"([^"]+)"', err)
            daily_quota = e.code == 429 and "per day" in err.lower()  # resets in hours: don't wait
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < attempts - 1 and not daily_quota:
                time.sleep(_retry_delay(err, attempt))
                continue
            text = (msg.group(1) if msg else err[:200]).split("\\n")[0]
            if daily_quota:
                text = "daily free-tier quota used up for this model"
            # Any HTTP error means the model never answered (bad model name, quota,
            # outage...), so it's reported as an error but not scored.
            return {"error": f"HTTP {e.code}: {text}", "latency_s": 0.0, "transport": True}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt * 3, 60))
                continue
            return {"error": f"connection error: {e}", "latency_s": 0.0, "transport": True}

    um = resp.get("usageMetadata", {})
    prompt_toks = um.get("promptTokenCount", 0)
    cached = um.get("cachedContentTokenCount", 0)
    out_toks = um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
    p = price(model)
    result = {
        "latency_s": latency,
        "input_tokens": prompt_toks,
        "output_tokens": out_toks,
        "thinking_tokens": um.get("thoughtsTokenCount", 0),
        "cost_usd": ((prompt_toks - cached) * p["in"] + cached * p["cache_read"] + out_toks * p["out"]) / 1e6,
    }
    if notes:
        result["notes"] = notes
    cands = resp.get("candidates") or []
    if not cands:
        result["error"] = f"blocked: {resp.get('promptFeedback', {}).get('blockReason', 'no candidates')}"
        return result
    cand = cands[0]
    result["stop_reason"] = cand.get("finishReason")
    raw = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    try:
        result["data"] = SCHEMAS[doc_type].model_validate_json(raw).model_dump()
    except ValidationError as e:
        result["error"] = f"invalid JSON for schema (finishReason={cand.get('finishReason')}): {str(e)[:200]}"
    return result


# --------------------------------------------------------------------------
# Claude (Anthropic SDK)
# --------------------------------------------------------------------------

_anthropic_client = None


def init_claude_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("Claude models need the Anthropic SDK: pip install anthropic")

    global _anthropic_client
    _anthropic_client = _anthropic_client or anthropic.Anthropic()
    return _anthropic_client


def extract_claude(model: str, path: Path, doc_type: str, effort: str) -> dict:
    init_claude_client()
    import anthropic
    text, media_type, data = load_doc(path)
    ask = {"type": "text", "text": f"Extract the {doc_type} fields from this document."}
    if text is not None:
        content = [{"type": "text", "text": f"<document>\n{text}\n</document>"}, ask]
    else:
        kind = "document" if media_type == "application/pdf" else "image"
        content = [{"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}, ask]

    t0 = time.perf_counter()
    try:
        resp = _anthropic_client.messages.parse(
            model=model, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_format=SCHEMAS[doc_type], output_config={"effort": effort},
        )
    except anthropic.APIStatusError as e:
        return {"error": f"HTTP {e.status_code}: {e.message}", "latency_s": 0.0, "transport": True}
    except anthropic.APIConnectionError as e:
        return {"error": f"connection error: {e}", "latency_s": 0.0, "transport": True}
    except Exception as e:  # schema validation failure from parse()
        return {"error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0}
    latency = time.perf_counter() - t0

    u, p = resp.usage, price(model)
    cost = (u.input_tokens * p["in"] + u.output_tokens * p["out"]
            + (u.cache_creation_input_tokens or 0) * p["in"] * 1.25
            + (u.cache_read_input_tokens or 0) * p["cache_read"]) / 1e6
    result = {"latency_s": latency, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
              "cost_usd": cost, "stop_reason": resp.stop_reason}
    if resp.stop_reason == "refusal":
        result["error"] = f"refusal ({getattr(resp.stop_details, 'category', None)})"
    elif resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        result["error"] = f"no parsed output (stop_reason={resp.stop_reason})"
    else:
        result["data"] = resp.parsed_output.model_dump()
    return result


def extract(model: str, path: Path, doc_type: str, effort: str) -> dict:
    fn = extract_gemini if model.startswith("gemini") else extract_claude
    return fn(model, path, doc_type, effort)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def find_docs(root: Path) -> list[tuple[Path, str, Path]]:
    """Docs with ground truth: <name>.<ext> next to <name>.expected.json.
    Type comes from a parent folder named invoices/ or resumes/."""
    docs = []
    for exp in sorted(root.rglob("*.expected.json")):
        stem = exp.name[: -len(".expected.json")]
        doc = next((p for p in exp.parent.glob(stem + ".*") if p != exp), None)
        if doc is None:
            continue
        parts = {p.lower() for p in doc.relative_to(root).parts[:-1]} | {root.name.lower()}
        doc_type = "invoice" if parts & {"invoice", "invoices"} else "resume" if parts & {"resume", "resumes"} else None
        if doc_type:
            docs.append((doc, doc_type, exp))
    return docs


def default_models() -> list[str]:
    models = []
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        models += GEMINI_DEFAULT
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        models += CLAUDE_DEFAULT
    return models or GEMINI_DEFAULT


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def compare(args) -> None:
    root = Path(args.folder)
    docs = find_docs(root)
    if not docs:
        sys.exit(f"No documents with *.expected.json ground truth under {root}")
    models = args.models or default_models()

    print(f"{len(docs)} documents x {len(models)} models x {args.runs} run(s), effort={args.effort}, "
          f"{args.parallel} request(s) in flight per model\n")

    # Every model runs at the same time, each in its own pool, so one slow or
    # overloaded model doesn't hold up the others. With --parallel 1 (default)
    # a model only ever has one request open, so its latency isn't skewed by
    # competing with itself.
    if any(m.startswith("claude") for m in models):
        init_claude_client()  # once, before threads start
    expected = {exp: json.loads(exp.read_text(encoding="utf-8")) for _, _, exp in docs}
    print_lock = threading.Lock()
    t_start = time.perf_counter()

    def job(model, doc, doc_type, exp_path, run):
        r = extract(model, doc, doc_type, args.effort)
        got = r.get("data")
        if r.get("transport"):
            e, p = 0, 0  # network/server failure after retries: not the model's answer, not scored
        else:
            e, p = score(expected[exp_path], got) if got is not None else (0, score(expected[exp_path], None)[1])
        status = (r.get("error", "") + (" (not scored)" if r.get("transport") else "")) or f"{e:g}/{p} fields"
        with print_lock:
            print(f"  {model:<22} {doc.name:<26} {r['latency_s']:6.2f}s  {status}", flush=True)
        return {
            "model": model, "doc": str(doc.relative_to(root)), "type": doc_type, "run": run,
            **{k: v for k, v in r.items() if k != "data"},
            "score": [e, p], "mismatches": field_errors(expected[exp_path], got) if got else [], "data": got,
        }

    pools = {m: ThreadPoolExecutor(max_workers=args.parallel) for m in models}
    futures = [pools[m].submit(job, m, doc, t, exp, run)
               for m in models for doc, t, exp in docs for run in range(args.runs)]
    details = [f.result() for f in futures]  # submission order: stable output
    for pool in pools.values():
        pool.shutdown()
    wall = time.perf_counter() - t_start

    rows = []
    for model in models:
        mine = [d for d in details if d["model"] == model]
        scored = [d for d in mine if not d.get("transport")]
        lat = [d["latency_s"] for d in scored]

        def acc(ds):
            pts = sum(d["score"][1] for d in ds)
            return sum(d["score"][0] for d in ds) / pts if pts else None

        rows.append({
            "model": model,
            "accuracy": acc(scored),
            "invoice_acc": acc([d for d in scored if d["type"] == "invoice"]),
            "resume_acc": acc([d for d in scored if d["type"] == "resume"]),
            "lat_p50": statistics.median(lat) if lat else None,
            "lat_p95": sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)] if lat else None,
            "cost_per_doc": sum(d.get("cost_usd", 0.0) for d in scored) / len(scored) if scored else 0.0,
            "errors": sum("error" in d for d in mine),
            "n": len(mine),
        })

    def secs(x):
        return f"{x:.2f}s" if x is not None else "-"

    header = ["Model", "Accuracy", "Invoices", "Resumes", "Latency p50", "Latency p95",
              "Cost / doc", "Cost / 1K docs", "Errors"]
    table = [header, ["---"] * len(header)]
    for r in rows:
        table.append([
            f"`{r['model']}`", pct(r["accuracy"]) if r["accuracy"] is not None else "-",
            pct(r["invoice_acc"]) if r["invoice_acc"] is not None else "-",
            pct(r["resume_acc"]) if r["resume_acc"] is not None else "-",
            secs(r["lat_p50"]), secs(r["lat_p95"]),
            f"${r['cost_per_doc']:.5f}", f"${r['cost_per_doc'] * 1000:.2f}", f"{r['errors']}/{r['n']}",
        ])
    md = "\n".join("| " + " | ".join(row) + " |" for row in table)
    print("\n" + md + "\n")
    print("Accuracy = matched fields / expected fields (extra hallucinated list items count against it).")
    print("Cost = list price from token usage (free-tier keys aren't actually charged).")
    print("Errors include requests the model never answered (network, quota, outage); those aren't scored.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"comparison-{stamp}.md").write_text(md + "\n", encoding="utf-8")
    (out / f"comparison-{stamp}.json").write_text(
        json.dumps({"effort": args.effort, "summary": rows, "details": details}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"Whole benchmark took {wall:.0f}s.")
    print(f"Saved {out / f'comparison-{stamp}.md'} and per-document details in the .json")


def main() -> None:
    efforts = ["low", "medium", "high", "xhigh", "max"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="extract JSON from one document")
    ex.add_argument("file")
    ex.add_argument("--type", required=True, choices=SCHEMAS)
    ex.add_argument("--model", default=None, help="default: first model for the key you have set")
    ex.add_argument("--effort", default="low", choices=efforts)

    cp = sub.add_parser("compare", help="benchmark models against ground-truth files")
    cp.add_argument("folder")
    cp.add_argument("--models", nargs="+", default=None,
                    help="default: 3 Gemini models if GEMINI_API_KEY is set, 3 Claude models if ANTHROPIC_API_KEY is")
    cp.add_argument("--effort", default="low", choices=efforts, help="thinking effort, same for every model")
    cp.add_argument("--runs", type=int, default=1, help="repeat each doc N times for steadier latency")
    cp.add_argument("--parallel", type=int, default=1,
                    help="requests in flight per model (models always run side by side); "
                         ">1 is faster but latency then includes self-contention and may hit rate limits")
    cp.add_argument("--out", default="results")

    args = ap.parse_args()
    if args.cmd == "extract":
        model = args.model or default_models()[0]
        r = extract(model, Path(args.file), args.type, args.effort)
        if "error" in r:
            sys.exit(r["error"])
        print(json.dumps(r["data"], indent=2, ensure_ascii=False))
        print(f"\n{model}: {r['latency_s']:.2f}s, {r['input_tokens']} in / {r['output_tokens']} out, "
              f"${r['cost_usd']:.5f}", file=sys.stderr)
    else:
        compare(args)


if __name__ == "__main__":
    main()
