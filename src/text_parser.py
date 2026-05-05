# text_parser.py
# Entry: analyze_text_llm(input_dir, output_dir, cost_cap_usd=10.0, verbose=True, save_batch_json=True, reuse_existing_outputs=True)
#
# - Finds the first .md under input_dir
# - Splits into pages -> sentences
# - Mines candidates via word-bounded regex for race and gender
# - Runs race AND gender LLM passes in parallel (shared budget)
# - Uses Chat Completions API in JSON mode (no temperature, no max_tokens)
# - Strict token-based spend from usage
# - Saves raw per-batch JSON into output_dir/{race|gender}_api_batches/batch_0001.json (toggle)
# - Reuses sentences/candidates JSON when present
# - Reuses final outputs if present (optional)
#
# Requirements:
#   pip install -U openai python-dotenv
#   export OPENAI_API_KEY=...

from __future__ import annotations

import os
import re
import json
import time
import random
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Pattern, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from openai import OpenAI
from dotenv import load_dotenv


# -------------------------- Config --------------------------

MODEL = "gpt-5-mini"

BATCH_SIZE = 10                               # sentences per API call

# Global parallel budget across race+gender tasks
GLOBAL_MAX_WORKERS = int(os.environ.get("OPENAI_GLOBAL_MAX_WORKERS", "64"))

# Per-task worker cap; both tasks run concurrently, so each gets half by default
DEFAULT_TASK_MAX_WORKERS = max(1, GLOBAL_MAX_WORKERS // 2)

# Retry + backoff
MAX_RETRIES = 6
BACKOFF_MIN = 0.8
BACKOFF_MAX = 1.6
BACKOFF_CAP = 60.0

# Conservative prompt token estimate PER BATCH for gating against the shared cost cap
EST_PROMPT_TOKENS_PER_BATCH = int(os.environ.get("OPENAI_EST_PROMPT_TOKENS_PER_BATCH", "1200"))

# $ per 1M tokens (no server-side cache accounting)
if MODEL == "gpt-5-nano":
    PRICING = {MODEL: {"in": 0.05, "cached_in": 0.005, "out": 0.40}}
elif MODEL == "gpt-5-mini":
    PRICING = {MODEL: {"in": 0.25, "cached_in": 0.025, "out": 2.00}}


# -------------------------- Labels and prompts --------------------------

RACE_LABELS = ["asian", "black", "white", "latino"]
RACE_LABEL_TO_CODE = {"asian": "a", "black": "b", "white": "w", "latino": "l"}
RACE_CODE_TO_LABEL = {v: k for k, v in RACE_LABEL_TO_CODE.items()}
RACE_CANON_ORDER = RACE_LABELS

GENDER_LABELS = ["male", "female"]
GENDER_LABEL_TO_CODE = {"male": "m", "female": "f"}
GENDER_CODE_TO_LABEL = {v: k for k, v in GENDER_LABEL_TO_CODE.items()}
GENDER_CANON_ORDER = GENDER_LABELS

RACE_SYSTEM_RULES = (
    "You extract EXPLICIT race mentions from text.\n"
    "Return ONLY compact JSON: {\"r\": [[i, codes, notes], ...]}\n"
    "- i: 0-based index of the provided list\n"
    "- codes: letters from {a=asian,b=black,w=white,l=latino} in alphabetical order\n"
    "- notes: ONLY near-miss terms rejected with reasons; empty string if none. "
    "Format: 'rejected=<term> (<reason>)'. Reasons: non_person, proper_name, color_only, object, "
    "nationality_only, region_only, organization_name, historical_event, medical_term, ambiguous.\n"
    "Rules: detect explicit person-race terms; accept new unambiguous terms; reject non-person uses and proper names; "
    "map to one of asian/black/white/latino; if none then codes=\"\" and notes=\"\" unless near-misses.\n"
    "Output one minified JSON object only. Do not echo inputs."
)

GENDER_SYSTEM_RULES = (
    "You extract EXPLICIT male/female mentions from text.\n"
    "Return ONLY compact JSON: {\"r\": [[i, codes, notes], ...]}\n"
    "- i: 0-based index of the provided list\n"
    "- codes: letters from {m=male,f=female} in alphabetical order\n"
    "- notes: ONLY near-miss terms rejected with reasons; empty string if none. "
    "Format: 'rejected=<term> (<reason>)'. Reasons: non_person, proper_name, object, procedure_term, "
    "role_title, ambiguous, physiology_only.\n"
    "Rules: detect explicit male/female terms in medical context; treat identities outside {male,female} as near-miss; "
    "reject technical uses lacking classification; map to male or female; if none then codes=\"\" and notes=\"\" unless near-misses.\n"
    "Output one minified JSON object only. Do not echo inputs."
)


# -------------------------- Candidate keywords --------------------------

def _compile_terms_word_boundary(terms: Iterable[str]) -> List[Pattern]:
    """Compile word-boundary regex patterns. 'black' won't match 'blackhead'."""
    rx = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        parts = [re.escape(w) for w in re.split(r"\s+", t)]
        rx.append(re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE))
    return rx


# Race terms incl. general race/ethnicity language and legacy typologies
RACE_TERMS: Iterable[str] = [
    # canonical and common
    "asian", "east asian", "south asian", "asian american",
    "black", "african american", "afro american", "people of african descent",
    "white", "caucasian",
    "latino", "latina", "latinx", "latine", "hispanic", "hispano", "chicano", "chicana",
    # legacy and typological
    "mongoloid", "caucasoid", "negroid", "australoid", "capoid",
    # broad geo labels often used racially in med texts
    "european", "european american",
    "indian", "pakistani", "bangladeshi", "sri lankan",
    # general language
    "race", "racial", "ethnicity", "ethnic", "ethnic group", "ethnic groups", "ethnicities",
    "ancestry", "heritage", "people of color", "persons of color", "bipoc",
]

# Context amplifiers: "black patients", "white participants", etc.
RACE_CONTEXT_NOUNS = r"(?:person|people|patients?|participants?|subjects?|individuals?|cohort|group|population|men|women|children|adults)"
BLACK_CONTEXT = re.compile(rf"\bblack\s+{RACE_CONTEXT_NOUNS}\b", re.IGNORECASE)
WHITE_CONTEXT = re.compile(rf"\bwhite\s+{RACE_CONTEXT_NOUNS}\b", re.IGNORECASE)

# Gender terms
GENDER_TERMS: Iterable[str] = [
    "male", "female", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "sex", "gender",
    "transgender", "cisgender", "nonbinary", "intersex", "amab", "afab",
    "assigned male at birth", "assigned female at birth", "pregnant women", "postmenopausal women",
]

RACE_REGEXES = _compile_terms_word_boundary(RACE_TERMS) + [BLACK_CONTEXT, WHITE_CONTEXT]
GENDER_REGEXES = _compile_terms_word_boundary(GENDER_TERMS)


# -------------------------- Data structures --------------------------

@dataclass
class SentenceRef:
    page: int
    index_on_page: int
    text: str

@dataclass
class LLMUsage:
    batches: int
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    cost_usd: float

@dataclass
class RunBudget:
    cap_usd: float
    spent_usd: float = 0.0
    tripped: bool = False


# -------------------------- File discovery and parsing --------------------------

def find_first_markdown(input_dir: str) -> str:
    """Find the first .md under the folder. Skips common image dirs."""
    skip_dirs = {"extracted_images", "images", "img", "figures", "__pycache__"}
    md_paths: List[str] = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
        for fn in files:
            if fn.lower().endswith(".md"):
                md_paths.append(os.path.join(root, fn))
    if not md_paths:
        raise FileNotFoundError(f"No .md files under: {input_dir}")
    md_paths.sort(key=lambda p: p.lower())
    return os.path.abspath(md_paths[0])


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_pages(md_text: str) -> Dict[int, str]:
    lines = md_text.splitlines()
    header_rx = re.compile(r"^\s*#{1,6}\s*Page\s+(\d{1,6})\s*$", re.IGNORECASE)
    anchor_rx = re.compile(r'<a\s+id=["\']page-(\d{1,6})["\']\s*>\s*</a>\s*$', re.IGNORECASE)

    def _pnum(s: str) -> int:
        return int(s.lstrip("0") or "0")

    pages: Dict[int, List[str]] = {}
    cur_page: Optional[int] = None
    buf: List[str] = []

    def _flush():
        nonlocal buf, cur_page
        if cur_page is None:
            return
        pages.setdefault(cur_page, []).append("\n".join(buf).strip())
        buf = []

    for line in lines:
        m_h = header_rx.match(line)
        if m_h:
            new_p = _pnum(m_h.group(1))
            if cur_page is not None:
                _flush()
            cur_page = new_p
            buf = []
            continue

        m_a = anchor_rx.match(line)
        if m_a:
            new_p = _pnum(m_a.group(1))
            if cur_page is None:
                cur_page = new_p
                buf = []
            elif new_p != cur_page:
                _flush()
                cur_page = new_p
                buf = []
            continue

        if cur_page is None:
            cur_page = 1
            buf = []
        buf.append(line)

    _flush()

    out: Dict[int, str] = {}
    for p in sorted(pages.keys()):
        joined = "\n\n".join(s for s in pages[p] if s)
        out[p] = joined.strip()
    return out


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in parts if s.strip()]


def _sentences_by_page(md_text: str) -> Dict[int, List[str]]:
    pages = _split_pages(md_text)
    return {p: _split_sentences(txt) for p, txt in pages.items()}


# -------------------------- Candidate mining --------------------------

def _find_candidates(sentences_by_page: Dict[int, List[str]],
                     regexes: List[Pattern]) -> Tuple[List[SentenceRef], Dict[int, List[int]]]:
    candidates: List[SentenceRef] = []
    page_to_idxs: Dict[int, List[int]] = {}
    for page in sorted(sentences_by_page.keys()):
        sents = sentences_by_page[page]
        idxs = []
        for i, s in enumerate(sents):
            if any(rx.search(s) for rx in regexes):
                candidates.append(SentenceRef(page=page, index_on_page=i, text=s))
                idxs.append(i)
        page_to_idxs[page] = idxs
    return candidates, page_to_idxs


# -------------------------- Pricing helpers --------------------------

def _usage_tokens_chat(resp) -> Tuple[int, int, int]:
    """
    Returns (prompt_tokens, cached_prompt_tokens, completion_tokens) for Chat Completions.
    cached_prompt_tokens from prompt_tokens_details.cached_tokens if present.
    """
    try:
        u = resp.usage
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        details = getattr(u, "prompt_tokens_details", None)
        cached = 0
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or (isinstance(details, dict) and details.get("cached_tokens", 0) or 0))
        return pt, cached, ct
    except Exception:
        return 0, 0, 0


def _price_per_token(model: str) -> Dict[str, float]:
    p = PRICING[model]
    return {"in": p["in"] / 1_000_000.0, "cached_in": p["cached_in"] / 1_000_000.0, "out": p["out"] / 1_000_000.0}


def _compute_cost(pt: int, cached_pt: int, ct: int, rates: Dict[str, float]) -> float:
    uncached = max(pt - cached_pt, 0)
    return uncached * rates["in"] + cached_pt * rates["cached_in"] + ct * rates["out"]


# -------------------------- OpenAI call (Chat Completions, JSON mode) --------------------------

def _with_jitter_delay(delay: float) -> float:
    return min(BACKOFF_CAP, delay * random.uniform(BACKOFF_MIN, BACKOFF_MAX))


def _call_batch_chat(api_key: str,
                     batch: List[str],
                     system_prompt: str) -> Tuple[
                         List[Tuple[str, str]],
                         Tuple[int, int, int],
                         Dict[str, str],
                         float,
                         str,
                         str
                     ]:
    """
    Returns:
      aligned [(codes, notes)],
      (prompt_tokens, cached_prompt_tokens, completion_tokens),
      headers,
      batch_cost_usd (strictly from usage),
      err_tag: "" | "rate_limited" | "server_error" | "client_error",
      content_json: raw JSON string produced by the model
    """
    client = OpenAI(api_key=api_key)
    rates = _price_per_token(MODEL)

    delay = 1.0
    last_err_tag = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = {"texts": batch}

            # Correct API call: Chat Completions JSON mode. No temperature, no max_tokens.
            raw = client.chat.completions.with_raw_response.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )

            resp = raw.parse()
            headers = {k.lower(): v for k, v in raw.headers.items()}

            content = resp.choices[0].message.content or ""
            try:
                data = json.loads(content)
                rows = data.get("r", [])
            except Exception:
                rows = []

            # Align outputs to batch indices
            tmp = [("", "") for _ in range(len(batch))]
            for row in rows:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                try:
                    i_local = int(row[0])
                except Exception:
                    continue
                if 0 <= i_local < len(batch):
                    codes = str(row[1])
                    notes = str(row[2]) if len(row) > 2 else ""
                    tmp[i_local] = (codes, notes)

            pt, cached_pt, ct = _usage_tokens_chat(resp)
            if pt == 0 and ct == 0:
                print("[WARN] usage missing or zero for batch; cost not added", flush=True)

            batch_cost = _compute_cost(pt, cached_pt, ct, rates)
            return tmp, (pt, cached_pt, ct), headers, batch_cost, "", content

        except Exception as e:
            # Verbose error logging
            ra = getattr(e, "response", None)
            status = getattr(ra, "status_code", None) if ra else None
            body = ""
            hdrs = {}
            try:
                if ra is not None:
                    body = getattr(ra, "text", "") or ""
                    hdrs = {k.lower(): v for k, v in ra.headers.items()}
            except Exception:
                pass

            print(f"[ERR] chat batch failed attempt {attempt}/{MAX_RETRIES} status={status} model={MODEL} err={str(e)}", flush=True)
            if body:
                print(f"[ERR] body: {body[:2000]}", flush=True)
            if hdrs:
                lim_r = hdrs.get("x-ratelimit-remaining-requests", "?")
                lim_t = hdrs.get("x-ratelimit-remaining-tokens", "?")
                print(f"[ERR] remaining: requests={lim_r} tokens={lim_t}", flush=True)

            if status == 429:
                last_err_tag = "rate_limited"
            elif status and 500 <= int(status) < 600:
                last_err_tag = "server_error"
            else:
                last_err_tag = "client_error"

            # Honor Retry-After if present
            try:
                ra_hdr = ra.headers.get("retry-after") if ra else None
            except Exception:
                ra_hdr = None

            if ra_hdr:
                try:
                    time.sleep(float(ra_hdr))
                    continue
                except Exception:
                    pass

            time.sleep(_with_jitter_delay(delay))
            delay *= 2.0

    return [("", "") for _ in range(len(batch))], (0, 0, 0), {}, 0.0, last_err_tag, ""


# -------------------------- Parallel + capped + progress --------------------------

def _chunks(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def _parse_limits(h: Dict[str, str]) -> Tuple[int, int]:
    def _int(x):
        try:
            return int(x)
        except Exception:
            return 0
    return _int(h.get("x-ratelimit-limit-requests", "0")), _int(h.get("x-ratelimit-limit-tokens", "0"))


def _estimate_batch_cost(rates: Dict[str, float]) -> float:
    # Estimate using prompt only; actual spend uses real usage later.
    return EST_PROMPT_TOKENS_PER_BATCH * rates["in"]


def _classify_parallel_capped(sentences: List[str],
                              system_prompt: str,
                              budget: RunBudget,
                              budget_lock: threading.Lock,
                              label: str,
                              *,
                              task_max_workers: int,
                              save_raw_dir: Optional[str]) -> Tuple[List[Tuple[str, str]], LLMUsage]:
    """
    Run batched classification with:
      - shared cost cap via budget + lock
      - verbose progress
      - per-batch raw JSON saving if enabled
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    rates = _price_per_token(MODEL)
    est_cost = _estimate_batch_cost(rates)

    batches = _chunks(sentences, BATCH_SIZE)
    total_batches = len(batches)

    results: List[Tuple[str, str]] = [("", "") for _ in range(len(sentences))]
    totals = {"batches": 0, "pt": 0, "cached": 0, "ct": 0}

    if total_batches == 0:
        return results, LLMUsage(0, 0, 0, 0, 0.0)

    workers = max(1, min(task_max_workers, total_batches))

    if save_raw_dir:
        os.makedirs(save_raw_dir, exist_ok=True)

    with budget_lock:
        print(f"[{label}] start: batches={total_batches} workers={workers} cap=${budget.cap_usd:.2f} "
              f"spent=${budget.spent_usd:.4f} est_batch_cost~${est_cost:.6f}", flush=True)

    next_submit = 0
    futures: Dict[Future, int] = {}

    def can_submit() -> bool:
        with budget_lock:
            return (next_submit < total_batches) and (budget.spent_usd + est_cost <= budget.cap_usd) and (not budget.tripped)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Prime
        while len(futures) < workers and can_submit():
            idx = next_submit
            fut = ex.submit(_call_batch_chat, api_key, batches[idx], system_prompt)
            futures[fut] = idx
            next_submit += 1

        completed = 0

        # Process
        while futures:
            for fut in as_completed(list(futures.keys())):
                idx = futures.pop(fut)
                base = idx * BATCH_SIZE

                tmp, (pt, cached_pt, ct), headers, batch_cost, err_tag, content = fut.result()

                # Rate-limit handling: log and resubmit same batch after short wait
                if err_tag == "rate_limited":
                    print(f"[{label}] 429 detected. resubmitting batch {idx+1}.", flush=True)
                    time.sleep(1.0)
                    nf = ex.submit(_call_batch_chat, api_key, batches[idx], system_prompt)
                    futures[nf] = idx
                    continue

                # Save raw batch JSON
                if save_raw_dir:
                    raw_path = os.path.join(save_raw_dir, f"batch_{idx+1:04d}.json")
                    try:
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(content if content.strip() else "{}")
                    except Exception as e:
                        print(f"[WARN] failed to save raw batch JSON: {e}", flush=True)

                # Write aligned results
                for j, pair in enumerate(tmp):
                    if base + j < len(results):
                        results[base + j] = pair

                # Update totals and budget
                with budget_lock:
                    totals["batches"] += 1
                    totals["pt"] += pt
                    totals["cached"] += cached_pt
                    totals["ct"] += ct
                    budget.spent_usd += batch_cost
                    completed += 1

                    rpm, tpm = _parse_limits(headers)
                    print(f"[{label}] done {completed}/{total_batches}  in={pt} out={ct} "
                          f"cost+${batch_cost:.6f}  spent=${budget.spent_usd:.6f}  rpm_lim={rpm} tpm_lim={tpm}",
                          flush=True)

                    if budget.spent_usd >= budget.cap_usd:
                        budget.tripped = True
                        print(f"[{label}] cost cap reached. stopping further submits.", flush=True)

                # Submit next if allowed
                while len(futures) < workers and can_submit():
                    if next_submit >= total_batches:
                        break
                    nf = ex.submit(_call_batch_chat, api_key, batches[next_submit], system_prompt)
                    futures[nf] = next_submit
                    next_submit += 1

    final_cost = _compute_cost(totals["pt"], totals["cached"], totals["ct"], _price_per_token(MODEL))
    usage = LLMUsage(
        batches=totals["batches"],
        prompt_tokens=totals["pt"],
        cached_prompt_tokens=totals["cached"],
        completion_tokens=totals["ct"],
        cost_usd=final_cost,
    )

    return results, usage


# -------------------------- Key normalization --------------------------

def _keys_to_int(d: Dict[Any, Any]) -> Dict[int, Any]:
    """Convert JSON-loaded dict keys to int where possible."""
    out: Dict[int, Any] = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            pass
    return out


# -------------------------- Aggregation --------------------------

def _decode_codes(codes: str, code2label: Dict[str, str], order: List[str]) -> List[str]:
    labels = [code2label[c] for c in sorted(set(codes)) if c in code2label]
    return [k for k in order if k in labels]


def _aggregate_by_page(sentences_by_page: Dict[int, List[str]],
                       candidates: List[SentenceRef],
                       codes_and_notes: List[Tuple[str, str]],
                       code2label: Dict[str, str],
                       label_order: List[str]) -> Dict[int, Dict[str, Any]]:
    # Normalize sentence map keys to int
    sbp: Dict[int, List[str]] = {}
    for k, v in sentences_by_page.items():
        try:
            sbp[int(k)] = v
        except Exception:
            continue

    # Build per_page over union of pages from sentences and candidates
    page_set = set(sbp.keys()) | {ref.page for ref in candidates}
    per_page: Dict[int, Dict[str, Any]] = {}
    for page in sorted(page_set):
        per_page[page] = {
            "total_sentences": len(sbp.get(page, [])),
            "counts": {lbl: 0 for lbl in label_order},
            "sentences": [],
            "notes": []
        }

    # Fill counts and lists
    for ref, (codes, notes) in zip(candidates, codes_and_notes):
        page = int(ref.page)
        if page not in per_page:
            per_page[page] = {
                "total_sentences": len(sbp.get(page, [])),
                "counts": {lbl: 0 for lbl in label_order},
                "sentences": [],
                "notes": []
            }
        labels = _decode_codes(codes, code2label, label_order)
        if not labels and not notes:
            continue
        entry = per_page[page]
        for lbl in labels:
            entry["counts"][lbl] += 1
        if labels:
            entry["sentences"].append({
                "index_on_page": ref.index_on_page,
                "text": ref.text,
                "labels": labels,
                "notes": notes
            })
        if notes:
            entry["notes"].append(notes)

    # Dedup notes
    for page in per_page:
        seen, uniq = set(), []
        for n in per_page[page]["notes"]:
            n2 = (n or "").strip()
            if n2 and n2 not in seen:
                uniq.append(n2)
                seen.add(n2)
        per_page[page]["notes"] = uniq

    return per_page


# -------------------------- I/O helpers --------------------------

def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _sentence_refs_from_json(cand_json: Dict[str, Any]) -> List[SentenceRef]:
    out = []
    for c in cand_json.get("candidates", []):
        out.append(SentenceRef(page=int(c["page"]), index_on_page=int(c["index_on_page"]), text=str(c["text"])))
    return out


def _candidates_payload(candidates: List[SentenceRef], page_map: Dict[int, List[int]]) -> Dict[str, Any]:
    return {
        "total_candidates": len(candidates),
        "counts_per_page": {str(p): len(idxs) for p, idxs in page_map.items()},
        "candidates": [
            {"page": r.page, "index_on_page": r.index_on_page, "text": r.text}
            for r in candidates
        ],
    }


# -------------------------- Public entry --------------------------

def analyze_text_llm(input_dir: str,
                     output_dir: str,
                     *,
                     cost_cap_usd: float = 10.0,
                     verbose: bool = True,
                     save_batch_json: bool = True,
                     reuse_existing_outputs: bool = True) -> Tuple[str, str, Dict[str, Any]]:
    """
    input_dir: folder containing at least one .md (searched recursively).
    Writes results directly into output_dir, reuses existing intermediate JSONs if present.
    Returns (race_json_path, gender_json_path, meta_dict)
    """
    load_dotenv()
    os.makedirs(output_dir, exist_ok=True)

    # Reuse existing final outputs if present
    race_out_path = os.path.join(output_dir, "race_results.json")
    gender_out_path = os.path.join(output_dir, "gender_results.json")
    if reuse_existing_outputs and os.path.exists(race_out_path) and os.path.exists(gender_out_path):
        if verbose:
            print("[reuse] using existing race_results.json and gender_results.json")
        race_json = _load_json(race_out_path) or {}
        gender_json = _load_json(gender_out_path) or {}
        meta = {
            "output_root": output_dir,
            "markdown": None,
            "race_usage": race_json.get("usage", {}),
            "gender_usage": gender_json.get("usage", {}),
            "pricing_per_1M": PRICING[MODEL],
            "cost_cap": race_json.get("cost_cap") or gender_json.get("cost_cap") or {},
            "saved_batches": {
                "race": os.path.isdir(os.path.join(output_dir, "race_api_batches")),
                "gender": os.path.isdir(os.path.join(output_dir, "gender_api_batches")),
                "race_dir": os.path.join(output_dir, "race_api_batches"),
                "gender_dir": os.path.join(output_dir, "gender_api_batches"),
            }
        }
        return race_out_path, gender_out_path, meta

    md_path = find_first_markdown(input_dir)

    # sentences_by_page.json reuse and key normalization
    sentences_path = os.path.join(output_dir, "sentences_by_page.json")
    sentences_by_page = _load_json(sentences_path)
    if sentences_by_page is None:
        md = _read_file(md_path)
        sentences_by_page = _sentences_by_page(md)  # keys are ints here
        _save_json(sentences_path, sentences_by_page)
        if verbose:
            print("[prep] built sentences_by_page.json")
    else:
        sentences_by_page = _keys_to_int(sentences_by_page)
        if verbose:
            print("[prep] reused sentences_by_page.json")

    # candidate reuse
    race_cand_path = os.path.join(output_dir, "race_candidates.json")
    gender_cand_path = os.path.join(output_dir, "gender_candidates.json")

    race_cand_json = _load_json(race_cand_path)
    gender_cand_json = _load_json(gender_cand_path)

    if race_cand_json is None or gender_cand_json is None:
        race_candidates, race_page_map = _find_candidates(sentences_by_page, RACE_REGEXES)
        gender_candidates, gender_page_map = _find_candidates(sentences_by_page, GENDER_REGEXES)
        _save_json(race_cand_path, _candidates_payload(race_candidates, race_page_map))
        _save_json(gender_cand_path, _candidates_payload(gender_candidates, gender_page_map))
        if verbose:
            print("[prep] built candidate lists")
    else:
        # backfill total count if missing
        if "total_candidates" not in race_cand_json:
            race_cand_json["total_candidates"] = len(race_cand_json.get("candidates", []))
            _save_json(race_cand_path, race_cand_json)
        if "total_candidates" not in gender_cand_json:
            gender_cand_json["total_candidates"] = len(gender_cand_json.get("candidates", []))
            _save_json(gender_cand_path, gender_cand_json)
        if verbose:
            print("[prep] reused candidate lists")

    # Load aligned candidate refs
    race_candidates = _sentence_refs_from_json(_load_json(race_cand_path) or {"candidates": []})
    gender_candidates = _sentence_refs_from_json(_load_json(gender_cand_path) or {"candidates": []})

    # Shared cost budget across race and gender
    budget = RunBudget(cap_usd=max(0.0, float(cost_cap_usd)))
    budget_lock = threading.Lock()

    # Prepare texts
    race_texts = [c.text for c in race_candidates]
    gender_texts = [c.text for c in gender_candidates]

    # Batch JSON output dirs
    race_raw_dir = os.path.join(output_dir, "race_api_batches") if save_batch_json else None
    gender_raw_dir = os.path.join(output_dir, "gender_api_batches") if save_batch_json else None

    # Run race and gender in parallel, each with half the global workers
    task_workers = max(1, DEFAULT_TASK_MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_race = pool.submit(
            _classify_parallel_capped,
            race_texts, RACE_SYSTEM_RULES, budget, budget_lock,
            "race", task_max_workers=task_workers, save_raw_dir=race_raw_dir
        )
        f_gender = pool.submit(
            _classify_parallel_capped,
            gender_texts, GENDER_SYSTEM_RULES, budget, budget_lock,
            "gender", task_max_workers=task_workers, save_raw_dir=gender_raw_dir
        )
        race_codes_notes, race_usage = f_race.result()
        gender_codes_notes, gender_usage = f_gender.result()

    # Aggregate per page
    race_per_page = _aggregate_by_page(
        sentences_by_page, race_candidates, race_codes_notes, RACE_CODE_TO_LABEL, RACE_CANON_ORDER
    )
    gender_per_page = _aggregate_by_page(
        sentences_by_page, gender_candidates, gender_codes_notes, GENDER_CODE_TO_LABEL, GENDER_CANON_ORDER
    )

    # Write final outputs
    race_path = os.path.join(output_dir, "race_results.json")
    gender_path = os.path.join(output_dir, "gender_results.json")

    _save_json(race_path, {
        "model": MODEL,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usage": {
            "batches": race_usage.batches,
            "prompt_tokens": race_usage.prompt_tokens,
            "cached_prompt_tokens": race_usage.cached_prompt_tokens,
            "completion_tokens": race_usage.completion_tokens,
            "cost_usd": round(race_usage.cost_usd, 6),
            "pricing_per_1M": PRICING[MODEL],
        },
        "per_page": race_per_page
    })

    _save_json(gender_path, {
        "model": MODEL,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usage": {
            "batches": gender_usage.batches,
            "prompt_tokens": gender_usage.prompt_tokens,
            "cached_prompt_tokens": gender_usage.cached_prompt_tokens,
            "completion_tokens": gender_usage.completion_tokens,
            "cost_usd": round(gender_usage.cost_usd, 6),
            "pricing_per_1M": PRICING[MODEL],
        },
        "per_page": gender_per_page
    })

    if verbose:
        print(f"\nProcessed markdown: {md_path}")
        print("\n--- Race ---")
        print(f"batches={race_usage.batches}  in={race_usage.prompt_tokens}  "
              f"cached_in={race_usage.cached_prompt_tokens}  "
              f"out={race_usage.completion_tokens}  cost_usd={race_usage.cost_usd:.6f}")
        print("--- Gender ---")
        print(f"batches={gender_usage.batches}  in={gender_usage.prompt_tokens}  "
              f"cached_in={gender_usage.cached_prompt_tokens}  "
              f"out={gender_usage.completion_tokens}  cost_usd={gender_usage.cost_usd:.6f}")
        if save_batch_json:
            print(f"Raw batches: {race_raw_dir} | {gender_raw_dir}")

    meta = {
        "output_root": output_dir,
        "markdown": md_path,
        "race_usage": race_usage.__dict__,
        "gender_usage": gender_usage.__dict__,
        "pricing_per_1M": PRICING[MODEL],
        "saved_batches": {
            "race": bool(save_batch_json),
            "gender": bool(save_batch_json),
            "race_dir": race_raw_dir,
            "gender_dir": gender_raw_dir,
        }
    }

    return race_path, gender_path, meta
