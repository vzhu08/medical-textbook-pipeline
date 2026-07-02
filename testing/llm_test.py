# classify_race_mentions.py
# Batch version (up to 10 sentences per API call), gpt-5-nano, JSON mode.
# Notes only include near-misses. No local or server caching.
#
# Usage:
#   pip install -U openai python-dotenv
#   export OPENAI_API_KEY=...
#   python classify_race_mentions.py

from __future__ import annotations
import os, json, time, re, math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- Config ----------------

load_dotenv()

MODEL = "gpt-5-nano"          # fixed per your request
BATCH_SIZE = 10               # max items per API call

# --- Storage locations (literal strings) ---
OUT_DIR = r"../data/LLM Tests"  # relative to your current working directory
OUT_NAME = "race_mentions"    # base filename

# ensure directory exists
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SENTENCES = [
    "The study included Asian and Black participants.",
    "She wore a white dress to the ceremony.",
    "The report compared outcomes for African American and Caucasian patients.",
    "Coach White praised Jordan Black for the win.",
    "No race is mentioned here."
]

# $ per 1M tokens (include cached_in for accounting; will be zero without server cache)
PRICING = {
    "gpt-5-nano": {"in": 0.05, "cached_in": 0.005, "out": 0.40},
}

# ---------------- Prompt ----------------
# Notes must ONLY include near-misses, explained clearly.
SYSTEM_RULES = (
    "You extract EXPLICIT race mentions from text.\n"
    "Return ONLY JSON: {\"results\":[{\"i\":<index>,\"labels\":[...],\"notes\":\"...\"}, ...]}\n"
    "\n"
    "Decision rules:\n"
    "1) Detect tokens/phrases that directly denote a person's race/ethnicity.\n"
    "   Use examples as hints but ALSO accept new, unambiguous terms that map to one label.\n"
    "2) Reject non-person uses (e.g., 'white dress', 'black box', 'white matter', 'black pepper') and reject proper names\n"
    "   (e.g., 'Mr. White', 'Jordan Black').\n"
    "3) Map accepted terms to exactly one canonical label:\n"
    "   asian  ← Asian, East Asian, South Asian, Asian American, Asiatic (people, historical), Pan-Asian, Sino-, Indo- (when racial)\n"
    "   black  ← Black, African American, Afro-American, Afro-descendant, people of African descent\n"
    "   white  ← White, Caucasian, European (as a racial descriptor), European American, Anglo (racial)\n"
    "   latino ← Latino, Latina, Latinx, Latine, Hispanic, Hispano, Chicano/a, Latino/a\n"
    "4) If no explicit person-race term appears, labels must be [].\n"
    "\n"
    "Output constraints:\n"
    "- labels: only from ['asian','black','white','latino'] (lowercase).\n"
    "- notes: ONLY list near-miss terms that you considered but rejected, with clear reasons, format:\n"
    "  rejected=<term1> (<reason>), <term2> (<reason>)\n"
    "  Use concise reasons from: non_person, proper_name, color_only, object, nationality_only, region_only,\n"
    "  organization_name, historical_event, medical_term, ambiguous. If none, use empty string.\n"
    "- Do not include matched terms in notes. Do not include extra fields. Keep one line."
)

# ---------------- Canonicalization ----------------

CANON_SYNS: Dict[str, List[str]] = {
    "asian":  ["asian", "east asian", "asian american", "south asian"],
    "black":  ["black", "african american", "african-american"],
    "white":  ["white", "caucasian"],
    "latino": ["latino", "latina", "latinx", "latine", "hispanic", "latino/a", "latina/o"],
}
CANON_ORDER = ["asian", "black", "white", "latino"]

# ---------------- Data ----------------

@dataclass
class RunResult:
    sentence: str
    labels: List[str]
    notes: str
    # batch usage (same for all items in a batch; totals printed separately)
    batch_id: int
    batch_prompt_tokens: int
    batch_cached_prompt_tokens: int
    batch_completion_tokens: int
    latency_s: float

# ---------------- Helpers ----------------

def price_per_token(model: str) -> Dict[str, float]:
    p = PRICING[model]
    return {"in": p["in"]/1_000_000.0, "cached_in": p["cached_in"]/1_000_000.0, "out": p["out"]/1_000_000.0}

def _usage_tokens_chat(resp) -> Tuple[int, int, int]:
    """Return (prompt_tokens, cached_prompt_tokens, completion_tokens)."""
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

def _canonicalize_label(raw: str) -> str | None:
    x = raw.strip().lower()
    x = re.sub(r"[/,_-]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    for canon, syns in CANON_SYNS.items():
        if x in syns:
            return canon
        parts = set(re.split(r"[^\w]+", x))
        if any(s in parts for s in syns):
            return canon
    return x if x in CANON_SYNS else None

def _normalize_labels(labels_in: List[Any]) -> List[str]:
    canon_set = set()
    for raw in labels_in or []:
        c = _canonicalize_label(str(raw))
        if c:
            canon_set.add(c)
    return [k for k in CANON_ORDER if k in canon_set]

def _chunks(arr: List[str], size: int) -> List[List[str]]:
    return [arr[i:i+size] for i in range(0, len(arr), size)]

# ---------------- Core ----------------

def classify_batched(sentences: List[str]) -> Tuple[List[RunResult], Dict[str, float]]:
    """Process sentences in batches, preserve original order, return per-item results and totals."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in env or .env")

    client = OpenAI(api_key=api_key)
    p = price_per_token(MODEL)

    results: List[RunResult] = []
    totals = {
        "batches": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }

    # Preallocate result slots by input order
    slots: List[RunResult | None] = [None] * len(sentences)

    for batch_id, batch in enumerate(_chunks(sentences, BATCH_SIZE), start=1):
        t0 = time.time()

        # Build a compact user payload that the model echoes in aligned results
        payload = {"texts": batch}

        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},   # JSON mode
        )

        pt, cached_pt, ct = _usage_tokens_chat(r)
        latency = time.time() - t0

        # Aggregate totals and cost at batch level
        uncached_in = max(pt - cached_pt, 0)
        batch_cost = uncached_in * p["in"] + cached_pt * p["cached_in"] + ct * p["out"]

        totals["batches"] += 1
        totals["prompt_tokens"] += pt
        totals["cached_prompt_tokens"] += cached_pt
        totals["completion_tokens"] += ct
        totals["cost_usd"] += batch_cost

        # Parse and normalize
        try:
            data = json.loads(r.choices[0].message.content)
            items = data.get("results", [])
        except Exception:
            items = []

        # Map batch-local indices back to global order
        base = (batch_id - 1) * BATCH_SIZE
        for item in items:
            i_local = int(item.get("i", -1))
            if not (0 <= i_local < len(batch)):
                continue
            i_global = base + i_local
            if i_global >= len(sentences):
                continue

            labels = _normalize_labels(item.get("labels", []))
            notes = str(item.get("notes", "") or "")

            slots[i_global] = RunResult(
                sentence=sentences[i_global],
                labels=labels,
                notes=notes,
                batch_id=batch_id,
                batch_prompt_tokens=pt,
                batch_cached_prompt_tokens=cached_pt,
                batch_completion_tokens=ct,
                latency_s=latency,
            )

        # Fill any missing items in this batch with safe defaults
        for i_local in range(len(batch)):
            i_global = base + i_local
            if i_global < len(sentences) and slots[i_global] is None:
                slots[i_global] = RunResult(
                    sentence=sentences[i_global],
                    labels=[],
                    notes="",
                    batch_id=batch_id,
                    batch_prompt_tokens=pt,
                    batch_cached_prompt_tokens=cached_pt,
                    batch_completion_tokens=ct,
                    latency_s=latency,
                )

    # Collapse slots to results
    results = [s for s in slots if s is not None]
    return results, totals

# ---------------- CLI ----------------

def main():
    print(f"Model: {MODEL}  |  BATCH_SIZE: {BATCH_SIZE}")

    runs, totals = classify_batched(TEST_SENTENCES)

    # Save normalized outputs to disk
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_path = os.path.join(OUT_DIR, f"{OUT_NAME}_{ts}.json")

    out_payload = {
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals": totals,
        "results": [
            {"i": idx, "text": r.sentence, "labels": r.labels, "notes": r.notes, "batch": r.batch_id}
            for idx, r in enumerate(runs)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    # Console print (compact)
    for i, r in enumerate(runs, 1):
        print(f"\n#{i}")
        print(f"Text: {r.sentence}")
        print(f"labels: {r.labels}")
        if r.notes:
            print(f"notes: {r.notes}")

    print("\n--- summary ---")
    print(f"saved_json: {out_path}")
    print(f"batches: {totals['batches']}")
    print(f"total_prompt_tokens: {totals['prompt_tokens']}  "
          f"total_cached_prompt_tokens: {totals['cached_prompt_tokens']}  "
          f"total_uncached_prompt_tokens: {totals['prompt_tokens'] - totals['cached_prompt_tokens']}")
    print(f"total_completion_tokens: {totals['completion_tokens']}")
    print(f"total_cost_usd: {totals['cost_usd']:.6f}")

if __name__ == "__main__":
    main()
