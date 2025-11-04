#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates counterfactual prompts for a single dimension, calls an LLM provider,
and saves raw responses as JSONL to runs/{model}_{dimension}.jsonl.

Providers supported:
  --provider openai|anthropic|gemini|ollama
Keys expected in .env:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
For ollama, ensure a local server at http://localhost:11434
"""
from pathlib import Path
import os, json, time, argparse, datetime as dt, csv, re
import random
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
ATTR_DIR    = ROOT / "attributes"
RUNS_DIR    = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# ---------- util: load simple CSV lists ----------
def load_list(csv_path: Path):
    vals = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                vals.append(row[0].strip())
    return vals

# ---------- provider adapters ----------
def call_openai(model, prompt, temperature=0.2, system_prompt="Answer directly and briefly."):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    use_responses = model.startswith("gpt-5")  # GPT-5 family → Responses API
    # Some GPT-5 variants (e.g., nano) reject temperature outright.
    supports_temperature = not ("nano" in model)

    if use_responses:
        url = "https://api.openai.com/v1/responses"
        data = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user",   "content": [{"type": "input_text", "text": prompt}]}
            ]
        }
        if supports_temperature:
            data["temperature"] = float(temperature)
    else:
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ]
        }

    def _post(payload):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            print("OpenAI error:", r.text)
        r.raise_for_status()
        return r.json()

    try:
        out = _post(data)
    except requests.HTTPError as e:
        # Auto-retry if the error is caused by sending 'temperature' to a model that doesn't support it
        if use_responses and "Unsupported parameter: 'temperature'" in getattr(e.response, "text", ""):
            data.pop("temperature", None)
            out = _post(data)
        else:
            raise

    if not use_responses:
        # Chat Completions extraction
        return out["choices"][0]["message"]["content"].strip()

    # Responses extraction
    # 1) direct convenience field if present
    txt = out.get("output_text")
    if isinstance(txt, str) and txt:
        return txt.strip()

    # 2) iterate over output items and gather assistant message text
    try:
        output_items = out.get("output") or []
        texts = []
        for item in output_items:
            # Prefer assistant messages
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    # Newer schema uses type=output_text
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        texts.append(part["text"])
        if texts:
            return "\n".join(t.strip() for t in texts if t).strip()
    except Exception:
        pass

    # 3) last resort: try to look for any text-like field
    try:
        maybe_text = out.get("text")
        if isinstance(maybe_text, str) and maybe_text.strip():
            return maybe_text.strip()
    except Exception:
        pass

    return ""

def call_anthropic(model, prompt, temperature=0.2, system_prompt="Answer directly and briefly."):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": 512,
        "temperature": float(temperature),
        "system": system_prompt,
        "messages":[{"role":"user","content": prompt}]
    }

    # Retry on timeouts, 429/529 and 5xx with exponential backoff + jitter
    max_retries = 6
    last_err = None
    for attempt in range(max_retries):
        try:
            # Separate connect/read timeouts: (10s connect, 120s read)
            r = requests.post(url, headers=headers, json=data, timeout=(10, 120))
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            delay = min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.3)
            time.sleep(delay)
            continue

        if r.status_code in (429, 529) or (500 <= r.status_code < 600):
            ra = r.headers.get("Retry-After")
            try:
                delay = float(ra) if ra else (0.5 * (2 ** attempt))
            except Exception:
                delay = 0.5 * (2 ** attempt)
            delay = min(8.0, delay) + random.uniform(0, 0.3)
            time.sleep(delay)
            last_err = r
            continue

        # Non-retriable or success
        r.raise_for_status()
        out = r.json()
        return "".join([c.get("text", "") for c in out.get("content", [])]).strip()

    # Exhausted retries
    if isinstance(last_err, requests.Response):
        last_err.raise_for_status()
    raise RuntimeError("Anthropic request failed after retries: %r" % (last_err,))

def call_gemini(model, prompt, temperature=0.2, system_prompt="Answer directly and briefly."):
    # Uses REST for Generative Language API (text-only).
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    # Minimal prompt; system prompt is prepended.
    full_prompt = system_prompt + "\n\n" + prompt
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": 512},
        "contents": [{"parts":[{"text": full_prompt}]}]
    }

    # Simple retry with exponential backoff for rate limiting / transient errors
    max_retries = 6
    last_err = None
    for attempt in range(max_retries):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 429 or (500 <= r.status_code < 600):
            # honor Retry-After when present; otherwise exponential backoff with jitter
            ra = r.headers.get("Retry-After")
            try:
                delay = float(ra) if ra else (0.5 * (2 ** attempt))
            except Exception:
                delay = 0.5 * (2 ** attempt)
            delay = min(delay, 8.0) + random.uniform(0, 0.3)
            time.sleep(delay)
            last_err = r
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError:
            # non-retriable error; surface it
            raise
        out = r.json()
        # extract text
        candidates = out.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join([p.get("text","") for p in parts]).strip()

    # If we exhausted retries, raise the last error
    if last_err is not None:
        last_err.raise_for_status()
    raise RuntimeError("Gemini request failed with unknown error after retries.")

def call_ollama(model, prompt, temperature=0.2, system_prompt="Answer directly and briefly."):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "options": {"temperature": float(temperature)}
    }
    r = requests.post(url, json=data, timeout=120)
    r.raise_for_status()
    # stream: accumulate
    text = []
    for line in r.iter_lines():
        if not line: continue
        try:
            obj = json.loads(line.decode("utf-8"))
            if "response" in obj:
                text.append(obj["response"])
        except Exception:
            pass
    return "".join(text).strip() if text else ""

def call_model(provider, model, prompt, temperature=0.2, system_prompt="Answer directly and briefly."):
    if provider == "openai":
        return call_openai(model, prompt, temperature, system_prompt)
    elif provider == "anthropic":
        return call_anthropic(model, prompt, temperature, system_prompt)
    elif provider == "gemini":
        return call_gemini(model, prompt, temperature, system_prompt)
    elif provider == "ollama":
        return call_ollama(model, prompt, temperature, system_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------- prompt filling ----------
PLACEHOLDERS = {
    "gender":      ["[NAME]"],
    "nationality": ["[NAME]","[COUNTRY]"],
    "ethnicity":   ["[NAME]","[ETHNICITY]"],
    "religion":    ["[NAME]","[RELIGION]"],
    "age":         ["[NAME]","[AGE]"],
    "disability":  ["[NAME]","[DISABILITY]"],
}

def fill_template(tpl, **kw):
    out = tpl
    for k, v in kw.items():
        out = out.replace(f"[{k.upper()}]", str(v))
    return out

def yield_items(dimension, n_names=5, temperature=0.2):
    yaml_path = PROMPTS_DIR / f"{dimension}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    prompts = y.get("prompts", [])
    if not prompts:
        raise ValueError(f"No prompts found in {yaml_path}")

    # load names & attributes
    neutral_names = load_list(ATTR_DIR / "neutral_names.csv")

    if dimension == "gender":
        male_names   = load_list(ATTR_DIR / "gender_male.csv")[:n_names]
        female_names = load_list(ATTR_DIR / "gender_female.csv")[:n_names]
        for pid, tpl in enumerate(prompts):
            for i in range(len(male_names)):
                yield {
                    "id": f"gender_m_{pid}_{i}",
                    "dimension": "gender",
                    "prompt_id": pid,
                    "name_index": i,
                    "attrs": {"gender": "male", "name": male_names[i]},
                    "prompt_text": fill_template(tpl, name=male_names[i]),
                    "params": {"temperature": temperature},
                }
                yield {
                    "id": f"gender_f_{pid}_{i}",
                    "dimension": "gender",
                    "prompt_id": pid,
                    "name_index": i,
                    "attrs": {"gender": "female", "name": female_names[i]},
                    "prompt_text": fill_template(tpl, name=female_names[i]),
                    "params": {"temperature": temperature},
                }
    elif dimension == "nationality":
        countries = load_list(ATTR_DIR / "countries.csv")
        for pid, tpl in enumerate(prompts):
            for i, nm in enumerate(neutral_names[:n_names]):
                for c in countries:
                    yield {
                        "id": f"nationality_{pid}_{i}_{c}",
                        "dimension": "nationality",
                        "prompt_id": pid,
                        "name_index": i,
                        "attrs": {"country": c, "name": nm},
                        "prompt_text": fill_template(tpl, name=nm, country=c),
                        "params": {"temperature": temperature},
                    }
    elif dimension == "ethnicity":
        groups = load_list(ATTR_DIR / "ethnicity.csv")
        for pid, tpl in enumerate(prompts):
            for i, nm in enumerate(neutral_names[:n_names]):
                for g in groups:
                    yield {
                        "id": f"ethnicity_{pid}_{i}_{g}",
                        "dimension": "ethnicity",
                        "prompt_id": pid,
                        "name_index": i,
                        "attrs": {"ethnicity": g, "name": nm},
                        "prompt_text": fill_template(tpl, name=nm, ethnicity=g),
                        "params": {"temperature": temperature},
                    }
    elif dimension == "religion":
        groups = load_list(ATTR_DIR / "religion.csv")
        for pid, tpl in enumerate(prompts):
            for i, nm in enumerate(neutral_names[:n_names]):
                for g in groups:
                    yield {
                        "id": f"religion_{pid}_{i}_{g}",
                        "dimension": "religion",
                        "prompt_id": pid,
                        "name_index": i,
                        "attrs": {"religion": g, "name": nm},
                        "prompt_text": fill_template(tpl, name=nm, religion=g),
                        "params": {"temperature": temperature},
                    }
    elif dimension == "age":
        ages = load_list(ATTR_DIR / "age.csv")
        for pid, tpl in enumerate(prompts):
            for i, nm in enumerate(neutral_names[:n_names]):
                for a in ages:
                    yield {
                        "id": f"age_{pid}_{i}_{a}",
                        "dimension": "age",
                        "prompt_id": pid,
                        "name_index": i,
                        "attrs": {"age": a, "name": nm},
                        "prompt_text": fill_template(tpl, name=nm, age=a),
                        "params": {"temperature": temperature},
                    }
    elif dimension == "disability":
        ds = load_list(ATTR_DIR / "disability.csv")
        for pid, tpl in enumerate(prompts):
            for i, nm in enumerate(neutral_names[:n_names]):
                for d in ds:
                    yield {
                        "id": f"disability_{pid}_{i}_{d}",
                        "dimension": "disability",
                        "prompt_id": pid,
                        "name_index": i,
                        "attrs": {"disability": d, "name": nm},
                        "prompt_text": fill_template(tpl, name=nm, disability=d),
                        "params": {"temperature": temperature},
                    }
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., gpt-4o-mini / claude-3-5-sonnet / gemini-1.5-pro / llama3:8b")
    ap.add_argument("--provider", required=True, choices=["openai","anthropic","gemini","ollama"])
    ap.add_argument("--dimension", required=True, choices=["gender","nationality","ethnicity","religion","age","disability"])
    ap.add_argument("--n_names", type=int, default=5, help="Neutral or gendered names per set")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--system_prompt", default="Answer directly and briefly.")
    ap.add_argument("--sleep", type=float, default=0.6, help="Sleep seconds between calls")
    ap.add_argument("--prompt_suffix", default="", help="Optional instruction appended to each prompt")
    ap.add_argument("--runs_dir", default=str(RUNS_DIR), help="Directory where JSONL outputs are stored")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.prompt_suffix.strip()

    out_path = runs_dir / f"{args.model}_{args.dimension}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        for item in yield_items(args.dimension, n_names=args.n_names, temperature=args.temperature):
            base_prompt = item["prompt_text"]
            prompt_to_send = base_prompt if not suffix else f"{base_prompt.rstrip()}\n\n{suffix}"

            resp = call_model(args.provider, args.model, prompt_to_send, args.temperature, args.system_prompt)
            record = {
                **item,
                "model": args.model,
                "provider": args.provider,
                "response_text": resp,
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            }
            record["prompt_text"] = prompt_to_send
            record["base_prompt_text"] = base_prompt
            record["prompt_suffix"] = suffix
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
