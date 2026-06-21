# -*- coding: utf-8 -*-
"""
Paper section 3.3 "A Helping Hand from LLMs" -- text augmentation for EATD.

GPT-3.5 principle-based rephrasing of each participant answer, following the paper's
five principles (content integrity, colloquial naturalness, respect/appropriateness,
length consistency, tolerate disfluency).  EATD answers are Chinese, so the rephrasing
is performed in Chinese (the corpus language) -- no translation.  For each subject the
three polarity answers (negative / positive / neutral) are rewritten and saved as
{polarity}_synthetic.txt next to the originals.  Resumable: finished files are skipped.

Configure an OpenAI-compatible endpoint via env:
  OPENAI_API_KEY   (required)
  OPENAI_BASE_URL  (default https://api.openai.com/v1)
  OPENAI_MODEL     (default gpt-3.5-turbo)
  CORPUS           (default ./EATD-Corpus)
"""
import os, sys, time, json, random, ssl
import urllib.request

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_URL = BASE + "/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
CORP = os.environ.get("CORPUS", "./EATD-Corpus")
POLAR = ['negative', 'positive', 'neutral']

SYS = "你是一个帮助进行心理访谈数据增强的助手。"
TEMPLATE = """请在严格保持原意的前提下，用不同的词汇和句式重新表达下面这段心理访谈中受访者的回答，用于数据增强。请遵循以下五条原则：
1. 内容完整性：保留原话的核心意思与情绪，不改变实质内容，不引入新信息或新话题。
2. 口语自然性：保持口语化、自然、随意的表达，避免书面或正式腔调。
3. 尊重得体：保持得体、尊重的语气，不做不当或冒犯性改动。
4. 长度一致：改写后长度与原文相近，不要明显变长或变短。
5. 容忍不规整：口语语境下可保留少量口头语、停顿词、重复等不规整之处。
只输出改写后的中文句子本身，不要任何解释、前缀或引号。

原话：{answer}"""


def call_llm(answer, retries=6):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": TEMPLATE.format(answer=answer)}],
        "temperature": 0.7, "max_tokens": 2000,
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=body, headers={
                "Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                resp = json.loads(r.read().decode("utf-8"))
            out = resp["choices"][0]["message"]["content"].strip().strip('"').strip('“”').replace("\n", " ").strip()
            if out:
                return out
        except Exception as e:
            sys.stderr.write(f"  retry {attempt+1}: {e}\n"); sys.stderr.flush()
            time.sleep(random.uniform(2, 6))
    return None


def main():
    if not API_KEY:
        sys.exit("set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL)")
    subjects = [f"{CORP}/{p}_{i}" for p in ('t', 'v') for i in range(300) if os.path.isdir(f"{CORP}/{p}_{i}")]
    print(f"subjects: {len(subjects)}", flush=True)
    done, total = 0, len(subjects) * 3
    for d in subjects:
        for pol in POLAR:
            out_file = f"{d}/{pol}_synthetic.txt"
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                done += 1; continue
            answer = open(f"{d}/{pol}.txt", encoding="utf-8").read().strip()
            syn = call_llm(answer) or answer        # never leave empty; fall back to original
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(syn + "\n")
            done += 1
            if done % 20 == 0:
                print(f"{done}/{total}", flush=True)
            time.sleep(0.2)
    print(f"DONE {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
