# -*- coding: utf-8 -*-
"""
Paper section 3.3 "A Helping Hand from LLMs" text augmentation for DAIC-WOZ, with GPT-3.5.
Principle-based rephrasing (the paper's five principles, in English) of each participant
answer turn: for each turn we ask GPT-3.5 for THREE paraphrases, written " [SEP] "-joined to
`syn_answers.txt` per participant folder. Chunked, parallel, resumable.

The synthetic text features used by the model are bundled (`aug/text_syn_{train,dev}.npz`),
so this generator is only needed to regenerate the augmentation from scratch. Set OPENAI_API_KEY
(and optionally OPENAI_BASE_URL for an OpenAI-compatible endpoint) before running.
"""
import os, sys, json, time, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import pandas as pd

API_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

ROOT = "/DATA/COLA/chenzhuang/NAACL24-SEGA"
DW = f"{ROOT}/DAIC-WOZ"; EMB = f"{DW}/Embedding"

SYS = "You are an assistant that performs data augmentation for psychological-interview transcripts."
INSTR = """Below are numbered spoken answers from a participant in a clinical depression interview.
Rephrase EACH answer following these five principles:
1. Content integrity: keep the core meaning and emotion; do not add new information or topics.
2. Colloquial naturalness: keep it spoken, casual and natural; avoid formal/written tone.
3. Respect & appropriateness: keep a respectful, appropriate tone; no offensive changes.
4. Length consistency: keep roughly the same length as the original.
5. Tolerate disfluency: light fillers, pauses or repetitions are fine in spoken context.

For EACH input line produce exactly THREE different rephrasings joined by " [SEP] ".
Output EXACTLY one line per input line, in the SAME order, prefixed with its number and a tab,
like:  1\t<para1> [SEP] <para2> [SEP] <para3>
Output nothing else (no explanations, no blank lines).

Answers:
{block}"""


def call_gpt(block, retries=6, max_tokens=4000):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": INSTR.format(block=block)}],
        "temperature": 0.8, "max_tokens": max_tokens,
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=body, headers={
                "Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            sys.stderr.write(f"  retry {attempt+1}: {e}\n"); sys.stderr.flush()
            time.sleep(random.uniform(2, 6))
    return None


def parse(out, n):
    res = [None] * n
    if not out:
        return res
    for line in out.splitlines():
        m = re.match(r"^(\d+)[\t\.\):]\s*(.+)$", line.strip())
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n:
                res[i] = m.group(2).strip()
    return res


def do_participant(pid):
    p = f"{EMB}/merged_{pid}_TRANSCRIPT_embedding"
    outf = f"{p}/syn_answers.txt"
    ans = open(f"{p}/answers.txt", encoding="utf-8").read().splitlines()
    if os.path.exists(outf) and len(open(outf, encoding="utf-8").read().splitlines()) == len(ans):
        return f"skip {pid}"
    CH = 12
    parsed = [None] * len(ans)
    for st in range(0, len(ans), CH):
        grp = ans[st:st + CH]
        pg = parse(call_gpt("\n".join(f"{j+1}\t{a}" for j, a in enumerate(grp))), len(grp))
        for j in range(len(grp)):
            parsed[st + j] = pg[j]
    for i in range(len(ans)):
        if not parsed[i] or "[SEP]" not in parsed[i]:
            pr = parse(call_gpt(f"1\t{ans[i]}", max_tokens=600), 1)[0]
            parsed[i] = pr if (pr and "[SEP]" in pr) else f"{ans[i]} [SEP] {ans[i]} [SEP] {ans[i]}"
    open(outf, "w", encoding="utf-8").write("\n".join(parsed) + "\n")
    return f"ok {pid} ({len(ans)} turns)"


def main():
    assert API_KEY, "set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL)"
    tr = pd.read_csv(f"{DW}/utils/train_split_Depression_AVEC2017.csv")["Participant_ID"].tolist()
    dev = pd.read_csv(f"{DW}/utils/dev_split_Depression_AVEC2017.csv")["Participant_ID"].tolist()
    ids = tr + dev
    print(f"participants: {len(ids)}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(do_participant, pid): pid for pid in ids}
        for fu in as_completed(futs):
            done += 1
            try:
                print(f"{done}/{len(ids)} {fu.result()}", flush=True)
            except Exception as e:
                print(f"{done}/{len(ids)} ERR {futs[fu]}: {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
