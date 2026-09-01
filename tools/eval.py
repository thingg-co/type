#!/usr/bin/env python3
"""Desktop evaluation of candidate models for Type's spell correction.

Runs llama-server for each GGUF in turn and replays the exact prompts the app uses
(see app/src/main/java/com/aosmith/board/llm/Prompts.kt). Reports accuracy on typo
cases, false positives on words that should be left alone, and latency.

usage: eval.py --server /path/to/llama-server model1.gguf [model2.gguf ...]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

WORD_SYSTEM = ("You are the spell checker of a phone keyboard. "
               "The user sends the previous words and the current word. "
               "The current word is usually misspelled. Reply with only the intended word, nothing else. "
               "If the current word is already a correctly spelled English word or a name, reply with it unchanged.")

SENTENCE_SYSTEM = ("You fix spelling mistakes in short messages typed on a phone. "
                   "Reply with only the corrected message, nothing else. "
                   "Keep the wording, casing and punctuation as they are, only fix misspelled words. "
                   "If there are no mistakes, repeat the message unchanged.")


def word_req(before, word):
    ctx = " ".join(before.split()[-5:]) or "(start of message)"
    return f"previous words: {ctx}\ncurrent word: {word}"


WORD_EXAMPLES = [
    (word_req("we walked to", "shcool"), "school"),
    (word_req("see you", "latre"), "later"),
    (word_req("thank you for", "everyhting"), "everything"),
    (word_req("the weather is", "sunny"), "sunny"),
    (word_req("I want to", "recieve"), "receive"),
    (word_req("it was", "realy"), "really"),
    (word_req("my friend", "Priya"), "Priya"),
    (word_req("do you", "rememebr"), "remember"),
    # contractions are in scope: fix when context calls for it, keep the bare word otherwise
    (word_req("i think", "were"), "we're"),
    (word_req("the dog wagged", "its"), "its"),
]
SENTENCE_EXAMPLES = [
    ("Th meeting is at nooon", "The meeting is at noon"),
    ("can you send me the recipt", "can you send me the receipt"),
    ("Good morning!", "Good morning!"),
]
WORD_GRAMMAR = "root ::= [A-Za-z] [A-Za-z'-]*"

# (context before the word, typed word, expected). Expected == typed means "leave it alone".
WORD_CASES = [
    ("I think we should", "seperate", "separate"),
    ("she is very", "beautifull", "beautiful"),
    ("let me know if you are", "availble", "available"),
    ("that was a", "wierd", "weird"),
    ("I will", "definitly", "definitely"),
    ("please", "acommodate", "accommodate"),
    ("the", "goverment", "government"),
    ("what", "occured", "occurred"),
    ("it was", "neccessary", "necessary"),
    ("I", "beleive", "believe"),
    ("thanks for the", "reccomendation", "recommendation"),
    ("see you", "tomorow", "tomorrow"),
    ("he", "recieved", "received"),
    ("", "Wednsday", "Wednesday"),
    ("it is", "untill", "until"),
    ("the", "enviroment", "environment"),
    ("our", "buisness", "business"),
    ("I", "cant", "can't"),
    ("running", "thru", "through"),
    ("did you", "finsih", "finish"),
    ("send it to", "hte", "the"),
    ("we are", "goign", "going"),
    ("I am", "hapy", "happy"),
    ("come", "hoem", "home"),
    ("what is", "teh", "the"),
    ("I love", "chocolat", "chocolate"),
    ("she", "beleived", "believed"),
    ("a", "calender", "calendar"),
    ("the", "libary", "library"),
    ("very", "embarassing", "embarrassing"),
    # keyboard-adjacent slips
    ("meet me at the", "ststion", "station"),
    ("I will call you", "latwr", "later"),
    ("good", "morninf", "morning"),
    ("see you", "tonighy", "tonight"),
    ("that sounds", "grwat", "great"),
    # ambiguous contractions: context should decide
    ("i think", "were", "we're"),
    ("they said we", "were", "were"),
    ("i think", "ill", "I'll"),
    ("he was very", "ill", "ill"),
    ("do you think", "its", "it's"),
    ("the dog wagged", "its", "its"),
    ("maybe", "lets", "let's"),
    ("she", "lets", "lets"),
    ("i hope", "well", "we'll"),
    ("that went", "well", "well"),
    ("if so", "id", "I'd"),
    # confusable homophones: context should decide (the app screens these with the
    # prediction net first; the model only sees the gray zone)
    ("is bigger", "then", "than"),
    ("see you", "then", "then"),
    ("look over", "their", "there"),
    ("i like", "their", "their"),
    ("thanks,", "your", "you're"),
    ("is", "your", "your"),
    ("i want some", "quite", "quiet"),
    ("do not", "loose", "lose"),
    # should be left alone
    ("my friend", "Priya", "Priya"),
    ("we use", "kubernetes", "kubernetes"),
    ("the", "quokka", "quokka"),
    ("send me the", "pdf", "pdf"),
    ("meet", "Alexei", "Alexei"),
    ("in", "Bangkok", "Bangkok"),
    ("running on", "arm64", "arm64"),
    ("check the", "changelog", "changelog"),
    ("it was", "lol", "lol"),
    ("a", "burrito", "burrito"),
]

SENTENCE_CASES = [
    ("I will meet you at the resturant tommorow", "I will meet you at the restaurant tomorrow"),
    ("can you beleive how expencive the tickets were", "can you believe how expensive the tickets were"),
    ("She sed the meeting is at 3pm on Wednsday.", "She said the meeting is at 3pm on Wednesday."),
    ("my adress is 12 main street", "my address is 12 main street"),
    ("Thanks for the recomendation, it was realy good", "Thanks for the recommendation, it was really good"),
    ("what time does the libary close tonite", "what time does the library close tonight"),
    ("This sentence has no mistakes.", "This sentence has no mistakes."),
    ("I'm goign to be late, sorry", "I'm going to be late, sorry"),
    # confusables in full sentences: this is the class the small models miss
    ("Your welcome.", "You're welcome."),
    ("I think were going to be late.", "I think we're going to be late."),
    ("its there problem not ours", "it's their problem not ours"),
    ("I want quite time to read", "I want quiet time to read"),
    ("Thanks for your help.", "Thanks for your help."),
    ("We were there yesterday.", "We were there yesterday."),
]


def wait_ready(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def chat(port, messages, max_tokens, grammar=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    ms = (time.time() - t0) * 1000
    text = out["choices"][0]["message"]["content"]
    return text, ms


def word_messages(before, word):
    msgs = [{"role": "system", "content": WORD_SYSTEM}]
    for u, a in WORD_EXAMPLES:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": word_req(before, word)})
    return msgs


def sentence_messages(text):
    msgs = [{"role": "system", "content": SENTENCE_SYSTEM}]
    for u, a in SENTENCE_EXAMPLES:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": text})
    return msgs


def clean_word(s):
    s = s.strip().strip("[]\"'.,!?:;").strip()
    return s.split()[0] if s else ""


def run_model(server, model, port, threads, jinja, ngl=99):
    cmd = [server, "-m", model, "--port", str(port), "-t", str(threads), "-c", "2048", "-ngl", str(ngl), "--log-disable"]
    if jinja:
        cmd.append("--jinja")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(port, proc)
        # warm up the shared prefix
        chat(port, word_messages("", "helo"), 4, WORD_GRAMMAR)
        fixed = fixed_total = 0
        kept = kept_total = 0
        lat = []
        misses = []
        for before, typed, expected in WORD_CASES:
            raw, ms = chat(port, word_messages(before, typed), 12, WORD_GRAMMAR)
            lat.append(ms)
            got = clean_word(raw)
            ok = got.lower() == expected.lower()
            if typed == expected:
                kept_total += 1
                kept += ok
            else:
                fixed_total += 1
                fixed += ok
            if not ok:
                misses.append(f"{typed}->{got} (want {expected})")
        s_ok = 0
        s_lat = []
        s_misses = []
        for text, expected in SENTENCE_CASES:
            raw, ms = chat(port, sentence_messages(text), 80)
            s_lat.append(ms)
            got = raw.strip().strip('"')
            if got == expected:
                s_ok += 1
            else:
                s_misses.append(f"{text!r} -> {got!r}")
        return {
            "model": model.split("/")[-1],
            "fixed": f"{fixed}/{fixed_total}",
            "kept": f"{kept}/{kept_total}",
            "word_ms": sum(lat) / len(lat),
            "sentences": f"{s_ok}/{len(SENTENCE_CASES)}",
            "sentence_ms": sum(s_lat) / len(s_lat),
            "misses": misses,
            "sentence_misses": s_misses,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-jinja", action="store_true")
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers; 0 keeps Metal free for a concurrent training run")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    results = []
    for m in args.models:
        print(f"== {m}", file=sys.stderr, flush=True)
        try:
            r = run_model(args.server, m, args.port, args.threads, not args.no_jinja, args.ngl)
        except Exception as e:
            print(f"   failed: {e}", file=sys.stderr)
            continue
        results.append(r)
        print(f"   fixed {r['fixed']}  kept {r['kept']}  word {r['word_ms']:.0f} ms  sentences {r['sentences']}  {r['sentence_ms']:.0f} ms", file=sys.stderr)
        for miss in r["misses"]:
            print(f"     - {miss}", file=sys.stderr)
        for miss in r["sentence_misses"]:
            print(f"     ~ {miss}", file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
