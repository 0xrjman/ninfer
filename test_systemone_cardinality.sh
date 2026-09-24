#!/usr/bin/env bash
# Cardinality test for POST /v1/systemone choice questions (Jev 255-option max).
# For N in {26,50,100,255}: one option is unambiguously correct from the state;
# checks probabilities sum ~= 1, argmax (choice) == ground truth, and confidence present.
set -euo pipefail
URL="${SYSTEMONE_URL:-http://127.0.0.1:8020/v1/systemone}"
MODEL="${SYSTEMONE_MODEL:-ninfer-jev}"
exec python3 - "$URL" "$MODEL" <<'PY'
import json, sys, urllib.request

url, model = sys.argv[1], sys.argv[2]
STATE = ("A customer contacts support and says their credit card payment just failed; "
         "they are asking for a refund of the money that was charged.")
CORRECT_KEY = "the_right_action"
CORRECT_DESC = ("Issue a refund for the failed credit card payment and return "
                "the amount that was charged to the card.")
TOPICS = ["a flight booking", "a hotel stay", "a job posting", "a pancake recipe",
          "a geometry proof", "a car repair", "a cinema ticket", "a grocery run",
          "a password reset", "a warranty claim", "a product review", "a tax form",
          "a gym plan", "a software licence", "a lease contract", "a travel visa"]
NOUNS = ["apples", "rockets", "guitars", "schedules", "invoices", "tickets",
         "orders", "reports", "photos", "cables", "shoes", "pizzas",
         "laptops", "paintings", "toys", "flowers"]

def distractor(i):
    return ("A completely unrelated task: handle %s involving %s #%d; it has nothing "
            "to do with payments or refunds." % (TOPICS[i % len(TOPICS)],
                                                 NOUNS[(i // len(TOPICS)) % len(NOUNS)], i))

def build(n):
    criteria = {"opt_%03d" % i: distractor(i) for i in range(n - 1)}
    criteria[CORRECT_KEY] = CORRECT_DESC  # 't' > 'o', so this sorts last
    return {"model": model, "state": STATE,
            "questions": {"q": {"type": "choice",
                                "instructions": ("Select the single action that exactly "
                                                 "matches the customer's request."),
                                "criteria": criteria}}}

def post(payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())

def check(n):
    try:
        ans = post(build(n))["answers"]["q"]
    except urllib.error.HTTPError as e:
        print("HTTP %s for N=%d: %s" % (e.code, n, e.read().decode()[:300]))
        raise SystemExit(1)
    probs = ans["probabilities"]
    total = sum(probs.values())
    ok_sum = abs(total - 1.0) < 1e-6
    ok_arg = ans.get("choice") == CORRECT_KEY
    conf = ans.get("confidence")
    ok_conf = isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
    return ok_sum, ok_arg, ok_conf, total, ans.get("choice"), conf

print("%4s | %12s | %13s | %10s" % ("N", "sum(probs)", "argmax==truth", "confidence"))
ok_all = True
for n in (26, 50, 100, 255):
    ok_sum, ok_arg, ok_conf, total, argmax, conf = check(n)
    if not (ok_sum and ok_arg and ok_conf):
        ok_all = False
    conf_s = ("%.4f" % conf) if isinstance(conf, (int, float)) else "MISSING"
    if not (ok_sum and ok_conf):
        conf_s += "*"
    print("%4d | %12.6f | %13s | %10s   (argmax=%s)" % (n, total, ok_arg, conf_s, argmax))
print()
print("ALL PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
PY
