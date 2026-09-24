#!/usr/bin/env bash
# HARD 255-way discrimination test (expanded) for POST /v1/systemone.
# 20 English scenarios: the original 6 from test_systemone_hard.sh (verbatim)
# PLUS 14 new diverse hard scenarios. Each scenario: genuinely ambiguous state,
# ground truth pinned by an explicit checkable criterion in the instructions,
# and 2-4 near-miss distractors that each fail the criterion for a specific reason.
# Filler options pad every choice to N=255.
# Per scenario (N=255): correct = (argmax == the_right_action), confidence, top-3.
# Prints a compact per-scenario table + summary. Exit non-zero if ANY is incorrect@255.
set -euo pipefail
URL="${SYSTEMONE_URL:-http://127.0.0.1:8020/v1/systemone}"
MODEL="${SYSTEMONE_MODEL:-ninfer-jev}"
ENV="${ENV_FILE:-/home/rjman/Servers/local-ai/.env}"
set -a; . "$ENV"; set +a
if [ -z "${API_KEY:-}" ]; then echo "FATAL: API_KEY not set in $ENV" >&2; exit 2; fi
export API_KEY
exec python3 - "$URL" "$MODEL" "$API_KEY" <<'PY'
import json, sys, urllib.request, urllib.error

URL, MODEL, KEY = sys.argv[1], sys.argv[2], sys.argv[3]
CORRECT_KEY = "the_right_action"
N_FILLER = 250  # 1 correct + 4 near-misses + 250 filler = 255
TOPICS = ["a flight booking", "a hotel stay", "a job posting", "a pancake recipe",
          "a geometry proof", "a car repair", "a cinema ticket", "a grocery run",
          "a password reset", "a warranty claim", "a product review", "a tax form",
          "a gym plan", "a software licence", "a lease contract", "a travel visa"]
NOUNS = ["apples", "rockets", "guitars", "schedules", "invoices", "tickets",
         "orders", "reports", "photos", "cables", "shoes", "pizzas",
         "laptops", "paintings", "toys", "flowers"]

def filler(i):
    return ("A completely unrelated task: handle %s involving %s #%d; it has nothing "
            "to do with this ticket." % (TOPICS[i % len(TOPICS)],
                                         NOUNS[(i // len(TOPICS)) % len(NOUNS)], i))

# ============ ORIGINAL 6 (verbatim from test_systemone_hard.sh) ============
SCENARIOS = [
    {
        "name": "refund_vs_recharge",
        "state": ("Ticket #4471: 'I was charged $39.00 on March 3 for order #88912, but the "
                  "order never shipped and I cancelled it before it went out. Please refund me. "
                  "Also note my card has since been replaced, so the old number will not accept "
                  "any new charge.'"),
        "instructions": ("The customer EXPLICITLY requests a refund of the $39.00 charge. "
                         "Choose the single action that refunds that charge. Any option that "
                         "re-charges, re-ships, or updates a card is wrong."),
        "correct": "Issue a refund of $39.00 for order #88912 back to the customer.",
        "near": [
            ("nm1", "Re-ship the cancelled order #88912 so the customer receives the item."),
            ("nm2", "Re-attempt the $39.00 charge on the customer's new card to finish the sale."),
            ("nm3", "Ask the customer to update the card on file, then process the $39.00 again."),
            ("nm4", "Void the order record only, with no monetary refund issued."),
        ],
    },
    {
        "name": "bug_root_cause",
        "state": ("Bug report: 'When I open the Settings page the app crashes to the background. "
                  "Log: TypeError: Cannot read properties of undefined (reading 'profile'). I just "
                  "updated to v3.2.0 and I am signed in as a fresh account with no data yet.'"),
        "instructions": ("The crash log shows a TypeError reading '.profile' on undefined, which is "
                         "the concrete evidence of the root cause (an unguarded null/undefined "
                         "access). Choose the fix whose cause matches that TypeError. Rollbacks, "
                         "relogins, and cache clears do not address that root cause."),
        "correct": ("Add a null-guard around the '.profile' access so Settings renders for a fresh "
                    "account that has no profile data yet."),
        "near": [
            ("nm1", "Roll the app back to v3.1.9, the last version before the crash was reported."),
            ("nm2", "Sign the user out and back in to refresh the session token."),
            ("nm3", "Clear the app cache and storage, then relaunch."),
            ("nm4", "Ask the user to create a profile manually before opening Settings."),
        ],
    },
    {
        "name": "polite_but_critical",
        "state": ("Message: 'Hi, hope you are doing well! Sorry to bother you on a Sunday and I know "
                  "it is the weekend so I will happily wait for whatever is convenient for your team. "
                  "Just double-checking something at my leisure... actually, our production database is "
                  "down right now and we cannot serve traffic. Thanks so much!'"),
        "instructions": ("Classify urgency. Despite the polite, casual phrasing, the message states "
                         "production is down and cannot serve traffic. Choose the triage label that "
                         "marks this a top-priority (P1/critical) incident."),
        "correct": "Triage as P1 critical: production outage, escalate to on-call immediately.",
        "near": [
            ("nm1", "Triage as P3 low: a casual, polite request that can wait until next business day."),
            ("nm2", "Triage as P2: schedule a review during the next regular business hours."),
            ("nm3", "Triage as P4: treat it as an information request, no outage implied."),
            ("nm4", "Triage as P2: note the weekend and respond Monday with a status."),
        ],
    },
    {
        "name": "complaint_disguised_question",
        "state": ("Customer email: 'Could you possibly let me know whether it is expected that a "
                  "delivery promised for 2-3 business days is now sitting in transit for 9 days with no "
                  "tracking update? I am just wondering if there is a reason, since my event is in two "
                  "days. Can you tell me what is going on?'"),
        "instructions": ("Detect the intent behind the polite question. The customer is asking about a "
                         "9-day-delayed delivery ahead of a hard deadline; the true intent is a "
                         "complaint/escalation about a late delivery, not a neutral status request. "
                         "Choose the intent label for a delayed-delivery complaint/escalation."),
        "correct": ("Treat as a complaint/escalation: flag the 9-day delay and escalate to logistics "
                    "given the customer's deadline."),
        "near": [
            ("nm1", "Treat as a neutral status request: look up tracking and post the current status."),
            ("nm2", "Treat as a how-to question: explain how tracking works and how to check it."),
            ("nm3", "Treat as a general inquiry: acknowledge and give a standard delivery window."),
            ("nm4", "Treat as a courtesy question: confirm the order is on its way, no action."),
        ],
    },
    {
        "name": "exact_refund_amount",
        "state": ("Ticket: 'I paid $120.00 for 3 items (3 x $40.00). Two arrived damaged and were "
                  "returned; you already refunded those two ($80.00, i.e. 2 x $40.00). The third item is "
                  "also defective and I want it refunded too. How much should I receive in addition?'"),
        "instructions": ("Compute the ADDITIONAL refund due. Two of three $40.00 items were already "
                         "refunded ($80.00). The customer now requests a refund for only the remaining "
                         "one defective item. Choose the option that refunds exactly $40.00 (one "
                         "unrefunded item)."),
        "correct": "Refund exactly $40.00 for the one remaining defective item (the other two are done).",
        "near": [
            ("nm1", "Refund $80.00 to cover the new item and re-verify the two already refunded."),
            ("nm2", "Refund the full $120.00 for all three items."),
            ("nm3", "No additional refund, since every item has already been refunded."),
            ("nm4", "Refund $60.00, splitting the remaining amount across the open item."),
        ],
    },
    {
        "name": "midcycle_upgrade_proration",
        "state": ("Ticket: 'I want to move from the $20/month Basic plan to the $80/month Pro plan. I "
                  "am upgrading mid-cycle: I still have 12 days left on this month's Basic period. I just "
                  "want the switch to happen and to be charged correctly, not double.'"),
        "instructions": ("Determine the correct billing action for a mid-cycle upgrade. The customer "
                         "has 12 unused Basic days and wants correct, non-double charging. Choose the "
                         "action that applies a prorated credit for the unused Basic days against the "
                         "new Pro charge. A full-price switch, a cancel-first, or a wait-for-next-cycle "
                         "is wrong."),
        "correct": ("Apply a prorated credit for the 12 unused Basic days against the new Pro charge "
                    "so the customer is not double-billed."),
        "near": [
            ("nm1", "Switch to Pro immediately and charge the full $80.00 this cycle with no credit."),
            ("nm2", "Cancel Basic at cycle end first, then start Pro next cycle at $80.00."),
            ("nm3", "Hold the switch until the next billing cycle and do nothing now."),
            ("nm4", "Charge $80.00 now and bill the $20.00 Basic balance separately."),
        ],
    },
    # ============ 14 NEW DIVERSE SCENARIOS ============
    {
        "name": "medical_acs_triage",
        "state": ("Patient presents with 20 minutes of pressure-like chest discomfort radiating to the "
                  "left arm, with cold sweats and mild nausea, not relieved by rest; history of "
                  "hypertension. Triage criteria: any red flag (pressure quality, arm radiation, "
                  "diaphoresis, nausea) mandates STAT emergency workup."),
        "instructions": ("Triage per the red-flag criterion: the patient exhibits multiple ACS red flags "
                         "(pressure quality, arm radiation, diaphoresis, nausea). The criterion is: any "
                         "red flag present -> STAT emergency cardiac workup. Symptom management, a "
                         "routine ECG appointment, or discharge all fail the criterion."),
        "correct": ("Treat as suspected acute coronary syndrome: STAT emergency workup and immediate "
                    "cardiac escalation (ECG, troponin, activate cardiology)."),
        "near": [
            ("nm1", "Treat as indigestion: give an antacid, observe 30 minutes, discharge if it eases."),
            ("nm2", "Schedule a routine ECG and cardiology clinic visit within the next 1-2 weeks."),
            ("nm3", "Start an NSAID and an antiemetic, then discharge with follow-up instructions."),
            ("nm4", "Do a 12-lead ECG only if the pain persists beyond 60 more minutes."),
        ],
    },
    {
        "name": "employee_vs_contractor",
        "state": ("A worker uses their own laptop, chooses some of their own hours, and bills per "
                  "project. However: they work exclusively for this client, on the client's premises, "
                  "follow the client's daily schedule, and the client dictates the methods, sequencing, "
                  "and quality standards of the work."),
        "instructions": ("Classify the worker. The controlling criterion is the client's right of "
                         "control over HOW the work is done (methods, sequencing, daily schedule), not "
                         "merely the outcome. Directing methods/sequencing/schedule indicates an "
                         "employee. Per-project billing, own equipment, and flexible hours do NOT "
                         "override the control criterion."),
        "correct": ("Classify as an employee: the client controls methods, sequencing, and daily "
                    "schedule, which establishes the right of control."),
        "near": [
            ("nm1", "Classify as an independent contractor because they are paid per project."),
            ("nm2", "Classify as an independent contractor because they use their own equipment."),
            ("nm3", "Classify as a self-employed freelancer because they set some of their own hours."),
            ("nm4", "Classify as an agent whose work creates joint liability for both parties."),
        ],
    },
    {
        "name": "code_review_race",
        "state": ("Code review: two threads increment a shared counter with a read-modify-write that is "
                  "not synchronized. Log: the counter sometimes ends up LOWER than the number of "
                  "completed increments, and the deficit grows as the number of threads increases. "
                  "Reviewer note: 'the increment is split across two operations.'"),
        "instructions": ("Identify the root cause of the under-count. The criterion: a non-atomic "
                         "read-modify-write on a shared counter under concurrency is a data race, and "
                         "the fix must make the increment atomic (mutex or atomic type). Adding logs, a "
                         "retry loop, or a thread-local counter do not fix a shared-state race."),
        "correct": ("Fix the data race: make the counter increment atomic (or guard the "
                    "read-modify-write with a mutex)."),
        "near": [
            ("nm1", "Add extra log lines around the counter update so the deficit is captured in prod."),
            ("nm2", "Retry the increment in a loop until the value looks correct."),
            ("nm3", "Make the counter a thread-local variable so each thread counts separately."),
            ("nm4", "Use fewer threads on a faster machine to shrink the race window."),
        ],
    },
    {
        "name": "ecommerce_damaged_return",
        "state": ("Order #5521: the customer returns a blender, 'damaged in transit - the blade housing "
                  "is cracked', and requests a refund; the customer paid the return shipping. Warehouse "
                  "policy: returns that arrive undamaged are restocked; damaged/defective units go to "
                  "scrap; when damage occurred in transit, the seller reimburses the return shipping."),
        "instructions": ("Choose the handling that satisfies ALL three criteria: (1) refund the "
                         "customer; (2) the cracked blender must NOT be restocked (it is damaged -> "
                         "scrap); (3) the seller reimburses return shipping (transit damage). Any option "
                         "that restocks the cracked unit or charges the customer for return shipping is "
                         "wrong."),
        "correct": ("Refund the purchase price, scrap the cracked blender (do not restock), and "
                    "reimburse the customer's return shipping (transit damage is seller cost)."),
        "near": [
            ("nm1", "Refund the purchase price and restock the blender so inventory can be resold."),
            ("nm2", "Refund the purchase price, scrap the unit, but pass the return-shipping fee to the customer."),
            ("nm3", "Offer a store-credit instead of a refund, scrap the unit, seller pays shipping."),
            ("nm4", "Send a replacement from stock and keep the damaged unit in the warehouse as used stock."),
        ],
    },
    {
        "name": "incident_sev2_vs_sev3",
        "state": ("Status: the checkout API is returning 500s on 30% of requests; the other 70% "
                  "succeed. No data loss. Revenue impact is partial and ongoing, and customers can "
                  "retry the page. On-call runbook: Sev1 = full outage; Sev2 = significant degradation "
                  "affecting a large subset of users or a core flow with no workaround; Sev3 = minor "
                  "degradation with a workaround."),
        "instructions": ("Assign the severity per the runbook criterion. 30% of a core flow (checkout) "
                         "failing, with no true workaround (retrying is not a workaround), is "
                         "significant degradation of a large subset -> Sev2. Sev1 requires a full "
                         "outage (absent), and Sev3 requires a workaround (absent)."),
        "correct": ("Assign Sev2: significant degradation - 30% of checkout failing, core flow, no "
                    "true workaround."),
        "near": [
            ("nm1", "Assign Sev1: treat any 500s on a core flow as a full outage and page everyone."),
            ("nm2", "Assign Sev3: minor degradation since 70% of requests still succeed."),
            ("nm3", "Assign Sev4: low impact, log it and monitor without paging."),
            ("nm4", "Assign Sev2 but downgrade it to Sev3 after 30 minutes if no more tickets arrive."),
        ],
    },
    {
        "name": "support_cancel_intent",
        "state": ("Chat: 'Hi, just checking - I have been meaning to mention that my sister is now "
                  "covering my plan, so I do not need my subscription any more from this month on. Is "
                  "there anything I need to do on my side? Thanks so much, sorry for the hassle!'"),
        "instructions": ("Detect the true intent. The customer says they 'do not need my subscription "
                         "any more from this month on' - the intent is a CANCELLATION effective this "
                         "month, not a pause, a downgrade, a plan change, or an information request. "
                         "Choose the action that cancels effective this month."),
        "correct": ("Cancel the subscription effective this month (immediate cancellation; no charge for the remainder of the month)."),
        "near": [
            ("nm1", "Pause the subscription until the customer asks to resume."),
            ("nm2", "Downgrade to the free plan and keep the account active."),
            ("nm3", "Treat as an information request: explain how cancellation works and await confirmation."),
            ("nm4", "Cancel effective the start of the NEXT billing cycle, no change this month."),
        ],
    },
    {
        "name": "nli_entailment_vs_neutral",
        "state": ("Premise A: 'The team shipped the feature on Friday.' Hypothesis B: 'The team shipped "
                  "the feature on Friday without any bugs.'"),
        "instructions": ("Classify the NLI relation. Criterion: B is entailed by A only if A guarantees "
                         "B. A states the feature shipped on Friday; it says nothing about the presence "
                         "or absence of bugs, so B adds new information not in A. Choose the neutral "
                         "label (not entailed and not contradicted)."),
        "correct": ("Label NEUTRAL: A neither entails nor contradicts B; 'without any bugs' is not stated by A."),
        "near": [
            ("nm1", "Label ENTAILMENT: shipping a feature implies it was bug-free."),
            ("nm2", "Label CONTRADICTION: shipping implies bugs must have been present."),
            ("nm3", "Label ENTAILMENT because both sentences mention Friday."),
            ("nm4", "Label NEUTRAL only on the assumption that A explicitly said 'with bugs'."),
        ],
    },
    {
        "name": "sentiment_sarcasm",
        "state": ("Review: 'Wow, amazing. Another two-hour delay, just what I needed to ruin my day. The "
                  "airline really thinks of everything. Five stars of frustration.'"),
        "instructions": ("Classify the customer's true sentiment. Criterion: sentiment reflects the "
                         "speaker's underlying attitude, not the literal wording. 'Just what I needed "
                         "to ruin my day' and 'Five stars of frustration' are sarcastic markers; the "
                         "attitude is negative. Choose the negative label."),
        "correct": ("Classify NEGATIVE: the sarcasm inverts the positive phrasing; the customer is unhappy about the delay."),
        "near": [
            ("nm1", "Classify POSITIVE because of 'amazing' and 'five stars'."),
            ("nm2", "Classify NEUTRAL because the tone is playful rather than angry."),
            ("nm3", "Classify MIXED: positive words plus a delay, so average to neutral."),
            ("nm4", "Classify POSITIVE because 'the airline really thinks of everything' is praise."),
        ],
    },
    {
        "name": "warehouse_dedup_rule",
        "state": ("A fact table contains duplicate rows for the same order. Fields: event_time (when the "
                  "business event happened) and created_at (when the row was inserted into the "
                  "warehouse). One duplicate was backfilled late (earlier created_at) but carries the "
                  "LATER, corrected event_time. Data contract: the canonical row for an order is the one "
                  "with the latest business event."),
        "instructions": ("Choose the dedup rule. Criterion: the canonical record is the latest event_time "
                         "(business event), NOT the latest created_at (insertion time). The late "
                         "backfilled corrected row has the older created_at but the later event_time and "
                         "must win."),
        "correct": ("Keep the row with the maximum event_time per order (the late backfilled corrected event wins)."),
        "near": [
            ("nm1", "Keep the row with the maximum created_at per order (most recently inserted wins)."),
            ("nm2", "Drop all duplicates and keep the first row seen."),
            ("nm3", "Keep the row with the minimum event_time (the original event wins)."),
            ("nm4", "Flag every duplicate for manual review and keep none automatically."),
        ],
    },
    {
        "name": "insurance_peril",
        "state": ("Claim: water-damaged carpet after a supply pipe burst inside the house during a "
                  "freezing night. Policy wording: 'Water damage from a burst supply pipe is a covered "
                  "peril; flood water from a natural source (river, rain, storm surge) is excluded; "
                  "wear and tear is excluded.' A plumber's report confirms the burst supply pipe."),
        "instructions": ("Determine coverage per the policy criterion. A burst supply pipe is an "
                         "explicitly covered peril; the damage is neither flood water nor wear and tear "
                         "and is documented. Choose the option that pays the claim under the covered "
                         "peril."),
        "correct": ("Pay the claim: burst supply pipe is the documented covered peril."),
        "near": [
            ("nm1", "Deny under the flood exclusion because the damage involves water."),
            ("nm2", "Deny as wear and tear: the carpet was old and worn."),
            ("nm3", "Pay only 50% under the shared-peril discount provision."),
            ("nm4", "Offer a replacement carpet at depreciated value under the wear-and-tear clause."),
        ],
    },
    {
        "name": "scheduling_overlap",
        "state": ("A candidate has two meetings Thursday 2:00-3:00 PM: (a) an interview with the hiring "
                  "manager for the role they applied for; (b) the team's internal weekly sync, which "
                  "any member can attend and which is minuted. Calendar policy: on overlap, keep the "
                  "meeting that is one-off with unique consequences; the recurring, minuted meeting "
                  "yields."),
        "instructions": ("Resolve the conflict per the policy criterion. The interview is one-off with "
                         "unique consequences for the candidate (a missed interview is not retried); the "
                         "sync is recurring and minuted. Choose to keep the interview and decline the "
                         "sync."),
        "correct": ("Keep the interview; decline the weekly sync (it is recurring and minuted)."),
        "near": [
            ("nm1", "Keep the team sync; ask the hiring manager to reschedule the interview."),
            ("nm2", "Join both meetings at once and switch between the audio feeds."),
            ("nm3", "Move the interview to Friday without checking the hiring manager's availability."),
            ("nm4", "Decline the interview to protect the team's weekly sync."),
        ],
    },
    {
        "name": "coupon_stacking",
        "state": ("Cart subtotal $100. Available: a 10% off coupon (capped at $10), a $15 off coupon, "
                  "and free shipping for orders over $75. Store policy: exactly ONE percent/fixed "
                  "coupon may be applied per order; free shipping always stacks; percent coupons are "
                  "capped at their maximum."),
        "instructions": ("Choose the discount application per the policy criterion. Exactly one of the "
                         "two coupons may apply - the one with the greater saving ($15 off beats the "
                         "capped $10); free shipping stacks on top; the 10% coupon must NOT be applied "
                         "alongside the $15 coupon."),
        "correct": ("Apply the $15 off coupon plus free shipping; do NOT apply the 10% coupon (one-coupon rule)."),
        "near": [
            ("nm1", "Stack both coupons: 10% ($10) plus $15 off, for $25 total off plus free shipping."),
            ("nm2", "Apply the 10% coupon ($10) because it appears first in the list, plus free shipping."),
            ("nm3", "Apply both coupons and drop free shipping to honor a single-discount rule."),
            ("nm4", "Apply the $15 coupon and charge $5 shipping because free shipping would be a second discount."),
        ],
    },
    {
        "name": "security_least_privilege",
        "state": ("Incident review: a deployment script runs with the same admin database account used "
                  "by the on-call DBA. The script only needs to INSERT rows into one audit table. Team "
                  "access policy: credentials must be least-privilege per use; a shared admin credential "
                  "across uses is a policy violation even though no breach occurred."),
        "instructions": ("Choose the remediation per the policy criterion. The violation is the shared "
                         "admin credential; the least-privilege fix is a dedicated account with INSERT "
                         "only on the audit table. Rotating the admin password, moving the script, or "
                         "granting the script broad access do not satisfy least privilege per use."),
        "correct": ("Create a dedicated service account with INSERT-only permission on the audit table and use it in the deployment script."),
        "near": [
            ("nm1", "Rotate the admin account's password to reduce the exposure window."),
            ("nm2", "Move the deployment script onto the DBA's workstation so one person owns the credential."),
            ("nm3", "Disable the admin account and grant the script full read/write on all tables."),
            ("nm4", "Keep the shared credential but restrict the script to run once per day."),
        ],
    },
    {
        "name": "unit_conversion_shipping",
        "state": ("A US seller lists an item at '2 pounds' and ships to Canada. The courier quotes by "
                  "kilograms. The label is correct (2 lb). The seller asks how to present the weight to "
                  "the courier so it is neither under- nor over-stated. Reference: 1 lb = 0.4536 kg."),
        "instructions": ("Convert 2 pounds to kilograms for the courier quote. Criterion: 2 x 0.4536 = "
                         "0.9072 kg, which rounds to 0.91 kg. Choose the option that states 0.91 kg. "
                         "Options using 2 kg, 0.45 kg, or 1.8 kg are arithmetic errors."),
        "correct": "State the weight as 0.91 kg (2 lb x 0.4536 kg/lb, rounded to two decimals).",
        "near": [
            ("nm1", "State the weight as 2 kg, assuming pounds and kilograms are interchangeable."),
            ("nm2", "State the weight as 0.45 kg, using the per-pound factor without multiplying by 2."),
            ("nm3", "State the weight as 1.8 kg, doubling the already-converted value."),
            ("nm4", "State the weight as 0.9 kg, truncating instead of rounding 0.9072 kg."),
        ],
    },
]

def build_q(scenario, n):
    crit = {}
    for k, d in scenario["near"]:
        crit[k] = d
    crit[CORRECT_KEY] = scenario["correct"]
    if n > 5:
        for i in range(N_FILLER):
            crit["opt_%03d" % i] = filler(i)
    return {
        "model": MODEL, "state": scenario["state"],
        "questions": {"q": {"type": "choice", "instructions": scenario["instructions"],
                            "criteria": crit}},
    }

def post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())

def top3(probs):
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(k, v) for k, v in items[:3]]

def run(scenario, n):
    try:
        ans = post(build_q(scenario, n))["answers"]["q"]
    except urllib.error.HTTPError as e:
        print("HTTP %s for %s N=%d: %s" % (e.code, scenario["name"], n,
                                           e.read().decode()[:400]))
        raise SystemExit(1)
    except (KeyError, urllib.error.URLError) as e:
        print("ERROR for %s N=%d: %r" % (scenario["name"], n, e))
        raise SystemExit(1)
    probs = ans.get("probabilities", {})
    return {
        "argmax": ans.get("choice"),
        "correct": (ans.get("choice") == CORRECT_KEY),
        "conf": ans.get("confidence"),
        "top3": top3(probs),
    }

def fmt_top3(t):
    return "  ".join("%s=%.4f" % (k, v) for k, v in t)

def fmt_conf(c):
    return ("%.4f" % c) if isinstance(c, (int, float)) else "MISS"

rows = []
for sc in SCENARIOS:
    r255 = run(sc, 255)
    rows.append((sc, r255))

print("=" * 96)
print("HARD 255-way discrimination report (expanded)  (correct key: %s)" % CORRECT_KEY)
print("=" * 96)
for sc, r in rows:
    mark = "OK " if r["correct"] else "BAD"
    print("[%s] %s | conf=%s | top-3: %s" % (mark, sc["name"], fmt_conf(r["conf"]),
                                             fmt_top3(r["top3"])))

n_correct = sum(1 for _, r in rows if r["correct"])
incorrect = [sc["name"] for sc, r in rows if not r["correct"]]
lowconf = [sc["name"] for sc, r in rows
           if isinstance(r["conf"], (int, float)) and r["conf"] < 0.5]

print("=" * 96)
print("SUMMARY")
print("  correct@255 : %d / %d" % (n_correct, len(rows)))
print("  incorrect@255 : " + (", ".join(incorrect) if incorrect else "none"))
for sc, r in rows:
    if not r["correct"]:
        print("    - %s: argmax=%s conf=%s top-3: %s" %
              (sc["name"], r["argmax"], fmt_conf(r["conf"]), fmt_top3(r["top3"])))
print("  low-confidence (<0.5) : " + (", ".join(lowconf) if lowconf else "none"))
print("=" * 96)
sys.exit(0 if n_correct == len(rows) else 1)
PY
