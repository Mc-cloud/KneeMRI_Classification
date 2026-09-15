"""
knee_mri_labeler.py

Generates weak labels for 12 knee-MRI findings from free-text, multilingual
radiology reports using the Google Gemini API (free tier, Flash model).

Design notes
------------
- Checkpointed: every response is appended to the output JSONL file
  immediately. Re-running with the same --output file automatically skips
  study_ids already present, so a free-tier daily quota cutoff just means
  "run it again tomorrow" with no lost work.
- Rate-limited: sleeps between calls based on --rpm, and on a 429 / quota
  error it stops cleanly (progress is already saved) instead of crashing
  mid-write.
- --dry_run lets you exercise the whole pipeline (checkpointing, parsing,
  scoring) with fabricated responses and zero API calls, to shake out bugs
  before spending your free-tier quota on real requests.
- --mode validate additionally scores predictions against a gold-labels CSV
  (your 58 gold-labeled studies) and prints per-finding precision/recall/F1.

Setup
-----
    pip install google-genai pandas

    export GEMINI_API_KEY="your-key-from-aistudio.google.com"

Input CSV (--input) must have columns:
    study_id, report_text

Gold CSV (--gold, required for --mode validate) must have columns:
    study_id, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
    Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
(each 0/1)

Usage
-----
    # quick pipeline smoke test, no API calls, no API key needed
    python knee_mri_labeler.py --input reports.csv --output labels_test.jsonl --dry_run

    # validate against the 58 gold-labeled studies
    python knee_mri_labeler.py --mode validate --input gold_reports.csv \\
        --gold gold_labels.csv --output labels_gold.jsonl --rpm 8

    # full run over all reports (resumable across multiple days on free tier)
    python knee_mri_labeler.py --mode full --input all_reports.csv \\
        --output labels_full.jsonl --rpm 8
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

FINDING_KEYS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Recommended per-finding policy for mapping "uncertain" -> a training label,
# based on a diagnostic run against the 58 gold-labeled studies (batch_size=20,
# gemini-3.6-flash). For each finding we checked how many "uncertain"
# predictions actually corresponded to a gold-POSITIVE label - i.e. how much
# recall you'd lose by defaulting "not mentioned" to absent.
#
#   ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral OA, PF OA, Effusion,
#   Baker's, Contusion, Fracture: only 0-4 out of 58 gold-positive cases were
#   marked uncertain -> converting uncertain to absent recovers a lot of
#   training data at a small, acceptable cost.
#
#   Synovitis: 14/58 gold-positive cases were marked uncertain, and in every
#   sampled case the report genuinely never mentions synovitis at all - yet
#   the (image-derived) gold label is positive. This isn't a labeler mistake;
#   it looks like a systematic report-vs-image gap for this specific finding
#   (radiologists apparently often see synovitis on images but don't dictate
#   it), matching the ~82% report-vs-image agreement caveat in the project
#   brief. Converting uncertain->absent here would inject a large number of
#   confidently-wrong negatives at scale. Keep 'exclude' for Synovitis, or
#   consider dropping it from LLM-derived weak labels entirely and leaning on
#   the 58 gold labels alone for that head.
#
# Re-validate this table if you change the prompt, the model, or batch_size -
# it was derived empirically, not assumed.
RECOMMENDED_UNCERTAIN_POLICY = {
    "ACL": "as_absent",
    "MCL": "as_absent",
    "Medial Meniscus": "as_absent",
    "Lateral Meniscus": "as_absent",
    "Medial OA": "as_absent",
    "Lateral OA": "as_absent",
    "PF OA": "as_absent",
    "Effusion": "as_absent",
    "Synovitis": "exclude",
    "Baker's": "as_absent",
    "Contusion": "as_absent",
    "Fracture": "as_absent",
}

FINDING_DEFINITIONS = {
    "ACL": "Anterior cruciate ligament INJURY - any described abnormality of "
           "the native ligament or graft: sprain, partial tear, full tear, "
           "or discontinuity.",
    "MCL": "Medial collateral ligament INJURY - sprain, partial tear, or "
           "full tear.",
    "Medial Meniscus": "Any described tear of the medial meniscus (radial, "
           "horizontal, vertical, bucket-handle, root, complex, etc). "
           "Degeneration/signal change WITHOUT a described tear line = absent.",
    "Lateral Meniscus": "Same tear criteria as Medial Meniscus, lateral side.",
    "Medial OA": "Osteoarthritis of the MEDIAL tibiofemoral compartment - "
           "cartilage loss, joint space narrowing, osteophytes, or "
           "subchondral sclerosis explicitly localized to the medial "
           "compartment. If compartment is not specified, mark uncertain.",
    "Lateral OA": "Same criteria as Medial OA, localized to the LATERAL "
           "tibiofemoral compartment.",
    "PF OA": "Same criteria as Medial/Lateral OA, localized to the "
           "PATELLOFEMORAL compartment (patella and trochlea).",
    "Effusion": "Joint effusion / excess intra-articular fluid described as "
           "more than trace or physiologic.",
    "Synovitis": "Synovial thickening, synovial enhancement, or synovitis "
           "explicitly described.",
    "Baker's": "Popliteal (Baker's) cyst described, any size.",
    "Contusion": "Bone contusion / bone bruise - a bone marrow edema-like "
           "signal pattern attributed to trauma, WITHOUT a described "
           "fracture line.",
    "Fracture": "Any described fracture - trabecular/microtrabecular, "
           "stress, occult, or overt cortical fracture. A bone bruise / "
           "contusion description alone, with no fracture line, is NOT a "
           "fracture.",
}

SYSTEM_PROMPT = """You are a radiology report analyst. You will be given a
BATCH of multiple knee MRI reports in a single request, each in its original
language (the dataset spans 12 languages across 16+ sites). Do not translate
any report. Read each one in its original language.

For EACH report in the batch, independently determine, for each of the 12
target findings listed below, whether the report indicates the finding is
PRESENT, ABSENT, or UNCERTAIN in that patient's knee at the time of that exam.
Treat each report completely independently - do not let one report's content
influence another's labels.

Rules (apply to every report in the batch):
1. Base your answer only on what that report states. Do not infer beyond the
   text. If the report does not mention a finding at all, and does not imply
   a normal/unremarkable exam covering that structure, mark it UNCERTAIN
   rather than ABSENT.
2. Pay close attention to negation scope ("no evidence of X", "X is not
   seen", "cannot exclude X") and to which anatomical structure a negation
   applies to in multi-clause sentences.
3. Distinguish CURRENT findings from PRIOR/HISTORICAL findings or surgical
   history (e.g. "s/p ACL reconstruction", "old meniscal tear, post-repair").
   Set surgical_history_noted = true whenever the report references prior
   surgery/intervention on the relevant structure, and base `status` on the
   structure's CURRENT described state.
4. For ACL and MCL: "injury" is broader than "tear" - sprains count too.
5. For Medial OA / Lateral OA / PF OA: only mark PRESENT if the degenerative
   change is explicitly localized to that compartment. If the report says
   "osteoarthritic changes" without specifying compartment, mark all three
   UNCERTAIN rather than guessing.
6. For Contusion vs Fracture: explicit fracture line/trabecular fracture ->
   Fracture PRESENT. Bone marrow edema/contusion with no fracture line ->
   Contusion PRESENT, Fracture ABSENT. Genuinely ambiguous -> both UNCERTAIN.
7. If a report's text appears truncated or missing an expected
   findings/impression section, set that report's report_quality =
   "suspected_incomplete" and prefer UNCERTAIN for findings that section
   would normally confirm.
8. Every non-uncertain status must include a short verbatim `evidence_quote`
   (<=15 words) copied exactly from that report in its original language. If
   you cannot produce a supporting quote, use UNCERTAIN instead.
9. Keep `reasoning` to a short phrase, <=10 words, not a full sentence.
10. Output ONLY a JSON array, one object per report in the batch. Each
    object's `study_id` field must exactly match the study_id given for that
    report below - this is how your output gets matched back to the input,
    so get it exact. The order of objects in the array does not matter as
    long as study_id is correct. No preamble, no markdown fences, no
    commentary outside the JSON array.

Finding definitions:
""" + "\n".join(f"- {k}: {v}" for k, v in FINDING_DEFINITIONS.items())


def build_item_schema():
    """Schema for a single report's result within the batch array."""
    finding_obj = {
        "type": "OBJECT",
        "properties": {
            "status": {"type": "STRING", "enum": ["present", "absent", "uncertain"]},
            "surgical_history_noted": {"type": "BOOLEAN"},
            "evidence_quote": {"type": "STRING"},
            "reasoning": {"type": "STRING"},
        },
        "required": ["status", "surgical_history_noted", "evidence_quote", "reasoning"],
    }
    return {
        "type": "OBJECT",
        "properties": {
            "study_id": {"type": "STRING"},
            "report_language_detected": {"type": "STRING"},
            "report_quality": {"type": "STRING", "enum": ["complete", "suspected_incomplete"]},
            "findings": {
                "type": "OBJECT",
                "properties": {k: finding_obj for k in FINDING_KEYS},
                "required": FINDING_KEYS,
            },
        },
        "required": ["study_id", "report_language_detected", "report_quality", "findings"],
    }


def build_batch_response_schema():
    return {"type": "ARRAY", "items": build_item_schema()}


def build_batch_user_prompt(batch):
    """batch: list of {"study_id": ..., "report_text": ...}"""
    findings_list = ", ".join(FINDING_KEYS)
    parts = [f"There are {len(batch)} reports in this batch.\n"]
    for i, row in enumerate(batch, 1):
        parts.append(
            f"--- Report {i} - study_id: {row['study_id']} ---\n"
            f"\"\"\"\n{row['report_text']}\n\"\"\"\n"
        )
    parts.append(
        f"Return a JSON array with exactly {len(batch)} objects, one per "
        f"report above, each with study_id set to that report's exact "
        f"study_id, covering exactly these 12 findings: {findings_list}"
    )
    return "\n".join(parts)


@dataclass
class RateLimiter:
    rpm: int
    _last_call: float = field(default=0.0, init=False)

    def wait(self):
        if self.rpm <= 0:
            return
        min_interval = 60.0 / self.rpm
        elapsed = time.time() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.time()


class QuotaExceeded(Exception):
    pass


class ModelUnavailable(Exception):
    pass


def call_gemini_batch(client, model, batch, max_retries=6):
    """Calls the Gemini API once for a whole batch of reports. Returns a list
    of result dicts (one per report). Raises QuotaExceeded / ModelUnavailable
    as before. Raises ValueError if the response isn't a parseable JSON array
    (e.g. truncated output) - caller should consider a smaller --batch_size
    if this happens repeatedly.
    """
    from google.genai import types  # imported here so --dry_run needs no SDK

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=build_batch_response_schema(),
        temperature=0,
    )
    prompt = build_batch_user_prompt(batch)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config,
            )
            parsed = json.loads(response.text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON array, got {type(parsed)}")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse batch response as JSON (likely truncated - "
                f"try a smaller --batch_size): {e}"
            ) from e
        except Exception as e:  # noqa: BLE001 - SDK exception types vary by version
            msg = str(e).lower()
            if "404" in msg or "not_found" in msg or "no longer available" in msg:
                raise ModelUnavailable(str(e)) from e
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                if attempt >= 2:
                    raise QuotaExceeded(str(e)) from e
                sleep_s = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_s)
                continue
            if "503" in msg or "unavailable" in msg or "overloaded" in msg:
                # Transient server-side overload, not our fault and not
                # quota-related - Google's own advice is just to wait and
                # retry, so give this more attempts and a longer backoff
                # than the generic path (up to ~2 minutes total here).
                if attempt == max_retries - 1:
                    raise
                sleep_s = min(60, (2 ** (attempt + 2))) + random.uniform(0, 2)
                print(f"  [retry] model reports high demand (503), waiting "
                      f"{sleep_s:.0f}s before retry {attempt + 1}/{max_retries}...",
                      file=sys.stderr)
                time.sleep(sleep_s)
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError("unreachable")


def preflight_check(client, model):
    """One cheap batched call before the main loop, so a bad model name or
    auth problem fails immediately and clearly instead of after 5 retries x
    every batch. NOTE: this itself consumes 1 of your daily request quota -
    skip it with --skip_preflight once you've confirmed the model works and
    are re-running to continue a multi-day job."""
    print(f"Preflight check: calling {model} once to confirm it's reachable "
          f"(uses 1 of your daily request quota)...")
    fake_batch = [{"study_id": "preflight", "report_text": "No abnormality."}]
    try:
        call_gemini_batch(client, model, fake_batch)
    except ModelUnavailable as e:
        sys.exit(
            f"\nModel '{model}' is not available: {e}\n"
            f"Check aistudio.google.com for the current Flash model name and "
            f"pass it with --model <name>."
        )
    except QuotaExceeded as e:
        sys.exit(f"\nQuota already exhausted before starting: {e}")
    print("Preflight check passed.\n")


def fabricate_dry_run_response(study_id):
    """Deterministic fake response for --dry_run, so the pipeline can be
    exercised without an API key or network access."""
    findings = {}
    for k in FINDING_KEYS:
        findings[k] = {
            "status": "absent",
            "surgical_history_noted": False,
            "evidence_quote": "",
            "reasoning": "dry_run placeholder",
        }
    return {
        "study_id": study_id,
        "report_language_detected": "unknown",
        "report_quality": "complete",
        "report_char_count": 9999,  # dry-run placeholder, well above any threshold
        "findings": findings,
    }


def load_done_study_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(str(rec.get("study_id")))
                except json.JSONDecodeError:
                    continue
    return done


def read_reports_csv(path, id_col, report_col):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames or report_col not in reader.fieldnames:
            raise SystemExit(
                f"Column not found in {path}. Looked for id_col='{id_col}' and "
                f"report_col='{report_col}'. Actual columns: {reader.fieldnames}\n"
                f"Pass --id_col / --report_col to match your file."
            )
        for row in reader:
            text = row[report_col]
            if text is None or text.strip() == "":
                continue  # skip studies with no report at all
            rows.append({"study_id": row[id_col], "report_text": text})
    return rows


def run_labeling(args):
    reports = read_reports_csv(args.input, args.id_col, args.report_col)
    n_total_input_rows = len(reports)

    if args.mode == "validate":
        gold_source = args.gold if args.gold else args.input
        gold = load_gold_csv(gold_source, args.id_col)
        reports = [r for r in reports if str(r["study_id"]) in gold]
        print(f"--mode validate: restricting labeling run to the "
              f"{len(reports)} gold-labeled studies "
              f"(ignoring the other {n_total_input_rows - len(reports)} rows in --input).")
    done_ids = load_done_study_ids(args.output)
    todo = [r for r in reports if str(r["study_id"]) not in done_ids]
    print(f"{len(reports)} total reports, {len(done_ids)} already labeled, "
          f"{len(todo)} remaining.")

    if not todo:
        print("Nothing to do - all reports in this range are already labeled.")
        return

    client = None
    if not args.dry_run:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("Set GEMINI_API_KEY in your environment before running "
                      "without --dry_run.")
        client = genai.Client(api_key=api_key)
        if not args.skip_preflight:
            preflight_check(client, args.model)

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    if args.max_calls is not None:
        batches = batches[: args.max_calls]
    print(f"{len(todo)} reports remaining -> {len(batches)} API call(s) "
          f"at batch_size={args.batch_size}"
          + (f" (capped by --max_calls={args.max_calls})" if args.max_calls is not None else "")
          + ".")

    limiter = RateLimiter(rpm=args.rpm)
    n_ok, n_err, n_calls = 0, 0, 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for batch in batches:
            batch_ids = {str(r["study_id"]) for r in batch}
            try:
                if args.dry_run:
                    results = [fabricate_dry_run_response(r["study_id"]) for r in batch]
                else:
                    limiter.wait()
                    results = call_gemini_batch(client, args.model, batch)
                n_calls += 1

                returned_ids = set()
                text_by_id = {str(r["study_id"]): r["report_text"] for r in batch}
                for result in results:
                    sid = str(result.get("study_id", ""))
                    if sid not in batch_ids:
                        print(f"[warn] response contained unexpected study_id "
                              f"{sid!r}, not in this batch - skipping it", file=sys.stderr)
                        continue
                    result["study_id"] = sid
                    result["report_char_count"] = len(text_by_id[sid])
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    returned_ids.add(sid)
                out_f.flush()

                missing = batch_ids - returned_ids
                if missing:
                    n_err += len(missing)
                    print(f"[warn] batch returned {len(returned_ids)}/{len(batch_ids)} "
                          f"expected studies; missing will be retried next run: {missing}",
                          file=sys.stderr)
                n_ok += len(returned_ids)
            except QuotaExceeded as e:
                print(f"\nDaily/rate quota appears exhausted after {n_calls} "
                      f"successful API call(s) ({n_ok} reports labeled) this run.\n"
                      f"Full error text (check WHICH quota metric was hit - "
                      f"requests/day, tokens/day, tokens/minute, etc):\n{e}\n\n"
                      f"Progress is saved in {args.output}. Re-run the same "
                      f"command later (e.g. tomorrow) to resume - already-"
                      f"labeled studies are automatically skipped.")
                break
            except ValueError as e:
                # likely a truncated/malformed batch response
                n_err += len(batch_ids)
                print(f"[error] batch failed to parse ({len(batch)} studies "
                      f"skipped, will retry next run): {e}", file=sys.stderr)
                continue
            except Exception as e:  # noqa: BLE001
                n_err += len(batch_ids)
                print(f"[error] batch failed ({len(batch)} studies skipped, "
                      f"will retry next run): {e}", file=sys.stderr)
                continue

    print(f"Done this run: {n_calls} API call(s), {n_ok} reports labeled, "
          f"{n_err} reports not labeled (will be retried next run). "
          f"Output so far: {args.output}")


# ---------------------------------------------------------------------------
# Gold-set scoring
# ---------------------------------------------------------------------------

def status_to_binary(status, uncertain_policy):
    """uncertain_policy: 'exclude', 'soft', or 'as_absent' - a resolved,
    concrete policy for a single finding (see resolve_policy for 'auto')."""
    if status == "present":
        return 1.0
    if status == "absent":
        return 0.0
    if status == "uncertain":
        if uncertain_policy == "exclude":
            return None
        if uncertain_policy == "soft":
            return 0.5
        if uncertain_policy == "as_absent":
            return 0.0
    return None


def resolve_policy(uncertain_policy, finding, report_quality=None, report_char_count=None, min_report_chars=200):
    """Resolves 'auto' to the empirically-derived per-finding recommendation;
    any other value is applied uniformly to every finding.

    Safety overrides (both force 'exclude' regardless of finding):
    1. report_quality == 'suspected_incomplete' - as_absent was validated on
       COMPLETE reports that simply don't mention a finding. A truncated
       report gives no such guarantee.
    2. report_char_count < min_report_chars - as_absent was validated on
       reports of roughly typical length (median ~980 chars in this
       dataset). A very short, single-finding report (e.g. "Hoffa fat pad
       impingement.", ~140 chars) may say nothing about the other 11
       findings simply because it never got that far, not because they're
       normal. About 2.4% of this dataset is under 150 chars - real, but
       not the majority, so exclude rather than guess for these.
    """
    if report_quality == "suspected_incomplete":
        return "exclude"
    if report_char_count is not None and report_char_count < min_report_chars:
        return "exclude"
    if uncertain_policy == "auto":
        return RECOMMENDED_UNCERTAIN_POLICY[finding]
    return uncertain_policy


def load_gold_csv(path, id_col):
    """Loads gold labels. Works both with a dedicated gold-only file and with
    the full train.csv (where most rows have blank finding columns for the
    ~4,349 unlabeled studies) - rows with any blank finding value are simply
    skipped rather than erroring, since that's how the real competition file
    marks 'no gold label for this study'.
    """
    gold = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames:
            raise SystemExit(
                f"Column '{id_col}' not found in {path}. "
                f"Actual columns: {reader.fieldnames}. Pass --id_col to match."
            )
        missing_finding_cols = [k for k in FINDING_KEYS if k not in reader.fieldnames]
        if missing_finding_cols:
            raise SystemExit(f"Gold source is missing finding columns: {missing_finding_cols}")
        n_skipped_blank = 0
        for row in reader:
            values = [row[k].strip() for k in FINDING_KEYS]
            if any(v == "" for v in values):
                n_skipped_blank += 1
                continue
            gold[str(row[id_col])] = {k: int(v) for k, v in zip(FINDING_KEYS, values)}
    print(f"Loaded {len(gold)} gold-labeled studies from {path} "
          f"({n_skipped_blank} rows skipped - no gold label).")
    return gold


def score_against_gold(output_path, gold_path, id_col, uncertain_policy="auto", min_report_chars=200):
    gold = load_gold_csv(gold_path, id_col)
    preds = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds[str(rec["study_id"])] = {
                "findings": rec["findings"],
                "report_quality": rec.get("report_quality"),
                "report_char_count": rec.get("report_char_count"),  # None for older files - fine, just skips this override
            }

    rows = []
    for finding in FINDING_KEYS:
        display_policy = resolve_policy(uncertain_policy, finding, report_quality=None)
        tp = fp = tn = fn = excluded = 0
        for study_id, gold_labels in gold.items():
            if study_id not in preds:
                continue
            pred_status = preds[study_id]["findings"][finding]["status"]
            report_quality = preds[study_id]["report_quality"]
            report_char_count = preds[study_id]["report_char_count"]
            policy = resolve_policy(uncertain_policy, finding, report_quality, report_char_count, min_report_chars)
            pred_val = status_to_binary(pred_status, policy)
            gold_val = gold_labels[finding]
            if pred_val is None:
                excluded += 1
                continue
            pred_bin = 1 if pred_val >= 0.5 else 0
            if pred_bin == 1 and gold_val == 1:
                tp += 1
            elif pred_bin == 1 and gold_val == 0:
                fp += 1
            elif pred_bin == 0 and gold_val == 0:
                tn += 1
            elif pred_bin == 0 and gold_val == 1:
                fn += 1
        n_scored = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) and precision == precision and recall == recall
              else float("nan"))
        accuracy = (tp + tn) / n_scored if n_scored else float("nan")
        rows.append({
            "finding": finding, "policy": display_policy, "n_scored": n_scored, "excluded_uncertain": excluded,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3) if precision == precision else precision,
            "recall": round(recall, 3) if recall == recall else recall,
            "f1": round(f1, 3) if f1 == f1 else f1,
            "accuracy": round(accuracy, 3) if accuracy == accuracy else accuracy,
        })
    return rows


def export_training_csv(predictions_path, export_path, id_col, uncertain_policy="auto", min_report_chars=200):
    """Converts a predictions JSONL (from --mode full) into a flat CSV of
    id_col + 12 binary finding columns, ready to merge into vision-model
    training. Cells where the resolved policy is 'exclude' and the
    prediction was 'uncertain' are left blank - your training loop should
    mask these out of the loss for that (study, finding) pair rather than
    treating a blank as 0.
    """
    n_rows = 0
    n_blank_by_finding = {k: 0 for k in FINDING_KEYS}
    with open(predictions_path, "r", encoding="utf-8") as f_in, \
         open(export_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([id_col] + FINDING_KEYS)
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = [rec["study_id"]]
            report_quality = rec.get("report_quality")
            report_char_count = rec.get("report_char_count")
            for finding in FINDING_KEYS:
                policy = resolve_policy(uncertain_policy, finding, report_quality, report_char_count, min_report_chars)
                status = rec["findings"][finding]["status"]
                val = status_to_binary(status, policy)
                if val is None:
                    row.append("")
                    n_blank_by_finding[finding] += 1
                else:
                    row.append(int(round(val)) if val in (0.0, 1.0) else val)
            writer.writerow(row)
            n_rows += 1
    print(f"Wrote {n_rows} rows to {export_path}.")
    print("Blank (excluded) cells per finding - mask these out of your loss:")
    for k, v in n_blank_by_finding.items():
        pct = 100 * v / n_rows if n_rows else 0
        print(f"  {k:18} {v:5} ({pct:.1f}%)")


def print_score_table(rows):
    headers = ["finding", "policy", "n_scored", "excluded_uncertain", "precision", "recall", "f1", "accuracy"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=False, help="CSV with an id column and a report-text column "
                   "(not needed for --mode export)")
    p.add_argument("--output", required=False, help="JSONL file to append predictions to (checkpoint file). "
                   "For --mode export, this is read as the INPUT predictions file.")
    p.add_argument("--export_csv", help="Output path for --mode export (the training-ready labels CSV)")
    p.add_argument("--mode", choices=["validate", "full", "export"], default="full")
    p.add_argument("--gold", help="CSV with id column + 12 finding columns (0/1). "
                   "Optional for --mode validate: if omitted, gold labels are read "
                   "straight out of --input (rows with blank finding columns are "
                   "treated as unlabeled and ignored).")
    p.add_argument("--id_col", default="StudyInstanceUID",
                   help="Name of the study-id column in --input and --gold (default: StudyInstanceUID)")
    p.add_argument("--report_col", default="Report",
                   help="Name of the report-text column in --input (default: Report)")
    p.add_argument("--model", default="gemini-3.6-flash",
                   help="Verify this model name is current in your AI Studio console before running")
    p.add_argument("--rpm", type=int, default=8, help="Max requests per minute (stay under your free-tier limit)")
    p.add_argument("--batch_size", type=int, default=25,
                   help="Number of reports sent per API call. Since the free tier caps you at a fixed "
                        "number of REQUESTS per day (not reports), a bigger batch means more reports "
                        "labeled per day. Start around 15-25; if you see JSON parse errors (truncated "
                        "output), lower it. If it works reliably, try raising it.")
    p.add_argument("--max_calls", type=int, default=None,
                   help="Stop after attempting this many NEW API calls (batches) this run - this maps "
                        "directly to your daily request quota. E.g. if your free-tier limit is 20 "
                        "requests/day, --max_calls 20 uses your whole daily budget in one run.")
    p.add_argument("--min_report_chars", type=int, default=200,
                   help="Reports shorter than this (in characters) never get as_absent/soft treatment "
                        "for 'uncertain' - always excluded instead, since a very short report may not "
                        "have gotten far enough to mention most findings either way (default: 200, "
                        "roughly the 5th percentile report length in this dataset)")
    p.add_argument("--skip_preflight", action="store_true",
                   help="Skip the 1-call preflight check (saves 1 unit of daily quota once you've "
                        "already confirmed the model/key work)")
    p.add_argument("--uncertain_policy", choices=["auto", "exclude", "soft", "as_absent"], default="auto",
                   help="How to map 'uncertain' predictions to a training label. 'auto' (default) uses "
                        "the empirically-derived per-finding policy in RECOMMENDED_UNCERTAIN_POLICY "
                        "(as_absent for most findings, exclude for Synovitis - see comment in source). "
                        "'exclude'/'soft'/'as_absent' force that single policy uniformly across all "
                        "12 findings, useful for re-testing the per-finding recommendation itself.")
    p.add_argument("--dry_run", action="store_true",
                   help="Exercise the full pipeline with fabricated responses, no API calls")
    args = p.parse_args()

    if args.mode in ("validate", "full") and not args.input:
        p.error("--input is required for --mode validate/full")
    if args.mode in ("validate", "full") and not args.output:
        p.error("--output is required for --mode validate/full")
    if args.mode == "validate" and not args.gold and not args.input:
        p.error("--mode validate requires --gold or --input with gold labels present")
    if args.mode == "export" and (not args.output or not args.export_csv):
        p.error("--mode export requires --output (the predictions JSONL to read) and --export_csv")

    if args.mode == "export":
        export_training_csv(args.output, args.export_csv, args.id_col, args.uncertain_policy, args.min_report_chars)
        return

    run_labeling(args)

    if args.mode == "validate":
        gold_source = args.gold if args.gold else args.input
        rows = score_against_gold(args.output, gold_source, args.id_col, args.uncertain_policy, args.min_report_chars)
        print("\nPer-finding scores vs. gold labels "
              f"(uncertain_policy={args.uncertain_policy}):\n")
        print_score_table(rows)


if __name__ == "__main__":
    main()