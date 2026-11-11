"""F04: a blind, source-grouped exam for a verifier, not a self-confidence score."""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Callable

from .common import TrialJournal, clone_json, digest, identifier, mean, ratio

_PRIVATE_KEYS = frozenset({"gold", "ground_truth", "private_annotations", "annotations",
                           "candidate_correct", "expected_verdict", "answer_key",
                           "observable", "is_corrupted", "is_error"})
_VERDICTS = frozenset({"correct", "incorrect", "abstain"})


def _public_input(value: Any, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            if key in _PRIVATE_KEYS:
                raise ValueError(f"Private annotation key {path}.{key} cannot reach verifier")
            _public_input(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _public_input(child, f"{path}[{i}]")


def _validate(cases: list[dict], modality: str) -> list[dict]:
    if modality not in {"text", "vision"}:
        raise ValueError("modality must be 'text' or 'vision'")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen, clean = set(), []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        identity = identifier(case.get("id"), "case.id")
        if identity in seen:
            raise ValueError(f"Duplicate case.id: {identity}")
        seen.add(identity)
        original = identifier(case.get("original_id"), "case.original_id")
        material = case.get("input")
        if not isinstance(material, dict):
            raise ValueError("case.input must be an object")
        _public_input(material)
        gold = case.get("gold")
        if not isinstance(gold, dict) or type(gold.get("candidate_correct")) is not bool:
            raise ValueError("gold.candidate_correct must be a boolean")
        observable = gold.get("observable")
        if not isinstance(observable, dict) or type(observable.get(modality)) is not bool:
            raise ValueError(f"gold.observable.{modality} must be a boolean")
        origin = case.get("origin", "unspecified")
        if not isinstance(origin, str) or not origin:
            raise ValueError("case.origin must be a string when provided")
        clean.append(clone_json({"id": identity, "original_id": original,
                                 "input": material, "gold": gold, "origin": origin}))
    return clean


def _metrics(rows: list[dict]) -> dict:
    observable = [r for r in rows if r["observable"]]
    good = [r for r in observable if r["candidate_correct"]]
    bad = [r for r in observable if not r["candidate_correct"]]
    count = lambda selected, verdict: sum(r["verdict"] == verdict for r in selected)
    result = {
        "trials": len(rows), "observable_trials": len(observable),
        "unobservable_trials": len(rows) - len(observable),
        "good_candidates": len(good), "bad_candidates": len(bad),
        "detected_bad": count(bad, "incorrect"),
        "wrongly_accepted_bad": count(bad, "correct"),
        "bad_abstentions": count(bad, "abstain"),
        "retained_good": count(good, "correct"),
        "false_alarms": count(good, "incorrect"),
        "good_abstentions": count(good, "abstain"),
        "verifier_errors": sum(r["status"] == "error" for r in rows),
        "observable_verifier_errors": sum(r["status"] == "error" for r in observable),
    }
    result.update({
        "error_detection_rate": ratio(result["detected_bad"], len(bad)),
        "wrong_accept_rate": ratio(result["wrongly_accepted_bad"], len(bad)),
        "false_alarm_rate": ratio(result["false_alarms"], len(good)),
        "correct_retention_rate": ratio(result["retained_good"], len(good)),
        "observable_abstention_rate": ratio(result["bad_abstentions"] +
                                              result["good_abstentions"], len(observable)),
    })
    return result


def run_blind_exam(cases: list[dict], verifier: Callable[[dict], dict | str], *,
                   seed: int = 0, modality: str = "text", trial_path=None,
                   verifier_metadata: dict | None = None) -> dict:
    """Run a source-grouped verifier exam with private gold kept out of its input.

    Case format::

      {"id":"private-case-id", "original_id":"source-group", "input":{...},
       "gold":{"candidate_correct":true,"observable":{"text":true,"vision":true}},
       "origin":"natural|injected|..."}

    The verifier receives **only** ``{trial_id, modality, input}``. A verdict is
    ``correct``, ``incorrect`` or ``abstain`` (a string or an object containing
    ``verdict``). ``abstain`` is reported separately: neither a detected error
    nor a correct accepted result. Unobservable cases are run and listed but
    excluded from detection/false-alarm denominators. Cases sharing an original
    are grouped; no token-level CI or p-value is manufactured.

    This is interface isolation, not a sandbox for an untrusted Python callback.
    Public evidence must independently be reviewed for answer leakage.
    """
    clean = _validate(cases, modality)
    if not callable(verifier):
        raise TypeError("verifier must be callable")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    metadata = clone_json({} if verifier_metadata is None else verifier_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("verifier_metadata must be an object")
    study_hash = digest({"cases": clean, "seed": seed, "modality": modality,
                         "verifier_metadata": metadata,
                         "protocol": "blind-verifier/1"})
    order = list(range(len(clean)))
    random.Random(seed).shuffle(order)
    rows = []
    with TrialJournal(trial_path, {"protocol": "blind-verifier/1", "study_sha256": study_hash,
                                  "seed": seed, "modality": modality,
                                  "verifier_metadata": metadata,
                                  "planned_trials": len(clean)}) as journal:
        for position, index in enumerate(order):
            case = clean[index]
            trial_id = "trial-" + digest({"study": study_hash, "position": position})[:20]
            public = {"trial_id": trial_id, "modality": modality,
                      "input": clone_json(case["input"])}
            row = {"record_type": "trial", "trial_id": trial_id,
                   "case_id": case["id"], "original_id": case["original_id"],
                   "origin": case["origin"], "candidate_correct": case["gold"]["candidate_correct"],
                   "observable": case["gold"]["observable"][modality],
                   "public_input_sha256": digest(public), "status": "ok", "verdict": None}
            try:
                answer = verifier(clone_json(public))
                verdict = answer if isinstance(answer, str) else answer.get("verdict") if isinstance(answer, dict) else None
                if verdict not in _VERDICTS:
                    raise ValueError("Verifier verdict must be correct/incorrect/abstain")
                row["answer"] = clone_json(answer)
                row["verdict"] = verdict
            except Exception as exc:
                row["status"] = "error"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            rows.append(row)
            journal.write(row)

    groups, origins = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["original_id"]].append(row)
        origins[row["origin"]].append(row)
    grouped = {key: _metrics(value) for key, value in sorted(groups.items())}
    metrics = _metrics(rows)
    macro = {key: mean([g[key] for g in grouped.values()]) for key in
             ("error_detection_rate", "wrong_accept_rate", "false_alarm_rate",
              "correct_retention_rate", "observable_abstention_rate")}
    return {"protocol": "blind-verifier/1", "study_sha256": study_hash,
            "modality": modality, "seed": seed, "verifier_metadata": metadata, "metrics": metrics,
            "original_groups": len(groups), "per_original": grouped,
            "macro_by_original": macro,
            "per_origin": {key: _metrics(value) for key, value in sorted(origins.items())},
            "trials": rows, "planned_trials": len(clean), "recorded_trials": len(rows),
            "interpretation": "DEV/evaluation result; source groups are units, not independent variants. No production qualification or independence of model errors is implied."}
