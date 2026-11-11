"""F09: explicitly marked DEV-only 2×2 observation interventions.

The callback can execute the same pipeline on modified observations, but this
module has no production integration and never pretends oracle material was an
actual model capture. It cannot establish a unique internal causal explanation.
"""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Callable

from .common import TrialJournal, clone_json, digest, finite_number, identifier, mean


def _validate(documents: list[dict]) -> list[dict]:
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a nonempty list")
    seen, clean = set(), []
    for document in documents:
        if not isinstance(document, dict) or document.get("split") != "dev":
            raise ValueError("Every oracle document must explicitly have split='dev'")
        identity = identifier(document.get("id"), "document.id")
        if identity in seen:
            raise ValueError(f"Duplicate document.id {identity}")
        seen.add(identity)
        original_id = identifier(document.get("original_id", identity), "document.original_id")
        material, corrections = document.get("input"), document.get("corrections")
        if not isinstance(material, dict) or not isinstance(corrections, dict):
            raise ValueError("document.input and corrections must be objects")
        for required in ("text", "roles"):
            if required not in material or required not in corrections:
                raise ValueError(f"Original and corrected {required!r} are required")
        if not isinstance(material["text"], (str, list)) or not isinstance(corrections["text"], (str, list)):
            raise ValueError("text observations must be strings or lists")
        if not isinstance(material["roles"], (dict, list)) or not isinstance(corrections["roles"], (dict, list)):
            raise ValueError("roles must be objects or lists")
        unknown = set(corrections) - {"text", "roles", "geometry"}
        if unknown:
            raise ValueError(f"Unknown intervention keys: {sorted(unknown)}")
        if "geometry" in corrections and "geometry" not in material:
            raise ValueError("A geometry correction needs an original geometry field")
        clean.append(clone_json({"id": identity, "original_id": original_id,
                                 "split": "dev", "input": material, "corrections": corrections}))
    return clean


def run_oracle_study(documents: list[dict], evaluator: Callable[[dict], dict | float], *,
                     mode: str = "dev", seed: int = 0, metric: str = "score",
                     trial_path=None, evaluator_metadata: dict | None = None) -> dict:
    """Evaluate original/corrected text × original/corrected roles on known DEV.

    Document::

      {"id":"doc", "original_id":"source-group", "split":"dev",
       "input":{"text":"observed OCR", "roles":{...}, "geometry":...},
       "corrections":{"text":"human transcript", "roles":{...},
                      "geometry":...}}

    ``geometry`` is optional and belongs to the roles/structure factor. Corrected
    observations are passed only as declared interventions; ``corrections`` or
    a separate expected answer are not passed to the evaluator. The callback
    receives ``{trial_id,input,research_mode,oracle}`` and returns a finite metric
    number or an object with the requested ``metric`` key. It must reset model
    generation state between trials and independently grade its output.

    This tool refuses production, calibration and blind inputs. Complete four-
    condition groups are required to compute effects; errored trials are not
    replaced by zero. Effects are descriptive, then averaged by original source
    group. No confidence intervals, production accuracy or unique neural causes
    are inferred from these manipulated examples.
    """
    if mode != "dev":
        raise ValueError("Oracle interventions are permitted only in explicit DEV mode")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    identifier(metric, "metric")
    clean = _validate(documents)
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    metadata = clone_json({} if evaluator_metadata is None else evaluator_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("evaluator_metadata must be an object")
    study_sha = digest({"documents": clean, "seed": seed, "metric": metric,
                        "evaluator_metadata": metadata, "protocol": "dev-oracle-factorial/1"})
    planned = [(index, text, roles) for index in range(len(clean))
               for text in (0, 1) for roles in (0, 1)]
    random.Random(seed).shuffle(planned)
    rows = []
    with TrialJournal(trial_path, {"protocol": "dev-oracle-factorial/1", "study_sha256": study_sha,
                                  "mode": "dev", "metric": metric, "seed": seed,
                                  "evaluator_metadata": metadata,
                                  "planned_trials": len(planned)}) as journal:
        for position, (index, text_factor, roles_factor) in enumerate(planned):
            document = clean[index]
            material = clone_json(document["input"])
            if text_factor:
                material["text"] = clone_json(document["corrections"]["text"])
            if roles_factor:
                material["roles"] = clone_json(document["corrections"]["roles"])
                if "geometry" in document["corrections"]:
                    material["geometry"] = clone_json(document["corrections"]["geometry"])
            marker = {"study": "dev-oracle-factorial/1", "enabled": bool(text_factor or roles_factor),
                      "text_factor": "corrected" if text_factor else "original",
                      "roles_factor": "corrected" if roles_factor else "original",
                      "not_model_capture": bool(text_factor or roles_factor),
                      "input_origin": "human_annotation_intervention" if text_factor or roles_factor
                                      else "unmodified_study_input"}
            trial_id = "oracle-" + digest({"study": study_sha, "position": position})[:20]
            public = {"trial_id": trial_id, "research_mode": "dev", "input": material,
                      "oracle": marker}
            row = {"record_type": "trial", "trial_id": trial_id, "document_id": document["id"],
                   "original_id": document["original_id"], "text_factor": text_factor,
                   "roles_factor": roles_factor, "oracle": marker,
                   "original_input_sha256": digest(document["input"]),
                   "intervened_input_sha256": digest(material), "status": "ok", "score": None}
            try:
                result = evaluator(clone_json(public))
                score = result.get(metric) if isinstance(result, dict) else result
                row["score"] = finite_number(score, f"evaluator.{metric}")
                row["evaluation"] = clone_json(result)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            rows.append(row)
            journal.write(row)

    by_document = defaultdict(dict)
    for row in rows:
        by_document[row["document_id"]][(row["text_factor"], row["roles_factor"])] = row
    effects = []
    incomplete = []
    for document in clean:
        trials = by_document[document["id"]]
        if len(trials) != 4 or any(r["status"] != "ok" for r in trials.values()):
            incomplete.append({"document_id": document["id"], "original_id": document["original_id"],
                               "missing_or_error_conditions": [f"text={a},roles={b}" for a, b in
                                   ((0, 0), (1, 0), (0, 1), (1, 1))
                                   if (a, b) not in trials or trials[(a, b)]["status"] != "ok"]})
            continue
        baseline = trials[(0, 0)]["score"]
        text = trials[(1, 0)]["score"]
        roles = trials[(0, 1)]["score"]
        both = trials[(1, 1)]["score"]
        effects.append({"document_id": document["id"], "original_id": document["original_id"],
                        "baseline": baseline, "text_corrected": text,
                        "roles_corrected": roles, "both_corrected": both,
                        "text_effect_at_original_roles": text - baseline,
                        "roles_effect_at_original_text": roles - baseline,
                        "joint_effect": both - baseline,
                        "interaction": both - text - roles + baseline})
    grouped = defaultdict(list)
    for row in effects:
        grouped[row["original_id"]].append(row)
    effect_keys = ("text_effect_at_original_roles", "roles_effect_at_original_text",
                   "joint_effect", "interaction")
    per_original = {original: {"complete_documents": len(group),
                              **{key: mean([r[key] for r in group]) for key in effect_keys}}
                    for original, group in sorted(grouped.items())}
    macro = {key: mean([r[key] for r in per_original.values()]) for key in effect_keys}
    return {"protocol": "dev-oracle-factorial/1", "mode": "dev", "study_sha256": study_sha,
            "metric": metric, "seed": seed, "evaluator_metadata": metadata,
            "planned_trials": len(planned), "recorded_trials": len(rows),
            "trial_errors": sum(r["status"] == "error" for r in rows),
            "complete_documents": len(effects), "incomplete_documents": incomplete,
            "original_groups": len({d["original_id"] for d in clean}),
            "complete_original_groups": len(per_original), "effects": effects,
            "per_original": per_original, "macro_by_original": macro, "trials": rows,
            "interpretation": "Oracle diagnostic effects on known DEV, not actual product quality. Remaining errors can involve geometry, schema, serialization, truncation, decoding, Qwen or resolver. Effects need not add; no unique internal cause is proved."}
