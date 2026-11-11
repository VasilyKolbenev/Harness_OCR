Harness OCR research tools (F04, F05, F09)
========================================

These stdlib-only tools evaluate supplied callbacks. They do not load a neural
model, route production jobs, repair observations automatically, or establish
production quality. Keep study files in a separate, access-controlled directory:
trial journals can contain private annotations and raw document observations.

The adapters at the end of this file connect the runners to actual stage traces
and the same supervised model process as the engine. Importing the package still
does not load a neural model.

All three functions accept trial_path. A journal is created exclusively (mode x),
then each manifest/trial is flushed and fsynced. Existing files are never reused.
An interrupted journal is retained: start a deliberate new run with a new path.
Inputs must be finite JSON data; unsupported objects are not silently stringified.

F04: blind verifier exam
-----------------------
from harness_ocr.research import run_blind_exam

cases = [{"id": "private-case", "original_id": "original-document-1",
          "input": {"candidate": {"value": "123"},
                    "evidence": {"text": "123"}},
          "gold": {"candidate_correct": True,
                   "observable": {"text": True, "vision": True}},
          "origin": "natural"}]

def verifier(payload):
    material = payload["input"]
    return {"verdict": "correct" if material["candidate"]["value"] ==
            material["evidence"]["text"] else "incorrect"}

report = run_blind_exam(cases, verifier, seed=0, modality="text",
                        trial_path="new-blind-run.jsonl",
                        verifier_metadata={"model_sha256": "actual-frozen-sha",
                                           "prompt_sha256": "actual-frozen-sha"})

The example verifier only tests plumbing, not OCR quality. Use real frozen natural
and injected cases for an actual study. Verifier receives exactly trial_id,
modality and input; private IDs, origin, gold and observability are excluded.
Reserved annotation keys inside input are rejected recursively. Audit filenames,
text and candidate patterns yourself: no Python interface is a sandbox against
arbitrary callbacks or semantic answer leakage.

Verdicts: correct, incorrect, abstain. Abstaining is neither catching a bad value
nor retaining a good value. Invalid output/exception is an explicit error.
Unobservable cases are listed separately and excluded from detection/false-alarm
denominators. Metrics include overall counts, per-original-source metrics,
unweighted macro averages of source metrics and per-origin metrics. Multiple
variants of one document are not independent observations. No invented p-values.

F05: bounded same-failure graph reduction
----------------------------------------
from harness_ocr.research import minimize_graph

graph = {"task_contract": {"field": "payer.inn"},
         "nodes": [{"id": "image", "kind": "artifact", "sha256": "actual-sha"},
                   {"id": "claim", "source_ids": ["image"]},
                   {"id": "confuser"}, {"id": "noise"}]}

def predicate(candidate):
    ids = {node["id"] for node in candidate["nodes"]}
    return {"failed": {"claim", "confuser"} <= ids,
            "failure_id": "payer-payee-swap"}

report = minimize_graph(graph, predicate, required_ids=["claim"], max_calls=80,
                        time_budget_seconds=30, trial_path="new-reduction.jsonl")

The original failure is observed before reduction. Only an identical failure_id
can justify accepting a reduction. Source/artifact/provenance nodes, required
nodes, evidence references and protected evidence edges are preserved. Metadata
and task_contract are unchanged. Source pixels are not edited. The graph schema
uses nodes with id and optional edges with source,target,type. Evidence/source/
provenance edges point claim to source; supports points source to claim.

Budget includes the baseline. Time is measured by perf_counter and checked
between calls. An arbitrary Python callback cannot be forcibly interrupted here;
put expensive inference in the service process watchdog. A boolean predicate
must already mean one specific defect; structured failure IDs are preferred.
Results are local diagnostic reductions under that predicate, not a globally
smallest graph or proof that retained features are the neural cause.

F09: explicitly marked DEV-only 2x2 oracle study
-----------------------------------------------
from harness_ocr.research import run_oracle_study

documents = [{"id": "dev-1", "original_id": "original-document-1", "split": "dev",
              "input": {"text": "original OCR", "roles": {"inn": "payee"}},
              "corrections": {"text": "human transcript",
                              "roles": {"inn": "payer"}}}]

def evaluator(payload):
    # Run the same pipeline with a fresh generation state, then independently
    # grade it against annotations kept outside this payload.
    return {"score": 0.0}  # Example only: no measured quality claimed.

report = run_oracle_study(documents, evaluator, mode="dev", seed=0,
                          trial_path="new-oracle-run.jsonl",
                          evaluator_metadata={"model_sha256": "actual-frozen-sha"})

mode must be dev and every document must explicitly have split=dev. Blind,
calibration and production inputs are refused before evaluator invocation.
Four shuffled conditions: original/original, corrected-text/original-roles,
original-text/corrected-roles, corrected/corrected. Optional corrected geometry
belongs to the roles factor and requires original geometry. Each callback input
is explicitly labelled research_mode and oracle factors; corrected observations
are labelled human intervention and not_model_capture. They cannot be reported
as text recognized by a model. There is no oracle production default/import.

Evaluator returns a finite number or {metric: number}. It must reset generation
state independently for every trial. An errored/missing condition leaves that
document's factorial effects uncomputed, without zero imputation. Effects are
computed within complete documents, then averaged within original source before
macro averaging. Effects are descriptive interventions; they do not establish a
unique internal cause or guarantee transfer to independent production documents.

REAL PIPELINE ADAPTERS (architecture audit, offline only)
-------------------------------------------------------
from harness_ocr.research import (load_trace_snapshot, blind_case_from_trace,
    run_trace_blind_exam, minimize_trace_failure, run_trace_oracle_study)

Freeze the source trace first (do not edit it during reading). Stop a competing
GPU service before offline model inference. Keep a separate output directory.

snapshot = load_trace_snapshot("/private/frozen/trace.sqlite", "actual-document-id")

The adapter audits the hash chain, verifies the chosen record again and refuses
records appended during snapshot reading. It captures the candidate-stage input,
compiled contract, recorded candidates and response from the same round. This
does not authenticate the original document or a malicious database administrator.

F04: construct correct/injected cases from real trace candidates:
case = blind_case_from_trace(snapshot, case_id="private-case-1",
    original_id="original-1", field_key="actual.field", candidate_id="actual-ID",
    candidate_correct=True, observable={"text": True})
report = run_trace_blind_exam([case], supervised_backend, qwen="qwen-4b",
    trial_path="new-blind-trials.jsonl")

supervised_backend is an explicit harness_ocr.inference.supervisor.Supervisor.
Its pinned Qwen files are SHA-verified before the study; custom/test backends are
labelled unverified rather than mistaken for measured real models. The verifier
uses select(purpose="verifier"), a distinct fixed server system instruction and
strict correct/incorrect/abstain JSON. Private labels, private case IDs and
observability labels are not in the model prompt. proposed_raw may deliberately
inject a wrong literal for an exam; it is never relabelled as OCR evidence.
An incomplete or malformed generation is an error, not a successful abstention.
The ready adapter is TEXT ONLY. A vision callback API exists, but no visual
verifier adapter or visual-quality result is claimed by this release.

F05: reduce a real candidate/resolver-stage defect without another neural call:
def same_defect(stage):
    claims = stage.get("resolved", {}).get("claims", [])
    failed = any(c["field_key"] == "actual.field" and
                 c["raw_value"] == "known wrong literal" for c in claims)
    return {"failed": failed, "failure_id": "fixed-field-literal-error"}
report = minimize_trace_failure(snapshot, same_defect, max_calls=80,
    time_budget_seconds=30, trial_path="new-minimization.jsonl")

Candidates and role proofs are rebuilt from retained observations for every
predicate call. Old eligible role flags are never blindly copied after removing
their label evidence. The fixed recorded Qwen response is replayed; no source
pixels change and no new Qwen response is sampled. The task contract and source
hash nodes are retained. A fixed failure ID avoids replacing the target defect
with the trivial error of deleting a referenced candidate. The supplied predicate
is trusted Python code and must itself be deterministic and bounded.

F09: run the actual selector stage under declared DEV interventions:
documents = [{"id": "private-dev-1", "original_id": "source-1", "split": "dev",
    "input": {"text": original_value_region_observations,
              "roles": {"observation-id": {"field_key": "wrong.field",
                                           "party_role": "its-role"}}},
    "corrections": {"text": corrected_value_region_observations,
                    "roles": {"observation-id": {"field_key": "correct.field",
                                                 "party_role": "its-role"}}}}]
report = run_trace_oracle_study(documents, snapshot["contract"], supervised_backend,
    {"original-source-sha256": {"correct.field": "private expected literal"}},
    qwen="qwen-4b", trial_path="new-oracle-trials.jsonl")

The observation list contains actual source-bound VALUE regions. A role map
contains IDs, field keys and roles only, never a replacement value. Original and
corrected optical strings keep the same source/observation identity. Optional
geometry is an observation-ID mapping to exactly one bbox/quad within source
bounds and belongs to the roles factor. Every model prompt/result carries the
explicit DEV intervention marker; corrected text is human input, not recognition.
The private expected-literal mapping reaches only the scorer after generation.
This adapter calls the actual candidate/selector/resolver/validator stages with a
new generate call each time, but it does NOT route through production policies,
publish autoacceptance or write oracle observations into DocumentStore. Production
and non-DEV datasets are refused. This measures selector-stage interventions,
not full optical acquisition, service accuracy or a unique neural mechanism.

The integration tests use fixture backend outputs to verify these connections.
They do not establish that Qwen0.8B/4B notices errors or improves real documents.
