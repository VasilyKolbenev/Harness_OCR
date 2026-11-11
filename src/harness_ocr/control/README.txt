Harness OCR controls F06 / F07 / F08
==================================

All interfaces use standard-library Python and JSON dictionaries; importing them
does not load a model. Supply the same SQLite path to all stores so quarantine
always overrides acceptance. SQLite WAL + synchronous FULL + transaction-bound
events provide durable, idempotent state. SHA detects accidental/torn edits; it
does not defend against an administrator rewriting both database and hashes.

TRUST BOUNDARY
HTTP/model callers must never provide actor_scopes, qualification configuration,
policy dictionaries, calibration labels or ranking features. The authenticated
service resolves authorization and policy IDs from server-owned state. A client
validated=True / confidence=1 / profile_qualified=True grants nothing.

F06 AUDIT AND DRIFT
AuditSampler({"default": 0.05, "rare-layout": 0.5}, private_seed, policy_id)
  .decide(original_group_id, stratum) -> selected, inclusion_probability,
  config_sha256, seed_id. The HMAC draw is stable across revisions/crops and
  strata, preventing repeated draws; strata and seed are server controlled.
AuditController(sampler, AuditLog(db)).plan(document_id, revision, binding,
  group_id=original_id, stratum="default", accepted=True) persists inclusion.
  .record_outcome(task_id, critical_error=bool, actor_id=..., actor_scopes=
  ["audit:write"], evidence_ids=[...]) needs reviewed evidence and is immutable.
  .summary() reports follow-up gaps and descriptive inverse-probability weights;
  missing follow-up and correlated revisions never produce a confident rate.
DriftMonitor(log, baseline_binding, baseline_acceptance_rate=..., ...)
  .observe(...) quarantines unknown versions, confirmed critical errors or a
  rolling acceptance drop. It never promotes a model/profile/card/policy and
  never clears quarantine automatically. clear_quarantine needs privileged
  policy:quarantine:clear and a logged reason. Old window errors can re-trigger.
Caller must attach human audit outcomes to monitor; stored outcomes alone do not
silently mutate a production policy.

F07 STRATEGY CARDS
StrategyStore(db).propose / approve / activate / revoke use separate server
strategy:propose / :approve / :activate / :revoke scopes. retrieve needs :read.
Payload schema (exact keys):
  {card_id, version, scope:{profile_id,profile_version,service_ids,field_keys},
   directives:[{action,field_keys,parameters}], evidence_sha256:[...],
   negative_findings:[{experiment_sha256,action,outcome}]}
There are no free text, document values, arbitrary prompts, URLs or tool code.
Identifiers are controlled ASCII tokens and explicit numeric versions. Reviewers
still must verify experiment references and generalization; a syntactically valid
card does not prove a gain. Active retrieval is exact profile/version/service and
field scope. New versions are immutable, old versions retain lifecycle history;
revoked cards cannot reactivate. Current runtime may implement fewer actions
than the declarative DSL: report unsupported directives, do not pretend they ran.

F08 DOCUMENT-CRITICAL RISK
Binding is EXACTLY:
  {profile_id,profile_version,subset_contract_id,model_bundle_sha256,
   policy_version,input_scope_version}
The engine model bundle hash must include pinned weights/tokenizer, backend,
dtype, pipeline and compiled profile/config SHA. IDs/versions are immutable.

DocumentRiskCalibrator(required_fields=[...],conditional_fields=[...],
  score_contract_id=..., target_error_rate=0.001, confidence=0.95, ...)
  .fit(records,binding) -> frozen, integrity-sealed candidate policy.
Record schema:
  {document_id,group_id,split:"calibration"|"qualification"|"test",
   risk_score:0..1,critical_error:bool,eligible:bool(optional),binding(optional)}
Use independently reviewed labels; one group is one original input including
all variants, pages, crops and revisions. Groups cannot cross partitions.
Threshold selection uses calibration only. .qualify(candidate, qualification,
binding) freezes that threshold and checks independent groups, coverage and an
exact one-sided Clopper-Pearson upper bound. Any selected critical error in a
group counts that group as an error. Zero errors in100 groups gives about2.95%
upper bound, not0%; a0.1% claim at95% requires at least2995 zero-error groups.
Reusing failed qualification data to search thresholds invalidates independence.
.evaluate(policy,test_records) requires a third held-out partition, disjoint from
both previous partitions. allow_known_groups=True is explicitly diagnostic.
No data / insufficient groups => insufficient_calibration_data and review.
Statistics apply to the fixed accepted population under representative,
independent sampling, not an individual document and not arbitrary shifted data.
No multiplication of field confidences or invented individual probabilities.

RiskPolicyStore(db).save_candidate / approve / activate / revoke require server
policy:propose / :approve / :activate / :revoke. Only independently qualified
policies can activate. .apply(policy_id,result,binding) requires ACTIVE stored
state, integrity, exact binding, no quarantine and all hard guards. Internal
calibrator.apply(policy_dict,...) is for offline use, never a public endpoint.
Required result fields: completed=True, terminal_status="completed", conflicts=[],
required_fields, triggered_conditional_fields, conditional_applicability,
fields and observations/evidence registry. Every mandatory, triggered or critical
field needs value, extraction_status="observed", validation_status="pass",
party_role and active completed source-SHA-bound evidence. Missing conditions,
incomplete work, conflicts, unsupported values or unknown score contract review.

SERVER RUNTIME BRIDGE
BindingPolicyGateway.from_file(data_dir/"runtime_policy.json", policy_store)
defaults to empty => review. Trusted configuration:
  {"schema_version":"harness-runtime-policy/1.0",
   "qualified_bindings":["exact SHA256(canonical binding)"],
   "policy_ids":{"same binding SHA256":"active stored policy ID"},
   "score_contract_id":"harness-hard-gate-rank-v1"}
This configuration is not shipped with qualified entries. Changing it is an
operator action after qualification, never automatic learning from corrections.
The bridge recomputes the fixed score: failed hard-check count divided by
(4+4*max(1,number of required-or-conditional fields)), capped at1. This is a rule
ranking feature, NOT an error probability. All passing documents have score0;
the package therefore makes no claim of learning to rank their optical errors.
Calibrate against these exact generated scores, not arbitrary input scores.
More informative ranks require a new contract and new independent qualification.
Stored policy lifecycle and sticky quarantine are checked live on every apply.

Acceptance means eligibility for autofill only. It never approves a government
service, proves a document authentic or changes independently observed raw text.

REVIEWER AUDIT OUTCOME -> QUARANTINE
The HTTP bridge is POST /v1/audits/{task_id}/outcome with HARNESS_REVIEW_TOKEN.
The analysis token has no permission, and MCP does not expose this operation.
Strict request shape:
  {"critical_error":true,"evidence_ids":["actual-observation-id"],
   "correction_revision":null}
Use an actually selected task_id from result.audit. Evidence must belong to that
document and remain active/completed; references from another document or partial
reads are rejected. If supplied, correction_revision must name a real authorized
human-correction event for the document, not an ordinary extraction revision.

A confirmed critical error appends the outcome, activates the exact binding's
quarantine and appends the quarantine event in ONE SQLite transaction. Active
F08 policies immediately become review-only; no later model call or monitoring
poll is needed to apply the safety action. Noncritical review only logs the
outcome. Neither action creates gold, retrains a model or promotes a policy.

Identical outcome requests are idempotent. Conflicting replacement content for
the same task fails closed. Replaying an already processed critical outcome
does not reverse a later explicit authorized quarantine clearance. A new audit
that confirms a new error can quarantine the binding again.
