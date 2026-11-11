"""Server-owned bridge from exact binding qualification to active risk policies.

No profile or production policy is qualified by this package. An operator may
install this configuration only after independent qualification. HTTP callers and
models never supply it. A ranking feature is deterministically recomputed here;
caller confidence, risk_score and validated flags cannot authorize acceptance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

from .common import digest, require_id, validate_binding
from .risk import RiskPolicyStore, _hard_guard_reasons


SCORE_CONTRACT_ID = "harness-hard-gate-rank-v1"
CONFIG_SCHEMA = "harness-runtime-policy/1.0"


def hard_gate_rank(result: dict, hard_contract: dict) -> dict:
    """A fixed rule count, never a probability or product of field confidences.

All hard-check-passing documents have score0. Qualification therefore tests the
complete fixed gate, including its errors on independently annotated documents;
the feature makes no claim to distinguish correct and incorrect passing OCR.
More informative ranks require a new score contract and new qualification.
"""
    reasons = _hard_guard_reasons({"hard_contract": hard_contract}, result)
    nfields = len(set(hard_contract["required_fields"] + hard_contract["conditional_fields"]))
    denominator = 4 + 4 * max(1, nfields)
    return {"score_contract_id": SCORE_CONTRACT_ID, "risk_score": min(1.0, len(reasons) / denominator),
            "risk_score_features": {"hard_guard_failure_count": len(reasons),
                "normalizer": denominator, "hard_guard_reason_codes": reasons},
            "score_is_probability": False}


class BindingPolicyGateway:
    def __init__(self, config: dict | None, store: RiskPolicyStore):
        self.store = store
        config = copy.deepcopy(config) if config is not None else {}
        if not isinstance(config, dict) or set(config) - {
                "schema_version", "qualified_bindings", "policy_ids", "score_contract_id"}:
            raise ValueError("Unknown server runtime policy configuration")
        if config.get("schema_version", CONFIG_SCHEMA) != CONFIG_SCHEMA:
            raise ValueError("Unsupported runtime policy schema")
        if config.get("score_contract_id", SCORE_CONTRACT_ID) != SCORE_CONTRACT_ID:
            raise ValueError("Runtime cannot fabricate or execute an unknown score contract")
        qualified = config.get("qualified_bindings", [])
        policies = config.get("policy_ids", {})
        if not isinstance(qualified, list) or len(qualified) > 10000 or not isinstance(policies, dict):
            raise ValueError("Invalid bounded runtime qualification records")
        for sha in qualified + list(policies):
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("Runtime qualification keys must be exact binding SHA256")
        for ident in policies.values():
            require_id(ident, "active server policy ID")
        if not set(policies) <= set(qualified):
            raise ValueError("Policy mapping has no server binding qualification")
        self.qualified = frozenset(qualified)
        self.policy_ids = policies

    @classmethod
    def from_file(cls, path: str | Path, store: RiskPolicyStore):
        path = Path(path)
        config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        return cls(config, store)

    def apply(self, result: dict, binding: dict) -> dict:
        binding = validate_binding(binding)
        key = digest(binding)
        reasons = []
        ident = self.policy_ids.get(key)
        if key not in self.qualified:
            reasons.append("profile_not_qualified")
        if ident is None:
            reasons.append("no_server_runtime_policy")
        policy = None
        if ident:
            try:
                policy = self.store.get(ident)["policy"]
            except (KeyError, ValueError):
                reasons.append("unknown_or_invalid_server_policy")
        if policy and policy.get("score_contract_id") != SCORE_CONTRACT_ID:
            reasons.append("unsupported_runtime_score_contract")
        if policy and policy["binding"] != binding:
            reasons.append("policy_binding_mismatch")
        prepared = copy.deepcopy(result)
        if not isinstance(prepared, dict):
            prepared = {}
        features = hard_gate_rank(prepared, policy["hard_contract"]) if policy else {
            "risk_score": None, "score_contract_id": SCORE_CONTRACT_ID,
            "risk_score_features": None, "score_is_probability": False}
        prepared.update(features)
        applied = self.store.apply(ident if not reasons else None, prepared, binding)
        applied.update(features)
        applied["profile_qualified"] = key in self.qualified
        applied["runtime_policy_source"] = "trusted_server_configuration"
        if reasons:
            applied["decision"] = "review"
            applied["reason_codes"] = sorted(set(applied["reason_codes"] + reasons))
        return applied


def unsupported_strategy_actions(cards: list[dict], supported: set[str] | None = None) -> list[str]:
    """A stored/vetted directive is not automatically an implemented runtime action."""
    supported = {"reread"} if supported is None else supported
    return sorted({directive["action"] for card in cards for directive in card["directives"]
                   if directive["action"] not in supported})
