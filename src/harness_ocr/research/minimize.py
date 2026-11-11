"""F05: budgeted delta debugging of observed graphs, never of source images."""
from __future__ import annotations

import math
import time
from typing import Callable

from .common import TrialJournal, clone_json, digest, finite_number, identifier

_REFERENCE_KEYS = {"evidence_ids", "source_ids", "source_region_ids", "depends_on",
                   "parent_ids", "required_ids"}
_PROVENANCE_KINDS = {"source", "artifact", "provenance"}
_PROTECTED_EDGE_TYPES = {"evidence", "source", "provenance", "supports"}


def _validate_graph(graph: dict) -> tuple[dict, set[str]]:
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges", []), list):
        raise ValueError("graph must have nodes:list and optional edges:list")
    nodes, ids = [], set()
    for node in graph["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("Each graph node must be an object")
        identity = identifier(node.get("id"), "node.id")
        if identity in ids:
            raise ValueError(f"Duplicate node.id {identity}")
        ids.add(identity)
        nodes.append(node)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("source") not in ids or edge.get("target") not in ids:
            raise ValueError("Every edge source/target must identify a graph node")
    return clone_json(graph), ids


def _references(value, ids: set[str]) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _REFERENCE_KEYS:
                values = child if isinstance(child, list) else [child]
                found.update(x for x in values if isinstance(x, str) and x in ids)
            found.update(_references(child, ids))
    elif isinstance(value, list):
        for child in value:
            found.update(_references(child, ids))
    return found


def _failure(result) -> tuple[bool, str | None]:
    if type(result) is bool:
        return result, "predicate_failure" if result else None
    if not isinstance(result, dict) or type(result.get("failed")) is not bool:
        raise ValueError("Predicate must return bool or {failed:bool, failure_id:str}")
    if not result["failed"]:
        return False, None
    return True, identifier(result.get("failure_id"), "predicate.failure_id")


def minimize_graph(graph: dict, predicate: Callable[[dict], bool | dict], *,
                   required_ids=(), max_calls: int = 100,
                   time_budget_seconds: float | None = None, trial_path=None) -> dict:
    """Find a locally smaller graph preserving the *same* witnessed failure.

    ``graph`` has ``nodes:[{id,...}]``, ``edges:[{source,target,type?,...}]`` and
    optional ``task_contract`` plus any provenance metadata. Metadata, surviving
    node contents, source images and task contract are never edited. Protected
    nodes: explicit ``required_ids``, ``required=True`` and source/artifact/
    provenance kinds. Evidence references of surviving nodes remain present.
    ``evidence/source/provenance`` edges point claim→source; ``supports`` points
    source→claim. Such edges are never stripped between retained endpoints.

    The predicate should return ``{failed:bool,failure_id:str}`` to distinguish
    error classes. A boolean predicate must already mean this particular defect.
    Predicate exceptions are recorded and never accepted as reproducing a bug.

    Time budget is checked *between* calls; this function cannot interrupt an
    arbitrary Python predicate. Wrap expensive callbacks in a process watchdog.
    No minimum or reproducibility claim is made for unobserved/stochastic cases.
    """
    original, ids = _validate_graph(graph)
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    if time_budget_seconds is not None:
        time_budget_seconds = finite_number(time_budget_seconds, "time_budget_seconds")
        if time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
    required = set(required_ids)
    if not required <= ids:
        raise ValueError("required_ids contains unknown nodes")
    required.update(n["id"] for n in original["nodes"] if n.get("required") is True or n.get("kind") in _PROVENANCE_KINDS)
    for edge in original.get("edges", []):
        if edge.get("required") is True:
            required.update((edge["source"], edge["target"]))
    node_by_id = {n["id"]: n for n in original["nodes"]}
    indexed_edges = list(enumerate(original.get("edges", [])))
    protected_edges = {i for i, e in indexed_edges if e.get("required") is True or e.get("type") in _PROTECTED_EDGE_TYPES}

    def materialize(keep: set[str], edge_keep: set[int]) -> dict:
        keep = set(keep) | required
        while True:
            previous = set(keep)
            for identity in list(keep):
                keep.update(_references(node_by_id[identity], ids))
            for i, edge in indexed_edges:
                kind = edge.get("type")
                if edge.get("required") is True:
                    keep.update((edge["source"], edge["target"]))
                elif kind == "supports" and edge["target"] in keep:
                    keep.add(edge["source"])
                elif kind in {"evidence", "source", "provenance"} and edge["source"] in keep:
                    keep.add(edge["target"])
            if keep == previous:
                break
        result = clone_json(original)
        result["nodes"] = [clone_json(n) for n in original["nodes"] if n["id"] in keep]
        if "edges" in original:
            result["edges"] = [clone_json(e) for i, e in indexed_edges
                               if e["source"] in keep and e["target"] in keep
                               and (i in edge_keep or i in protected_edges)]
        return result

    current = original
    keep = set(ids)
    edge_keep = {i for i, _ in indexed_edges}
    # CPython 3.12 Windows monotonic can be GetTickCount64 (~15.6 ms). Use the
    # monotonic high-resolution performance clock for small declared budgets.
    start = time.perf_counter()
    calls, history, seen, errors = 0, [], set(), 0
    target = None
    stop = None

    def allowed():
        nonlocal stop
        if calls >= max_calls:
            stop = "call_budget_exhausted"
            return False
        if time_budget_seconds is not None and time.perf_counter() - start >= time_budget_seconds:
            stop = "time_budget_exhausted"
            return False
        return True

    manifest = {"protocol": "graph-minimizer/1", "source_graph_sha256": digest(original),
                "required_ids": sorted(required), "max_calls": max_calls,
                "time_budget_seconds": time_budget_seconds,
                "time_guard_scope": "between predicate calls; no in-process hard interruption"}
    with TrialJournal(trial_path, manifest) as journal:
        def call(candidate: dict, phase: str):
            nonlocal calls, errors
            calls += 1
            sha = digest(candidate)
            seen.add(sha)
            row = {"record_type": "trial", "call": calls, "phase": phase,
                   "graph_sha256": sha, "nodes": len(candidate["nodes"]),
                   "edges": len(candidate.get("edges", []))}
            try:
                failed, identity = _failure(predicate(clone_json(candidate)))
                row.update(status="ok", failed=failed, failure_id=identity)
            except Exception as exc:
                failed, identity = False, None
                errors += 1
                row.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            history.append(row)
            journal.write(row)
            return failed, identity, row["status"]

        failed, target, baseline_status = call(original, "baseline")
        if baseline_status == "error":
            stop = "baseline_predicate_error"
        elif not failed:
            stop = "baseline_does_not_fail"
        else:
            while stop is None:
                changed_in_pass = False
                for kind in ("nodes", "edges"):
                    candidates = ([n["id"] for n in current["nodes"] if n["id"] not in required]
                                  if kind == "nodes" else
                                  [i for i, e in indexed_edges if i in edge_keep and i not in protected_edges
                                   and e in current.get("edges", [])])
                    granularity = 2
                    while candidates and stop is None:
                        width = max(1, math.ceil(len(candidates) / granularity))
                        accepted = False
                        for offset in range(0, len(candidates), width):
                            removed = set(candidates[offset:offset + width])
                            proposed_nodes = keep - removed if kind == "nodes" else keep
                            proposed_edges = edge_keep - removed if kind == "edges" else edge_keep
                            candidate = materialize(proposed_nodes, proposed_edges)
                            sha = digest(candidate)
                            if sha == digest(current) or sha in seen:
                                continue
                            if not allowed():
                                break
                            reproduced, failure_id, _ = call(candidate, kind)
                            if reproduced and failure_id == target:
                                current = candidate
                                keep = {n["id"] for n in current["nodes"]}
                                edge_keep = set(proposed_edges)
                                changed_in_pass = accepted = True
                                candidates = ([n["id"] for n in current["nodes"] if n["id"] not in required]
                                              if kind == "nodes" else [i for i in candidates if i in edge_keep])
                                granularity = max(2, granularity - 1)
                                break
                        if stop is not None:
                            break
                        if not accepted:
                            if granularity >= len(candidates):
                                break
                            granularity = min(len(candidates), granularity * 2)
                if not changed_in_pass and stop is None:
                    stop = "search_complete_with_predicate_errors" if errors else "locally_minimal"

    assert current.get("task_contract") == original.get("task_contract")
    assert required <= {n["id"] for n in current["nodes"]}
    return {"protocol": "graph-minimizer/1", "graph": current,
            "source_graph_sha256": digest(original), "minimized_graph_sha256": digest(current),
            "target_failure_id": target, "changed": digest(current) != digest(original),
            "calls": calls, "predicate_errors": errors, "stop_reason": stop,
            "elapsed_seconds": time.perf_counter() - start, "history": history,
            "required_ids": sorted(required), "task_contract_preserved": True,
            "source_pixels_modified": False,
            "interpretation": "A diagnostic reduction under the supplied predicate, not production accuracy or a globally minimal causal explanation."}
