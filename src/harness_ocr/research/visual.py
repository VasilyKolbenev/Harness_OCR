"""F04 blind VLM exam: immutable pixels -> optical capture -> exact comparison.

Ground truth is held by ``run_blind_exam`` only. Good and corrupted proposals
share an optical capture keyed by verified source/raster/crop pixels, model and
preprocessing; proposals and scoring labels are absent from that cache key and
from every first-stage model prompt. No text-only fallback is supported.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import math

from ..inference.vision import (PREPROCESS, VERSION, prepare_visual_regions,
                               compare_visual_capture)
from ..util import digest
from .blind import run_blind_exam
from .integration import _backend_manifest


class QwenVisualVerifier:
    def __init__(self, backend, *, qwen='qwen-0.8b', timeout=120, trace=None, max_cache_entries=64):
        if qwen not in {'qwen-0.8b','qwen-4b'}:raise ValueError('unsupported_verifier_model')
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('finite_positive_timeout_required')
        if type(max_cache_entries) is not int or not 1<=max_cache_entries<=256:
            raise ValueError('bounded_visual_capture_cache_required')
        self.backend=backend;self.qwen=qwen;self.timeout=timeout;self.trace=trace
        self.max_cache_entries=max_cache_entries;self.cache=OrderedDict();self.manifest=_backend_manifest(backend,qwen)
        self.model_fingerprint=self.manifest['pinned_model_sha256']

    def __call__(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'trial_id','modality','input'} or payload['modality']!='vision':
            raise ValueError('this_adapter_supports_actual_vision_modality_only')
        material=payload['input']
        if not isinstance(material,dict) or set(material)!={'regions','candidate'}:
            raise ValueError('visual_requires_regions_and_candidate_only')
        candidate=material['candidate']
        if (not isinstance(candidate,dict) or set(candidate)!={'region_id','raw_value'}
            or not isinstance(candidate['raw_value'],str) or not candidate['raw_value']):
            raise ValueError('visual_candidate_requires_region_and_literal')
        regions=material['regions']
        # Validate even on cache hits: changed/missing files must never borrow an
        # old success merely because a caller reused a descriptor's hash string.
        _,provenance=prepare_visual_regions(regions)
        if candidate['region_id'] not in {p['region_id'] for p in provenance}:
            raise ValueError('visual_candidate_region_not_allowlisted')
        key=digest({'provenance':provenance,'model':self.model_fingerprint,'preprocess':PREPROCESS,'version':VERSION})
        hit=key in self.cache
        if hit:
            cached=self.cache.pop(key);self.cache[key]=cached;output=copy.deepcopy(cached)
        else:
            # Candidate values and private labels never cross the process API.
            output=self.backend.call('visual_read',{'name':self.qwen,'regions':copy.deepcopy(regions)},timeout=self.timeout)
            if (output.get('modality')!='vision' or output.get('purpose')!='blind_optical_capture'
                or output.get('completed') is not True or output.get('stop_reason')!='eos'
                or output.get('proposed_value_supplied') is not False
                or output.get('ground_truth_supplied') is not False):
                raise RuntimeError('visual_backend_not_completed_blind_vlm_capture')
            if output.get('model_bundle_sha256')!=self.model_fingerprint:
                raise ValueError('visual_capture_model_fingerprint_mismatch')
            captures=output.get('captures')
            if not isinstance(captures,list) or len(captures)!=len(provenance):
                raise ValueError('visual_capture_count_mismatch')
            by_id={c.get('provenance',{}).get('region_id'):c for c in captures}
            if set(by_id)!={p['region_id'] for p in provenance}:
                raise ValueError('visual_capture_region_scope_mismatch')
            for p in provenance:
                capture=by_id[p['region_id']]
                if capture.get('provenance')!=p or capture.get('model_id')!=self.qwen:
                    raise ValueError('visual_capture_provenance_mismatch')
                # Validate the sealed capture before caching it.
                compare_visual_capture(capture,candidate['raw_value'])
            # Cache only immutable observations and diagnostics, never KV state,
            # parser/enforcer instances, tensors or an already-compared verdict.
            self.cache[key]={'captures':copy.deepcopy(captures)}
            while len(self.cache)>self.max_cache_entries:self.cache.popitem(last=False)
        capture=next(c for c in output['captures'] if c['provenance']['region_id']==candidate['region_id'])
        answer={**compare_visual_capture(capture,candidate['raw_value']),
            'capture':copy.deepcopy(capture),'optical_cache_hit':hit,'optical_cache_key_sha256':key,
            'new_model_calls':0 if hit else 1,'modality':'vision','purpose':'blind_optical_capture_then_exact_compare',
            'same_model_text_and_vision_are_independent':False}
        if self.trace:
            self.trace.record('research_blind_visual_verifier',{'trial_id':payload['trial_id'],
                'regions':regions,'candidate':candidate,'optical_cache_key_sha256':key},answer,
                {**self.manifest,'adapter_version':VERSION,'modality':'vision'},trial_id=payload['trial_id'],
                stage_kind='model' if not hit else 'pure',status='completed')
        return answer


def visual_case_from_regions(regions, *, case_id, original_id, region_id, proposed_raw,
                             candidate_correct, observable=True, origin='controlled'):
    """The caller chooses private scoring labels; they are never model inputs."""
    return {'id':case_id,'original_id':original_id,'origin':origin,
        'input':{'regions':copy.deepcopy(regions),'candidate':{'region_id':region_id,'raw_value':proposed_raw}},
        'gold':{'candidate_correct':candidate_correct,'observable':{'vision':observable}}}


def run_visual_blind_exam(cases,backend,*,qwen='qwen-0.8b',seed=0,trial_path=None,trace=None,timeout=120):
    verifier=QwenVisualVerifier(backend,qwen=qwen,timeout=timeout,trace=trace)
    return run_blind_exam(cases,verifier,seed=seed,modality='vision',trial_path=trial_path,
        verifier_metadata={**verifier.manifest,'adapter':'real-supervisor-blind-visual-verifier-v1',
            'modality':'vision','purpose':'blind_optical_capture_then_exact_compare',
            'optical_sources':'verified_original_raster_and_roi_pixels','quality_qualified':False,
            'role_verification':'not_inferred_from_literal_match','automatic_acceptance':False,
            'same_model_text_and_vision_are_independent':False})
