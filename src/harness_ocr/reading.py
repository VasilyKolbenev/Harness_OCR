"""Bounded evidence-driven reading: deterministic facts, targeted model work.

Models may add optical observations and choose existing IDs. They never supply
trusted values, erase contradictory readings, or repair business-invalid data.
"""
from __future__ import annotations

import json
from pathlib import Path

from .grounding import (build_candidates, candidate_is_eligible, canonical_candidate_key,
    field_region_hints, rank_reread_regions, qwen_prompt, resolve_response)
from .typed_fields import value_identity
from .util import digest
from .inference.supervisor import ModelCanceled, ModelDeadline
from .inference.region_tasks import (context_box, materialize_region, page_from_visual_descriptor,
                                    same_region_geometry)

VERSION = 'evidence-driven-reading-v2-upright'


def unique_source_proposals(candidates, contract):
    """One canonical source for a unique *observed* value; not a majority vote."""
    proposals = []
    for field in contract['requested_fields']:
        group = [c for c in candidates.values()
                 if c['field_key'] == field and candidate_is_eligible(c)]
        if len({digest(value_identity(c['raw_value'])) for c in group}) != 1:
            continue
        chosen = min(group, key=canonical_candidate_key)
        proposals.append({'field_key':field, 'candidate_ids':[chosen['id']],
                          'party_role':chosen['party_role']})
    return proposals


class ReadingController:
    def __init__(self, engine, *, document_id, source_path, source_sha256, contract,
                 snapshot, observations, regions, pages, visual_regions, visual_regions_by_id,
                 call, record, checkpoint, qwen, mode, cards, reread_limit, max_rounds):
        self.engine, self.settings = engine, engine.settings
        self.document_id, self.source_path, self.source_sha = document_id, source_path, source_sha256
        self.contract, self.snapshot = contract, snapshot
        self.observations, self.regions = observations, regions
        self.pages = dict(pages)
        self.visual_regions, self.visual_regions_by_id = visual_regions, visual_regions_by_id
        for descriptor in visual_regions.values():
            self.pages.setdefault(descriptor['page_id'], page_from_visual_descriptor(descriptor))
        self.call, self.record, self.checkpoint = call, record, checkpoint
        self.qwen, self.mode, self.cards = qwen, mode, cards
        self.reread_limit, self.max_rounds = reread_limit, max_rounds
        self.used = set(); self.reads = 0; self.rescued = set(); self.whole_pages = set()
        self.generated_context_regions = set()
        self.read_crops = set()
        self.selector_fields = set(); self.errors = []; self.review_reasons = []
        self.complete = True; self.candidates = {}; self.decisions = []

    def revise(self, *, observations=(), claims=(), reason):
        self.snapshot = self.engine.documents.revise(self.document_id, observations=observations,
            claims=claims, expected_revision=self.snapshot['revision'], reason=reason)

    def refresh(self, *, commit=False):
        self.candidates = build_candidates(self.observations, self.contract)
        self.record('candidates', {'observations':self.observations, 'contract':self.contract}, self.candidates)
        proposals = unique_source_proposals(self.candidates,self.contract)
        if not commit:
            resolved_fields = {p['field_key'] for p in proposals}
            return [field for field in self.contract['requested_fields'] if field not in resolved_fields]
        # Candidate hypotheses remain revisable while more pixels are read. Do
        # not commit a short word as a field, then manufacture a contradiction
        # when new geometry proves that the field continues to the right.
        # All original observations and every candidate snapshot stay in trace.
        conflicts = self.engine._candidate_conflict_claims(self.candidates, self.snapshot)
        if conflicts:
            self.revise(claims=conflicts, reason='all_observed_conflicts_preserved')
        response = json.dumps({'proposals':proposals,
                               'reread_requests':[]})
        resolved = resolve_response(response,self.candidates,self.regions,self.contract)
        self.record('resolve', {'response':response,'candidates':self.candidates,
            'regions':self.regions,'contract':self.contract}, resolved)
        self.record('deterministic_selection', {'candidates':self.candidates,'contract':self.contract},
                    {'response':response,'resolved':resolved,'authority':'existing_literal_and_role_guards'})
        claims = []
        for claim in resolved['claims']:
            claim = {**claim,'id':'auto_'+digest([claim['field_key'],claim['id'],claim['raw_value']])[:24], 'origin':'rule'}
            if claim['id'] not in self.snapshot['claims']:
                claims.append(claim)
        if claims:
            self.revise(claims=claims, reason='unique_source_grounded_values')
        return [field for field in self.contract['requested_fields']
                if self.snapshot['fields'].get(field,{}).get('raw_value') is None
                or self.snapshot['fields'].get(field,{}).get('needs_review')]

    def register(self, observations, regions=(), *, page=None, reason):
        # Do not mutate old observations: model output is append-only evidence.
        if observations:
            self.revise(observations=observations, reason=reason)
            self.observations.extend(observations)
        self.regions.extend(regions)
        if not self.settings.visual_verifier or page is None:
            return
        from .inference.vision import make_visual_region
        for region in regions:
            descriptor = make_visual_region(self.source_path,page,region,source_sha256=self.source_sha)
            self.visual_regions_by_id[region['id']] = descriptor
            for obs in observations:
                if same_region_geometry(obs, region):
                    self.visual_regions[obs['id']] = descriptor

    def action(self, name, data):
        decision = {'policy':VERSION,'action':name,**data,'value_or_role_authority':False}
        self.decisions.append(decision)
        self.record('reading_decision', {}, decision)

    def try_call(self, op, args, *, task):
        try:
            return self.call(op,args)
        except ModelCanceled:
            raise
        except ModelDeadline as exc:
            self.checkpoint()  # Never convert document/global cancellation to a field retry.
            self.errors.append('reading_task_deadline:'+task+':'+str(exc))
        except Exception as exc:
            self.errors.append('reading_task_failed:'+task+':'+type(exc).__name__+':'+str(exc)[:160])
        self.complete = False
        self.review_reasons.append('reading_task_unavailable:'+task)
        return None

    def local_read(self, hint):
        by_id = {r['id']:r for r in self.regions}
        original = by_id.get(hint['region_id'])
        if original is None:
            return False
        self.used.add(original['id']); self.reads += 1
        page = self.pages.get(original.get('page_id'))
        region = original
        if page is not None:
            base = {**original,'bbox':hint['bbox']}
            # Hint unions are real box unions; do not let an old quad override them.
            base.pop('quad',None)
            box = context_box(base)
            if box is None:
                # A broad original region is not a free field ROI. Preserve
                # the separate, explicit whole-page fallback budget.
                self.review_reasons.append('context_crop_too_broad:'+hint['field_key'])
                self.action('targeted_read_deferred',{'field_key':hint['field_key'],
                    'region_id':original['id'],'reason':'requires_explicit_page_budget'})
                return True
            if box is not None:
                try:
                    region = materialize_region(page,box,self.engine.root/'reading_regions'/self.document_id,
                        source_sha256=self.source_sha,purpose='label_context',
                        max_page_pixels=self.settings.max_page_pixels)
                except (ValueError,OSError) as exc:
                    self.review_reasons.append('context_crop_unavailable:'+hint['field_key'])
                    self.record('context_crop',hint,{'error':type(exc).__name__+':'+str(exc)[:160]},status='incomplete')
                    return True
        if region['id'] != original['id']:
            # A context reading is append-only evidence, not a fresh detector
            # anchor that can recursively spend the next local-read slot.
            self.generated_context_regions.add(region['id'])
            self.used.add(region['id'])
        self.action('targeted_xiaomi_read', {'field_key':hint['field_key'],
            'region_id':region['id'],'basis_region_id':original['id'],'reason':hint['reason_code']})
        crop_key = digest([self.source_sha,region.get('page_id'),region.get('quad'),region.get('bbox')])
        if crop_key in self.read_crops:
            self.action('identical_reread_avoided',{'region_id':region['id']})
            return True
        self.read_crops.add(crop_key)
        ocr_args={'path':region['path']}
        if region.get('crop_sha256'):
            ocr_args['expected_image_sha256']=region['crop_sha256']
        out = self.try_call('ocr',ocr_args,task='roi:'+hint['field_key'])
        if out is None:
            return True
        geometry = 'quad' if region.get('quad') is not None else 'bbox'
        obs = {'id':'read_'+digest([self.document_id,region['id'],self.reads])[:24],
            'source_sha256':self.source_sha,'text':out['text'],'evidence_kind':'image_region',
            'page':region['page'],'page_id':region['page_id'],'source_size':region['source_size'],
            geometry:region[geometry],'completed':out['completed'],
            'capture_id':'xiaomi_roi_'+digest([self.document_id,region['id'],self.reads])[:24],
            'model_id':'xiaomi','reading_stream_id':'xiaomi:'+region['id'],
            'raster_transform_proof':region.get('transform',{}),'reading_purpose':'field_label_context'}
        self.complete = self.complete and out['completed']
        self.register([obs],[region] if region['id'] != original['id'] else [],page=page,reason='targeted_source_read')
        if self.settings.visual_verifier:
            descriptor = self.visual_regions_by_id.get(region['id'])
            if descriptor:
                self.visual_regions[obs['id']] = descriptor
        return True

    def rescue(self, unresolved):
        if not self.settings.rescue_enabled or len(self.rescued) >= self.settings.max_detector_rescues:
            return False
        hints = field_region_hints(self.candidates,self.regions,self.contract,self.observations,
            used_region_ids=(),max_regions=20,field_keys=unresolved)
        preferred = [h['page_id'] for h in hints]
        page_ids = list(dict.fromkeys([*preferred,*self.pages]))
        page_id = next((p for p in page_ids if p not in self.rescued),None)
        if page_id is None:
            return False
        self.rescued.add(page_id); page = self.pages[page_id]
        self.action('detector_rescue',{'page_id':page_id,'detector':self.settings.rescue_text_detector,
            'unresolved_fields':unresolved,'trigger':'missing_or_incomplete_field_evidence'})
        out = self.try_call('cheap_rescue',{'page':page,'source_sha256':self.source_sha,
            'output_dir':str(self.engine.root/'rescue_regions'/self.document_id/page_id)},task='detector:'+page_id)
        if out is not None:
            self.register(out['observations'],out['regions'],page=page,reason='complementary_detector_evidence')
            if not out.get('completed',True):
                # Preserve the uncertainty, even if a useful localized value was read.
                self.review_reasons.append('detector_rescue_incomplete:'+page_id)
                self.complete = False
        return True

    def whole_page(self, unresolved):
        if len(self.whole_pages) >= self.settings.max_whole_page_reads:
            return False
        page_id = next((p for p in self.pages if p not in self.whole_pages),None)
        if page_id is None:
            return False
        self.whole_pages.add(page_id); page = self.pages[page_id]
        self.action('whole_page_fallback',{'page_id':page_id,'unresolved_fields':unresolved,
            'reason':'targeted_evidence_exhausted','bounded_by':'max_whole_page_reads'})
        out = self.try_call('ocr',{'path':page['path'],'expected_image_sha256':page['raster_sha256']},task='page:'+page_id)
        if out is None:
            return True
        obs = {'id':page_id+'_xiaomi_fallback','source_sha256':self.source_sha,
            'text':out['text'],'evidence_kind':'image_region','page':page['page'],'page_id':page_id,
            'source_size':page['source_size'],'bbox':[0,0,*page['source_size']],
            'completed':out['completed'],'capture_id':'xiaomi_page_'+digest([self.document_id,page_id])[:24],
            'model_id':'xiaomi','reading_stream_id':'xiaomi:'+page_id+':whole',
            'raster_transform_proof':page['transform'],'reading_purpose':'explicit_whole_page_fallback'}
        self.complete = self.complete and out['completed']
        self.register([obs],page=page,reason='bounded_whole_page_fallback')
        if self.settings.visual_verifier:
            from .inference.vision import make_visual_region
            self.visual_regions[obs['id']] = make_visual_region(self.source_path,page,source_sha256=self.source_sha)
        return True

    def selector_hint(self, unresolved):
        """Only bounded ambiguous-field routing; cannot overrule source conflicts."""
        if self.mode != 'hybrid' or len(self.selector_fields) >= 2 or self.reads >= self.reread_limit:
            return None
        field = next((f for f in unresolved if f not in self.selector_fields and len({
            digest(value_identity(c['raw_value'])) for c in self.candidates.values()
            if c['field_key']==f and candidate_is_eligible(c)}) > 1),None)
        if field is None:
            return None
        self.selector_fields.add(field)
        # UTF-8 byte bound leaves generous space for the fixed chat template;
        # the tokenizer's exact full-template guard remains in force as well.
        byte_budget = self.settings.max_input_tokens-1024
        if byte_budget < 1024:
            self.review_reasons.append('selector_field_exceeds_input_budget:'+field)
            return None
        prompt = None
        for cap in (min(12000,byte_budget), max(1024,byte_budget//2)):
            try:
                candidate_prompt = qwen_prompt(self.candidates,self.regions,self.contract,
                    observations=self.observations,strategies=self.cards,field_keys=[field],
                    max_prompt_characters=cap,max_source_characters=min(2000,cap//3),max_regions=8,
                    remaining_budget={'reread_regions':self.reread_limit-self.reads,'reread_rounds':1,
                                      'max_new_tokens':self.settings.max_new_tokens})
            except ValueError as exc:
                if 'partition' not in str(exc):
                    raise
                continue
            if len(candidate_prompt.encode('utf-8')) <= byte_budget:
                prompt = candidate_prompt;break
        if prompt is None:
            self.review_reasons.append('selector_field_exceeds_input_budget:'+field)
            self.action('selector_field_deferred',{'field_key':field,'reason':'bounded_input_budget'})
            return None
        view = json.loads(prompt)
        ids = {c['id'] for c in view['candidates']}; region_ids = {r['id'] for r in view['regions']}
        self.action('qwen_field_routing',{'field_key':field,'candidate_count':len(ids),
                                        'prompt_utf8_bytes':len(prompt.encode('utf-8'))})
        out = self.try_call('select',{'name':self.qwen,'prompt':prompt},task='selector:'+field)
        if out is None or not out.get('completed'):
            if out is not None:
                self.complete=False; self.review_reasons.append('selector_incomplete:'+field)
            return None
        resolved = resolve_response(out['text'],{k:c for k,c in self.candidates.items() if k in ids},
                                    [r for r in self.regions if r['id'] in region_ids],self.contract)
        self.record('resolve',{'response':out['text'],'candidates':self.candidates,
            'regions':self.regions,'contract':self.contract},resolved)
        # All contradictory claims have already been recorded, so selecting an
        # ID cannot erase the opposition or turn it into an accepted value.
        # Selector results are routing proposals, not early truth maintenance.
        # The final full candidate ledger decides source-supported claims once
        # all bounded reads finish; model preference never removes opposition.
        return next((r for r in resolved['reread_requests'] if r['field_key']==field),None)

    def _run(self):
        unresolved = []
        hard_steps = self.reread_limit + self.settings.max_detector_rescues + self.settings.max_whole_page_reads + 4
        for _ in range(hard_steps):
            self.checkpoint(); unresolved = self.refresh()
            if not unresolved:
                break
            broken_boundary = any(c['field_key'] in unresolved and c.get('value_boundary_complete') is False
                                  for c in self.candidates.values())
            if broken_boundary and self.rescue(unresolved):
                continue
            selected = self.selector_hint(unresolved)
            if self.reads < self.reread_limit and self.reads < max(0,self.max_rounds)*max(1,self.reread_limit):
                hints = field_region_hints(self.candidates,self.regions,self.contract,self.observations,
                    used_region_ids=self.used,max_regions=1,field_keys=unresolved)
                if selected:
                    preferred = [h for h in field_region_hints(self.candidates,self.regions,self.contract,
                        self.observations,used_region_ids=self.used,max_regions=20,
                        field_keys=[selected['field_key']]) if h['region_id'] in selected['region_ids']]
                    if preferred:
                        hints = preferred[:1]
                if not hints:
                    # Structured identifiers need both a printed kind and a
                    # party label. Their bare optical digits are deliberately
                    # not field candidates, but may justify reading context.
                    # This is work routing only: qualification still requires
                    # build_typed_candidates to observe the actual kind/role.
                    from .typed_fields import STRUCTURED_TYPES
                    structured=[f for f in unresolved
                        if self.contract['field_definitions'][f]['value_type'] in STRUCTURED_TYPES]
                    if structured:
                        from .inference.region_tasks import region_box
                        possible=[]
                        for region in self.regions:
                            if region['id'] in self.used:
                                continue
                            related=[o for o in self.observations if o.get('completed')
                                and same_region_geometry(o, region)
                                and any(ch.isdigit() for ch in o.get('text',''))
                                and len(o.get('text',''))<=256]
                            if not related:
                                continue
                            try:
                                box=region_box(region)
                            except ValueError:
                                continue
                            possible.append((-(box[2]-box[0])*(box[3]-box[1]),region['id'],box))
                        if possible:
                            _,rid,box=min(possible,key=lambda x:(x[0],x[1]))
                            hints=[{'field_key':structured[0],'region_id':rid,
                                    'bbox':box,'reason_code':'ambiguous_role'}]
                if not hints:
                    # A detected region with no successful reading is still
                    # optical evidence worth inspecting. Probe it without
                    # assigning a role or preferring a convenient value.
                    from .inference.region_tasks import region_box
                    def unread(region):
                        if region['id'] in self.used:
                            return False
                        return not any(o.get('completed') and o.get('text')
                            and same_region_geometry(o, region) for o in self.observations)
                    empty=[]
                    for region in self.regions:
                        if unread(region):
                            try:
                                box=region_box(region)
                            except ValueError:
                                continue
                            empty.append((-(box[2]-box[0])*(box[3]-box[1]),region['id'],region,box))
                    if empty:
                        _,_,region,box=min(empty,key=lambda x:(x[0],x[1]))
                        hints=[{'field_key':unresolved[0],'region_id':region['id'],
                                'bbox':box,'reason_code':'missing'}]
                if hints and self.local_read(hints[0]):
                    continue
            if self.rescue(unresolved):
                continue
            if self.whole_page(unresolved):
                continue
            break
        unresolved = self.refresh(commit=True)
        return {'snapshot':self.snapshot,'completed':self.complete,'errors':self.errors,
            'review_reasons':sorted(set(self.review_reasons)),
            'unresolved_rereads':[{'field_key':f,'region_ids':[],'reason_code':'missing',
                'unresolved_reason':'evidence_driven_budget_or_missing_anchor'} for f in unresolved],
            'decisions':self.decisions,
            'used_reread_regions':sorted(self.used-self.generated_context_regions),
            'generated_context_regions':sorted(self.generated_context_regions),
            'rescue_pages':sorted(self.rescued),'whole_page_reads':sorted(self.whole_pages)}

    def run(self):
        try:
            return self._run()
        except BaseException as original:
            # Cancellation must preserve already-read, independently supported
            # partial values. This is a pure bounded projection, never another
            # model call or an attempt to repair the interrupted generation.
            try:
                self.refresh(commit=True)
            except BaseException as projection:
                # Best-effort preservation must never replace the original
                # cancellation/deadline with a secondary projection failure.
                original.add_note('partial_projection_failed:'+type(projection).__name__)
            raise
