"""Bounded literal line graph and conservative label/value association.

No line bbox is inferred from a whole-page text string. Association is either
explicit within one line, consecutive OCR text lines, or a unique geometrically
bounded neighbor from the SAME source/page. It remains an extraction hypothesis,
not proof of visual correctness; changed layouts require independent qualification.
"""
from __future__ import annotations
import math
import re
from .util import digest
from .validators import pattern_spans,PATTERNS

VERSION='literal-line-graph-v1'

def lines(observations,max_lines=3000):
    output=[]
    for obs in observations:
        if not obs.get('completed') or obs.get('active') is False:continue
        parts=obs['text'].splitlines(keepends=True);nonempty=[x for x in parts if x.strip()]
        box=None;region_box=None
        if len(nonempty)==1:
            value=obs.get('bbox')
            if value is None and obs.get('quad'):
                quad=obs['quad'];value=[min(p[0] for p in quad),min(p[1] for p in quad),max(p[0] for p in quad),max(p[1] for p in quad)]
            if (isinstance(value,(list,tuple)) and len(value)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value)
                    and value[2]>value[0] and value[3]>value[1]):
                region_box=list(value)
                if obs.get('quad') is not None or obs.get('geometry_granularity')=='line' or obs.get('evidence_kind')=='native_pdf_text':box=list(value)
        offset=0
        for index,part in enumerate(parts):
            clean=part.rstrip('\r\n');start=len(clean)-len(clean.lstrip());end=len(clean.rstrip())
            if start<end:
                output.append({'id':obs['id']+':line:'+str(index),'observation_id':obs['id'],
                     'text':clean[start:end],'start':offset+start,'end':offset+end,'line_index':index,
                     'page':obs.get('page'),'page_id':obs.get('page_id'),
                     'source_sha256':obs.get('source_sha256'),'bbox':box,'source_region_bbox':region_box,
                     'coordinates_exact_for_line':box is not None})
                if len(output)>max_lines:raise ValueError('line_graph_limit_no_silent_truncation')
            offset+=len(part)
    return output

def label_hits(text,labels):
    hits=[]
    for label,owners in labels.items():
        for match in re.finditer(r'(?<![\w])'+re.escape(label)+r'(?![\w])',text,re.IGNORECASE):
            hits.append((match.start(),match.end(),owners))
    kept=[]
    for hit in sorted(hits,key=lambda h:(-(h[1]-h[0]),h[0])):
        if not any(hit[0]<old[1] and old[0]<hit[1] for old in kept):kept.append(hit)
    return sorted(kept,key=lambda h:h[0])

def label_only(node,labels,field):
    hits=label_hits(node['text'],labels)
    if len(hits)!=1 or hits[0][2]!={field}:return None
    start,end,_=hits[0]
    if node['text'][:start].strip(' \t№:—-') or node['text'][end:].strip(' \t№:—-'):return None
    return {'observation_id':node['observation_id'],'start':node['start']+start,'end':node['start']+end}

def neighbor_values(label,nodes,labels):
    # A whole-page OCR has text order but no honest line coordinates.
    after=sorted([n for n in nodes if n['observation_id']==label['observation_id'] and 1<=n['line_index']-label['line_index']<=2],key=lambda n:n['line_index'])
    if after and not label_hits(after[0]['text'],labels) and ':' not in after[0]['text']:
        return [(after[0],{'kind':'consecutive_text_lines','label_line_id':label['id'],
            'value_line_id':after[0]['id'],'fabricated_line_geometry':False})]
    if label['bbox'] is None or not label.get('source_sha256') or label.get('page_id') is None:return []
    x0,y0,x1,y1=label['bbox'];lh=y1-y0;possible=[]
    for n in nodes:
        if (n['id']==label['id'] or n['bbox'] is None or n.get('source_sha256')!=label['source_sha256']
                or n.get('page_id')!=label['page_id'] or label_hits(n['text'],labels) or ':' in n['text']):continue
        a,b,c,d=n['bbox'];nh=d-b;scale=max(lh,nh)
        y_overlap=max(0,min(y1,d)-max(y0,b));x_overlap=max(0,min(x1,c)-max(x0,a))
        right=(0<=a-x1<=12*scale and y_overlap>=.6*min(lh,nh))
        below=(-.2*scale<=b-y1<=2*scale and (x_overlap>=.5*min(x1-x0,c-a) or abs(a-x0)<=1.5*scale))
        if not (right or below):continue
        # A second labelled cell between the two anchors blocks association.
        cx=(a+c)/2;cy=(b+d)/2;blocked=False
        for other in nodes:
            if other['id'] in {label['id'],n['id']} or other['bbox'] is None:continue
            if other.get('source_sha256')!=label['source_sha256'] or other.get('page_id')!=label['page_id']:continue
            if not label_hits(other['text'],labels):continue
            q=other['bbox'];qx=(q[0]+q[2])/2;qy=(q[1]+q[3])/2
            if right and x1<=qx<a and max(y0,b)<=qy<=min(y1,d):blocked=True
            if below and y1<=qy<b and min(x0,a)<=qx<=max(x1,c):blocked=True
        if not blocked:
            possible.append((n,{'kind':'unique_spatial_neighbor','relation':'right' if right else 'below',
                 'label_line_id':label['id'],'value_line_id':n['id'],'label_bbox':label['bbox'],
                 'value_bbox':n['bbox'],'source_sha256':label['source_sha256'],'page_id':label['page_id'],
                 'fabricated_line_geometry':False}))
    # Multiple plausible cells are ambiguous; never break ties by a lucky value.
    return possible if len(possible)==1 else []

def add_adjacent(observations,contract,labels,candidates,max_candidates):
    nodes=lines(observations)
    for field,spec in contract['field_definitions'].items():
        for label in nodes:
            role_span=label_only(label,labels,field)
            if role_span is None:continue
            for value,proof in neighbor_values(label,nodes,labels):
                spans=pattern_spans(value['text'],spec['value_type'])
                if len(spans)>1:continue
                kind=spec['value_type'].lower()
                if not spans and (kind.startswith('digits_') or kind in PATTERNS) and kind!='date_or_literal':continue
                if spans:
                    a,b=spans[0];prefix=value['text'][:a].strip();suffix=value['text'][b:].strip()
                    suffixes={'','руб.','руб','₽','RUB','RUR'} if spec['value_type'] in {'money_decimal','money','amount'} else {''}
                    if prefix not in {'','№'} or suffix not in suffixes:continue
                if not spans:spans=[(0,len(value['text']))]
                for start,end in spans:
                    raw=value['text'][start:end]
                    if not raw:continue
                    # A name consisting of one component is not a full name.
                    if spec['value_type']=='person_name' and len(raw.split())<2:continue
                    span={'observation_id':value['observation_id'],'start':value['start']+start,'end':value['start']+end}
                    cid='c_'+digest([field,span,role_span,proof])[:24]
                    candidates[cid]={'id':cid,'field_key':field,'raw_value':raw,
                        'evidence_ids':[value['observation_id']],'spans':[span],
                        'party_role':spec['role'],'role_evidence_ids':[label['observation_id']],
                        'role_spans':[role_span],'role_verified':True,'association_proof':proof,
                        'candidate_generator':VERSION,'requires_layout_qualification':True}
                    if len(candidates)>max_candidates:raise ValueError('candidate_limit_no_silent_truncation')
    add_passport_name_components(nodes,contract,labels,candidates,max_candidates)

def add_passport_name_components(nodes,contract,labels,candidates,max_candidates):
    field='holder.full_name';spec=contract['field_definitions'].get(field)
    if not spec or contract.get('profile_id')!='rf_passport_main_v1':return
    component_labels={'surname':'фамилия','given':'имя','patronymic':'отчество'}
    component_map={v:{'component:'+k} for k,v in component_labels.items()}
    all_labels={**labels,**component_map};groups={}
    for n in nodes:
        # Separate whole-page captures from coordinate-accurate line captures.
        group=('spatial',n.get('source_sha256'),n.get('page_id')) if n['bbox'] is not None else ('text',n['observation_id'])
        groups.setdefault(group,[]).append(n)
    for group,items in groups.items():
        found={};seen_labels=set();ambiguous=False
        for kind,label in component_labels.items():
            field_id='component:'+kind
            for node in items:
                hits=label_hits(node['text'],all_labels)
                matching=[h for h in hits if h[2]=={field_id}]
                if not matching:continue
                seen_labels.add(kind)
                for start,end,_ in matching:
                    role={'observation_id':node['observation_id'],'start':node['start']+start,'end':node['start']+end}
                    tail=end
                    while tail<len(node['text']) and node['text'][tail] in ' :\t—-':tail+=1
                    tail_text=node['text'][tail:]
                    options=[]
                    if tail_text and len(hits)==1:options=[(node,tail,len(node['text']))]
                    elif not tail_text:
                        for value,proof in neighbor_values(node,items,all_labels):options.append((value,0,len(value['text'])))
                    for value,a,b in options:
                        raw=value['text'][a:b]
                        if not raw or len(raw)>128 or any(not (ch.isalpha() or ch in "-'’") for ch in raw):continue
                        part={'raw':raw,'span':{'observation_id':value['observation_id'],'start':value['start']+a,'end':value['start']+b},'role':role}
                        found.setdefault(kind,[]).append(part)
        if not {'surname','given'}<=set(found):continue
        if 'patronymic' in seen_labels and 'patronymic' not in found:continue
        if any(len(v)!=1 for v in found.values()):continue
        ordered=[found[k][0] for k in ['surname','given','patronymic'] if k in found]
        raw=' '.join(p['raw'] for p in ordered);spans=[]
        for index,part in enumerate(ordered):spans.append({**part['span'],'separator_after':' ' if index<len(ordered)-1 else ''})
        cid='c_'+digest([field,spans,'passport-components-v1'])[:24]
        candidates[cid]={'id':cid,'field_key':field,'raw_value':raw,'spans':spans,
            'evidence_ids':sorted({p['span']['observation_id'] for p in ordered}),
            'role_evidence_ids':sorted({p['role']['observation_id'] for p in ordered}),
            'role_spans':[p['role'] for p in ordered],'party_role':spec['role'],'role_verified':True,
            'candidate_generator':VERSION,'assembly':'rf-passport-explicit-name-components-v1',
            'association_proof':{'kind':'explicit_labeled_components','group':list(group),
                                 'added_characters':'whitespace_only'},'requires_layout_qualification':True}
        if len(candidates)>max_candidates:raise ValueError('candidate_limit_no_silent_truncation')
