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

VERSION='literal-line-graph-v2'
TEXT_ASSEMBLY_TYPES={'person_name','organization_name','multispan_text','text','address','string'}

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
                     'source_sha256':obs.get('source_sha256'),'source_size':obs.get('source_size'),
                     'observation_text_characters':len(obs['text']),
                     'reading_stream':obs.get('reading_stream_id',obs.get('model_id')),
                     'bbox':box,'source_region_bbox':region_box,
                     'coordinates_exact_for_line':box is not None})
                if len(output)>max_lines:raise ValueError('line_graph_limit_no_silent_truncation')
            offset+=len(part)
    return output

def label_hits(text,labels):
    hits=[]
    for label,owners in labels.items():
        # OCR may produce tabs/repeated spaces between the printed words. Only
        # whitespace varies: no fuzzy character repair or transliteration.
        pattern=r'(?<![\w])'+r'\s+'.join(re.escape(x) for x in label.split())+r'(?![\w])'
        for match in re.finditer(pattern,text,re.IGNORECASE):
            if _inside_quotes(text,match.start()):continue
            hits.append((match.start(),match.end(),owners))
    kept=[]
    for hit in sorted(hits,key=lambda h:(-(h[1]-h[0]),h[0])):
        if not any(hit[0]<old[1] and old[0]<hit[1] for old in kept):kept.append(hit)
    return sorted(kept,key=lambda h:h[0])


def _inside_quotes(text,position):
    """A role word inside a quoted value is data, not a field heading."""
    stack=[]
    closing={'«':'»','“':'”','„':'“','"':'"'}
    for ch in text[:position]:
        if stack and ch==stack[-1]:stack.pop()
        elif ch in closing:stack.append(closing[ch])
    return bool(stack)


def _union(nodes):
    boxes=[n['bbox'] for n in nodes]
    return [min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes)]


def _row_connected(a,b):
    if (a['bbox'] is None or b['bbox'] is None
        or a.get('source_sha256')!=b.get('source_sha256')
        or not a.get('source_sha256') or a.get('page_id')!=b.get('page_id')
        or a.get('reading_stream') is None or a.get('reading_stream')!=b.get('reading_stream')):return False
    x,y,r,t=a['bbox'];u,v,w,z=b['bbox'];h=t-y;k=z-v
    overlap=max(0,min(t,z)-max(y,v))
    gap=max(0,max(x,u)-min(r,w))
    return overlap>=.65*min(h,k) and gap<=2.5*max(h,k) and .45<=h/k<=2.2


def assembled_value_rows(nodes,labels):
    """Join observed word regions, never manufacture character boxes or text.

    A virtual row is only a routing/association union. Every value span refers
    to its original immutable observation. Substantial/nested disagreements
    block a claim and become an explicit crop hint instead of being dropped.
    """
    values=[n for n in nodes if n['bbox'] is not None and not label_hits(n['text'],labels)]
    remaining={n['id']:n for n in values};rows=[]
    while remaining:
        first=remaining.pop(next(iter(remaining)));group=[first];changed=True
        while changed:
            changed=False
            for ident,n in list(remaining.items()):
                if any(_row_connected(n,old) for old in group):
                    group.append(n);del remaining[ident];changed=True
        if len(group)<2:continue
        group.sort(key=lambda n:(n['bbox'][0],n['bbox'][1],n['id']))
        # Never merge separate passes over the same physical cell. Containment
        # or large overlap may represent a contradictory independent reading.
        ambiguous=False
        for i,a in enumerate(group):
            for b in group[i+1:]:
                overlap=max(0,min(a['bbox'][2],b['bbox'][2])-max(a['bbox'][0],b['bbox'][0]))
                if overlap>.25*min(a['bbox'][2]-a['bbox'][0],b['bbox'][2]-b['bbox'][0]):
                    ambiguous=True
        raw=' '.join(n['text'] for n in group)
        virtual={**first,'id':'joined:'+digest([n['id'] for n in group])[:24],
            'text':raw,'bbox':_union(group),'source_region_bbox':_union(group),
            'coordinates_exact_for_line':False,'assembled_parts':group,
            'assembly_ambiguous':ambiguous,'start':0,'end':len(raw)}
        rows.append(virtual)
    return rows


def _value_spans(value,start,end):
    parts=value.get('assembled_parts')
    if not parts:return [{'observation_id':value['observation_id'],
                          'start':value['start']+start,'end':value['start']+end}]
    if start!=0 or end!=len(value['text']):return []
    return [{'observation_id':n['observation_id'],'start':n['start'],'end':n['end'],
             'separator_after':' ' if i<len(parts)-1 else ''} for i,n in enumerate(parts)]


def _composite_labels(nodes,labels):
    """Recover labels split over OCR regions/lines using their exact words."""
    result=[]
    for first in nodes:
        for label,owners in labels.items():
            words=label.split()
            if len(words)<2 or first['text'].casefold()!=words[0].casefold():continue
            chain=[first]
            for word in words[1:]:
                last=chain[-1];choices=[]
                for n in nodes:
                    if n in chain or n['text'].casefold()!=word.casefold():continue
                    same_text=(n['observation_id']==last['observation_id'] and n['line_index']==last['line_index']+1)
                    right=(_row_connected(last,n) and n['bbox'][0]>=last['bbox'][2]-.2*(last['bbox'][2]-last['bbox'][0]))
                    if same_text or right:choices.append(n)
                if len(choices)!=1:break
                chain.append(choices[0])
            if len(chain)!=len(words):continue
            geometric=all(n['bbox'] is not None for n in chain)
            text=' '.join(n['text'] for n in chain)
            result.append({**first,'id':'label:'+digest([n['id'] for n in chain])[:24],
                'text':text,'bbox':_union(chain) if geometric else None,
                'line_index':chain[-1]['line_index'],'start':0,'end':len(text),
                'label_parts':chain,'label_owners':owners})
    return result


def mark_incomplete_boundaries(observations,contract,labels,candidates):
    """Withhold a text prefix when actual same-row regions continue it.

    This does not delete a reading or resolve contradictory complete values.
    It adds a stricter eligibility condition and a source-bound reread hint.
    """
    nodes=lines(observations);by_observation={}
    for node in nodes:by_observation.setdefault(node['observation_id'],[]).append(node)
    for candidate in candidates.values():
        spec=contract['field_definitions'][candidate['field_key']]
        if spec['value_type'] not in TEXT_ASSEMBLY_TYPES or not candidate.get('role_verified'):continue
        spans=candidate.get('spans',[])
        if len(spans)!=1:continue
        span=spans[0]
        for node in by_observation.get(span['observation_id'],[]):
            if node['bbox'] is None or span.get('end')!=node['end']:continue
            continued=[]
            for other in nodes:
                if other['observation_id']==node['observation_id'] or label_hits(other['text'],labels):continue
                if not _row_connected(node,other):continue
                a,b=node['bbox'],other['bbox']
                # Rightward extension must add actual spatial coverage, rather
                # than a duplicate second reader over the same crop.
                if b[0]>=a[0] and b[2]>a[2]+.5*(a[3]-a[1]):continued.append(other)
            if continued:
                candidate['value_boundary_complete']=False
                candidate['boundary_review']={'reason':'observed_same_row_continuation',
                    'observation_ids':[node['observation_id'],*sorted({n['observation_id'] for n in continued})],
                    'bbox':_union([node,*continued]),'page_id':node['page_id'],
                    'source_sha256':node['source_sha256'],'purpose':'reread_complete_value_not_claim'}

def label_only(node,labels,field):
    hits=label_hits(node['text'],labels)
    if len(hits)!=1 or hits[0][2]!={field}:return None
    start,end,_=hits[0]
    if node['text'][:start].strip(' \t№:—-') or node['text'][end:].strip(' \t№:—-'):return None
    return {'observation_id':node['observation_id'],'start':node['start']+start,'end':node['start']+end}

def neighbor_values(label,nodes,labels,*,allow_value_colon=False,allow_above=False):
    # A whole-page OCR has text order but no honest line coordinates.
    after=sorted([n for n in nodes if n['observation_id']==label['observation_id'] and 1<=n['line_index']-label['line_index']<=2],key=lambda n:n['line_index'])
    if after and not label_hits(after[0]['text'],labels) and (allow_value_colon or ':' not in after[0]['text']):
        return [(after[0],{'kind':'consecutive_text_lines','label_line_id':label['id'],
            'value_line_id':after[0]['id'],'fabricated_line_geometry':False})]
    if label['bbox'] is None or not label.get('source_sha256') or label.get('page_id') is None:return []
    x0,y0,x1,y1=label['bbox'];lh=y1-y0;possible=[]
    for n in nodes:
        if (n['id']==label['id'] or n['bbox'] is None or n.get('source_sha256')!=label['source_sha256']
                or n.get('page_id')!=label['page_id'] or label_hits(n['text'],labels)
                or (not allow_value_colon and ':' in n['text'])):continue
        a,b,c,d=n['bbox'];nh=d-b;scale=max(lh,nh)
        y_overlap=max(0,min(y1,d)-max(y0,b));x_overlap=max(0,min(x1,c)-max(x0,a))
        right=(0<=a-x1<=12*scale and y_overlap>=.6*min(lh,nh))
        below=(-.2*scale<=b-y1<=2*scale and (x_overlap>=.5*min(x1-x0,c-a) or abs(a-x0)<=1.5*scale))
        above=(allow_above and -.2*scale<=y0-d<=1.5*scale
               and (x_overlap>=.5*min(x1-x0,c-a) or abs(a-x0)<=1.5*scale))
        if not (right or below or above):continue
        # A second labelled cell between the two anchors blocks association.
        cx=(a+c)/2;cy=(b+d)/2;blocked=False
        for other in nodes:
            if other['id'] in {label['id'],n['id']} or other['bbox'] is None:continue
            if other.get('source_sha256')!=label['source_sha256'] or other.get('page_id')!=label['page_id']:continue
            if not label_hits(other['text'],labels):continue
            q=other['bbox'];qx=(q[0]+q[2])/2;qy=(q[1]+q[3])/2
            if right and x1<=qx<a and max(y0,b)<=qy<=min(y1,d):blocked=True
            if below and y1<=qy<b and min(x0,a)<=qx<=max(x1,c):blocked=True
            if above and d<=qy<y0 and min(x0,a)<=qx<=max(x1,c):blocked=True
        if not blocked:
            possible.append((n,{'kind':'unique_spatial_neighbor','relation':'right' if right else 'below' if below else 'above',
                 'label_line_id':label['id'],'value_line_id':n['id'],'label_bbox':label['bbox'],
                 'value_bbox':n['bbox'],'source_sha256':label['source_sha256'],'page_id':label['page_id'],
                 'fabricated_line_geometry':False}))
    # Distinct OCR readings of ONE physical cell must all reach conflict
    # handling. Treating them as two competing spatial cells would silently
    # erase disagreement after a reread. Two genuinely different cells abstain.
    if len(possible)<=1:return possible
    first=possible[0][0]['bbox']
    def same_cell(box):
        area=(box[2]-box[0])*(box[3]-box[1]);base=(first[2]-first[0])*(first[3]-first[1])
        overlap=max(0,min(box[2],first[2])-max(box[0],first[0]))*max(0,min(box[3],first[3])-max(box[1],first[1]))
        return (overlap>=.8*min(area,base) and max(area,base)<=1.8*min(area,base)
            and abs((box[0]+box[2]-first[0]-first[2])/2)<=.3*max(box[2]-box[0],first[2]-first[0])
            and abs((box[1]+box[3]-first[1]-first[3])/2)<=.3*max(box[3]-box[1],first[3]-first[1]))
    if all(same_cell(n['bbox']) for n,_ in possible):
        return [(n,{**proof,'kind':'unique_spatial_cell_multiple_readings',
                    'cell_reading_line_ids':[x['id'] for x,_ in possible],
                    'reading_consensus_claimed':False}) for n,proof in possible]
    return []

def add_adjacent(observations,contract,labels,candidates,max_candidates):
    nodes=lines(observations)
    assembled=assembled_value_rows(nodes,labels)
    members={n['id'] for row in assembled for n in row['assembled_parts']}
    value_nodes=[n for n in nodes if n['id'] not in members]+[r for r in assembled if not r['assembly_ambiguous']]
    composite_labels=_composite_labels(nodes,labels)
    for field,spec in contract['field_definitions'].items():
        for label in [*nodes,*composite_labels]:
            if label.get('label_parts'):
                if label['label_owners']!={field}:continue
                role_spans=[{'observation_id':n['observation_id'],'start':n['start'],'end':n['end']} for n in label['label_parts']]
                label_part_ids={n['id'] for n in label['label_parts']}
            else:
                role_span=label_only(label,labels,field)
                if role_span is None:continue
                role_spans=[role_span];label_part_ids={label['id']}
            choices=value_nodes if spec['value_type'] in TEXT_ASSEMBLY_TYPES else nodes
            choices=[n for n in choices if n['id'] not in label_part_ids]
            for value,proof in neighbor_values(label,choices,labels,
                    allow_value_colon=spec['value_type'] in {'text','multispan_text','address','string'},
                    allow_above=spec['value_type'] in {'date','date_or_literal'}):
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
                    value_spans=_value_spans(value,start,end)
                    if not value_spans:continue
                    if value.get('assembled_parts'):
                        proof={**proof,'value_assembly':{'kind':'observed_horizontal_fragments',
                            'member_line_ids':[n['id'] for n in value['assembled_parts']],
                            'added_characters':'whitespace_only','coordinates_scope':'union_for_routing_only'}}
                    if label.get('label_parts'):
                        proof={**proof,'label_assembly':{'kind':'exact_observed_label_words',
                            'member_line_ids':[n['id'] for n in label['label_parts']],
                            'added_characters':'whitespace_only'}}
                    cid='c_'+digest([field,value_spans,role_spans,proof])[:24]
                    candidates[cid]={'id':cid,'field_key':field,'raw_value':raw,
                        'evidence_ids':sorted({s['observation_id'] for s in value_spans}),'spans':value_spans,
                        'party_role':spec['role'],'role_evidence_ids':sorted({s['observation_id'] for s in role_spans}),
                        'role_spans':role_spans,'role_verified':True,'association_proof':proof,
                        'assembly':'observed-horizontal-fragments-v1' if value.get('assembled_parts') else None,
                        'value_geometry_localized':value['bbox'] is not None,
                        'value_evidence_characters':sum({n['observation_id']:n['observation_text_characters']
                            for n in value.get('assembled_parts',[value])}.values()),
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
            'value_geometry_localized':group[0]=='spatial',
            'value_evidence_characters':sum({n['observation_id']:n['observation_text_characters']
                for n in items if n['observation_id'] in {p['span']['observation_id'] for p in ordered}}.values()),
            'candidate_generator':VERSION,'assembly':'rf-passport-explicit-name-components-v1',
            'association_proof':{'kind':'explicit_labeled_components','group':list(group),
                                 'added_characters':'whitespace_only'},'requires_layout_qualification':True}
        if len(candidates)>max_candidates:raise ValueError('candidate_limit_no_silent_truncation')
