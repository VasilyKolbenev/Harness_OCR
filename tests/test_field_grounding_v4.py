"""Evidence/geometry contracts, independent synthetic values; no NN."""
from copy import deepcopy
import json
import unittest

from harness_ocr.core.evidence import literal_supported
from harness_ocr.grounding import (build_candidates,candidate_is_eligible,qwen_prompt,
    resolve_response,rank_reread_regions,field_region_hints,validate_literal)

SHA='a'*64

def contract(*fields):
    definitions={key:{'key':key,'label':label,'label_aliases':[],'role':role,
        'value_type':kind,'critical':True} for key,label,kind,role in fields}
    return {'profile_id':'unseen_synthetic_layout','field_definitions':definitions,
        'requested_fields':list(definitions),'candidate_contract':{'version':'test-v1'}}

def obs(ident,text,box=None,*,source=SHA,page='p1',model='same-reader'):
    value={'id':ident,'text':text,'source_sha256':source,'completed':True,
        'evidence_kind':'image_region','page':1,'page_id':page,'source_size':[1000,1500],
        'model_id':model}
    if box:
        a,b,c,d=box;value['quad']=[[a,b],[c,b],[c,d],[a,d]]
    else:value['bbox']=[0,0,1000,1500]
    return value

def region(ident,observation):
    return {k:v for k,v in {'id':ident,'page':1,'page_id':observation['page_id'],
        'source_sha256':observation['source_sha256'],'source_size':observation['source_size'],
        'quad':observation.get('quad'),'bbox':observation.get('bbox')}.items() if v is not None}

def qualified(candidates,field):
    return [c for c in candidates.values() if c['field_key']==field and candidate_is_eligible(c)]

class FieldGroundingV4Tests(unittest.TestCase):
    def test_whitespace_label_variants_preserve_invalid_calendar_literal(self):
        c=contract(('holder.birth_date','Дата рождения','date','holder'))
        source=[obs('new','Дата\t  рождения: 31.02.2021')]
        found=qualified(build_candidates(source,c),'holder.birth_date')
        self.assertTrue(found);self.assertEqual({v['raw_value'] for v in found},{'31.02.2021'})
        self.assertEqual(validate_literal(found[0]['raw_value'],c['field_definitions']['holder.birth_date'])['validation_status'],'fail')

    def test_quoted_role_word_is_not_a_new_heading(self):
        c=contract(('payer.name','Плательщик','organization_name','payer'),
                   ('payee.name','Получатель','organization_name','payee'))
        found=build_candidates([obs('one','Плательщик: ООО «Получатель»')],c)
        self.assertEqual({x['raw_value'] for x in qualified(found,'payer.name')},{'ООО «Получатель»'})
        self.assertFalse(qualified(found,'payee.name'))

    def test_split_label_has_actual_role_spans_not_invented_text(self):
        c=contract(('holder.birth_date','Дата рождения','date','holder'))
        source=[obs('la','Дата',[10,10,40,25]),obs('lb','рождения',[45,10,115,25]),
                obs('v','29.02.2024',[10,35,115,55])]
        found=qualified(build_candidates(source,c),'holder.birth_date')
        self.assertTrue(found)
        self.assertTrue(any(set(v['role_evidence_ids'])=={'la','lb'} for v in found))
        self.assertTrue(all(literal_supported(v['raw_value'],source,v['spans']) for v in found))

    def test_date_can_be_above_explicit_label_but_two_cells_abstain(self):
        c=contract(('holder.birth_date','Дата рождения','date','holder'))
        source=[obs('v','29.02.2024',[10,10,115,30]),obs('label','Дата рождения',[10,35,130,50])]
        self.assertTrue(qualified(build_candidates(source,c),'holder.birth_date'))
        source.append(obs('other','01.01.2020',[10,58,115,78]))
        self.assertFalse(qualified(build_candidates(source,c),'holder.birth_date'))

    def test_complete_observed_word_chain_and_distinct_columns(self):
        c=contract(('payee.name','Получатель','organization_name','payee'),
                   ('payer.name','Плательщик','person_name','payer'))
        source=[obs('label','Получатель',[10,10,100,30]),
            obs('a','ООО',[10,40,45,60]),obs('b','«Гранитный',[50,40,155,60]),obs('c','Мир»',[160,40,210,60]),
            obs('pl','Плательщик',[500,10,610,30]),obs('pa','Сергеев',[500,40,590,60]),obs('pb','Илья',[595,40,650,60])]
        found=build_candidates(source,c)
        self.assertEqual({v['raw_value'] for v in qualified(found,'payee.name')},{'ООО «Гранитный Мир»'})
        self.assertEqual({v['raw_value'] for v in qualified(found,'payer.name')},{'Сергеев Илья'})
        for field in ('payee.name','payer.name'):
            value=qualified(found,field)[0]
            self.assertTrue(literal_supported(value['raw_value'],source,value['spans']))
            self.assertGreater(len(value['spans']),1)

    def test_inline_prefix_is_withheld_with_full_context_hint(self):
        c=contract(('payee.name','Получатель','organization_name','payee'))
        source=[obs('a','Получатель: ООО',[10,10,150,30]),obs('b','«Базальт»',[155,10,250,30])]
        found=build_candidates(source,c)
        prefixes=[v for v in found.values() if v['raw_value']=='ООО' and v['role_verified']]
        self.assertTrue(prefixes);self.assertTrue(all(v['value_boundary_complete'] is False for v in prefixes))
        self.assertFalse(qualified(found,'payee.name'))
        regions=[region('first',source[0]),region('next',source[1])]
        hints=field_region_hints(found,regions,c,source,max_regions=1)
        self.assertTrue(hints);self.assertGreaterEqual(hints[0]['bbox'][2],250)
        self.assertFalse(hints[0]['semantic_role_established'])
        response=json.dumps({'proposals':[{'field_key':'payee.name','candidate_ids':[prefixes[0]['id']],
            'party_role':'payee'}],'reread_requests':[]})
        self.assertEqual(resolve_response(response,found,regions,c)['rejected'][0]['reason'],'incomplete_value_boundary')

    def test_nested_conflicting_ink_is_reread_not_silently_dropped(self):
        c=contract(('payee.name','Получатель','organization_name','payee'))
        source=[obs('label','Получатель',[10,10,100,30]),obs('a','ООО',[10,40,60,60]),
                obs('b','Базальт',[65,40,155,60]),obs('noise','–',[90,45,100,55])]
        candidates=build_candidates(source,c)
        self.assertFalse(qualified(candidates,'payee.name'))
        hints=field_region_hints(candidates,[region('l',source[0]),region('v',source[1])],c,source)
        self.assertTrue(hints);self.assertGreaterEqual(hints[0]['bbox'][2],155)

    def test_unknown_reader_streams_do_not_authorize_word_concatenation(self):
        c=contract(('payee.name','Получатель','organization_name','payee'))
        source=[obs('label','Получатель',[10,10,100,30],model=None),
                obs('a','ООО',[10,40,50,60],model=None),obs('b','Гранит',[55,40,130,60],model=None)]
        found=qualified(build_candidates(source,c),'payee.name')
        self.assertFalse(any(v['raw_value']=='ООО Гранит' for v in found))

    def test_unlabelled_passport_values_route_by_type_without_role_promotion(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        source=[obs('noise','СЕРГЕЕВ',[10,10,130,30]),obs('digits','483920',[700,300,730,500])]
        regions=[region('first',source[0]),region('number-context',source[1])]
        candidates=build_candidates(source,c);saved=deepcopy(candidates)
        self.assertFalse(qualified(candidates,'passport.number'))
        requests=rank_reread_regions(candidates,regions,c,source,max_regions=1)
        self.assertEqual(requests,[{'field_key':'passport.number','region_ids':['number-context'],'reason_code':'ambiguous_role'}])
        self.assertEqual(candidates,saved)

    def test_routing_does_not_cross_sources_or_reuse_spent_region(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        source=[obs('digits','483920',[10,10,100,30])];candidates=build_candidates(source,c)
        forged=region('foreign',source[0]);forged['source_sha256']='b'*64
        self.assertFalse(rank_reread_regions(candidates,[forged],c,source))
        correct=region('same',source[0])
        self.assertFalse(rank_reread_regions(candidates,[correct],c,source,used_region_ids=['same']))

    def test_conflicting_complete_numeric_values_remain_conflicting(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        source=[obs('a','Номер паспорта: 483920'),obs('b','Номер паспорта: 483921',model='other-reader')]
        candidates=build_candidates(source,c);values=qualified(candidates,'passport.number')
        result=resolve_response(json.dumps({'proposals':[{'field_key':'passport.number',
            'candidate_ids':[v['id'] for v in values],'party_role':'document'}],'reread_requests':[]}),candidates,[],c)
        self.assertFalse(result['claims']);self.assertEqual(result['rejected'][0]['reason'],'candidate_disagreement')

    def test_equal_readings_have_stable_local_value_spans_and_all_corroboration(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        source=[obs('page','Заголовок\nНомер паспорта: 483920\nПрочие данные',model='whole-page'),
                obs('local','Номер паспорта: 483920',[10,10,220,30])]
        candidates=build_candidates(source,c);ids=[x['id'] for x in qualified(candidates,'passport.number')]
        self.assertEqual(len(ids),2)
        outputs=[]
        for chosen in (ids,list(reversed(ids))):
            result=resolve_response(json.dumps({'proposals':[{'field_key':'passport.number',
                'candidate_ids':chosen,'party_role':'document'}],'reread_requests':[]}),candidates,[],c)
            claim=result['claims'][0];outputs.append(claim)
            self.assertEqual(claim['evidence_ids'],['local','page'])
            self.assertEqual({s['observation_id'] for s in claim['spans']},{'local'})
        self.assertEqual(outputs[0],outputs[1])

    def test_reread_disagreement_in_same_cell_is_not_erased_as_layout_ambiguity(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        source=[obs('label','ИНН получателя',[10,10,160,30]),
            obs('ctc','7707083893',[10,40,160,60]),obs('again','7707083894',[10,40,160,60],model='second-reader')]
        candidates=build_candidates(source,c);found=qualified(candidates,'payee.inn')
        self.assertEqual({x['raw_value'] for x in found},{'7707083893','7707083894'})
        self.assertTrue(all(v['association_proof']['reading_consensus_claimed'] is False for v in found))
        result=resolve_response(json.dumps({'proposals':[{'field_key':'payee.inn',
            'candidate_ids':[x['id'] for x in found],'party_role':'payee'}],'reread_requests':[]}),candidates,[],c)
        self.assertFalse(result['claims']);self.assertEqual(result['rejected'][0]['reason'],'candidate_disagreement')

    def test_free_text_punctuation_is_literal_not_a_field_separator(self):
        c=contract(('payment.purpose','Назначение платежа','text','payment'))
        candidates=build_candidates([obs('a','Назначение платежа: Тест: студия; заявка 17;')],c)
        self.assertEqual({v['raw_value'] for v in qualified(candidates,'payment.purpose')},{'Тест: студия; заявка 17;'})

    def test_large_irrelevant_source_is_explicitly_out_of_context(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        source=[obs('a','Номер паспорта: 483920'),obs('background','ДОКУМЕНТНЫЕ ДАННЫЕ '*2000)]
        candidates=build_candidates(source,c);saved=deepcopy((source,candidates))
        prompt=json.loads(qwen_prompt(candidates,[],c,observations=source))
        self.assertEqual(prompt['source_context']['coverage'],'field_scoped_exact_excerpts')
        self.assertEqual({n['observation_id'] for n in prompt['source_line_graph']},{'a'})
        self.assertTrue(prompt['source_context']['all_source_retained_on_server'])
        self.assertEqual((source,candidates),saved)
        self.assertLess(len(json.dumps(prompt,ensure_ascii=False)),24000)

    def test_unqualified_text_inventory_does_not_exhaust_field_candidate_budget(self):
        c=contract(('payee.name','Получатель','organization_name','payee'))
        text='\n'.join(['Прочий неразмеченный текст документа']*1000+['Получатель: ООО «Лазурит»'])
        source=[obs('long',text)];candidates=build_candidates(source,c)
        self.assertEqual({v['raw_value'] for v in qualified(candidates,'payee.name')},{'ООО «Лазурит»'})
        self.assertEqual(len(candidates),1)
        self.assertEqual(source[0]['text'],text)
        prompt=json.loads(qwen_prompt(candidates,[],c,observations=source))
        self.assertIn('ООО «Лазурит»','\n'.join(n['text'] for n in prompt['source_line_graph']))
        self.assertEqual(prompt['source_context']['original_nodes'],1001)

    def test_source_excerpt_has_exact_offsets_and_no_character_box(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'))
        text='длинный фон '*1000+' Номер паспорта: 483920 '+'хвост '*1000
        source=[obs('a',text)];candidates=build_candidates(source,c)
        prompt=json.loads(qwen_prompt(candidates,[],c,observations=source))
        self.assertTrue(prompt['source_line_graph'])
        for n in prompt['source_line_graph']:
            self.assertEqual(n['text'],text[n['start']:n['end']])
            self.assertFalse(n['coordinates_exact_for_line'])
        self.assertIn('483920',prompt['source_line_graph'][0]['text'])

    def test_oversized_candidate_set_requires_partition_never_truncation(self):
        c=contract(('payee.name','Получатель','organization_name','payee'))
        source=[obs('a','Получатель: '+'Ж'*30000)];candidates=build_candidates(source,c);saved=deepcopy(candidates)
        with self.assertRaisesRegex(ValueError,'requires_partition|requires_field_partition'):
            qwen_prompt(candidates,[],c,observations=source)
        self.assertEqual(candidates,saved)

    def test_prompt_field_scope_and_region_coverage_are_explicit(self):
        c=contract(('passport.number','Номер паспорта','digits_6','document'),('holder.birth_date','Дата рождения','date','holder'))
        source=[obs('a','Номер паспорта: 483920'),obs('b','Дата рождения: 29.02.2024')]
        candidates=build_candidates(source,c);regions=[{'id':'r'+str(i),'page_id':'p1','bbox':[0,0,10,10]} for i in range(100)]
        prompt=json.loads(qwen_prompt(candidates,regions,c,observations=source,field_keys=['passport.number'],max_regions=3))
        self.assertEqual([f['key'] for f in prompt['fields']],['passport.number'])
        self.assertTrue(all(v['field_key']=='passport.number' for v in prompt['candidates']))
        self.assertEqual(len(prompt['regions']),3);self.assertEqual(prompt['region_projection']['server_regions'],100)

    def cross_reader_sources(self,raw='7707083893',*,whole_text=None,local_text=None):
        return [obs('whole',whole_text or 'ИНН получателя: '+raw,model='whole-page-reader'),
                obs('cell',local_text or raw,[10,10,180,35],model='cyrillic-ctc')]

    def test_unique_exact_cross_reader_role_uses_real_ctc_pixel_span(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        source=self.cross_reader_sources();saved=deepcopy(source);candidates=build_candidates(source,c)
        mapped=[x for x in candidates.values() if x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader']
        self.assertEqual(len(mapped),1);value=mapped[0]
        self.assertEqual(value['evidence_ids'],['cell']);self.assertEqual(value['role_evidence_ids'],['whole'])
        self.assertEqual(value['raw_value'],'7707083893');self.assertTrue(value['role_verified'])
        self.assertTrue(value['requires_layout_qualification']);self.assertFalse(value['association_proof']['automatic_acceptance'])
        self.assertFalse(value['association_proof']['statistical_independence_claimed'])
        self.assertTrue(literal_supported(value['raw_value'],source,value['spans']))
        self.assertEqual(source,saved)

    def test_cross_reader_invalid_checksum_stays_exact_and_rejected(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        mapped=[x for x in build_candidates(self.cross_reader_sources('7707083894'),c).values()
            if x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader']
        self.assertEqual(len(mapped),1);self.assertEqual(mapped[0]['raw_value'],'7707083894')
        self.assertEqual(validate_literal(mapped[0]['raw_value'],c['field_definitions']['payee.inn'])['validation_status'],'fail')

    def test_cross_reader_repetition_and_two_spatial_cells_abstain(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        repeated=self.cross_reader_sources(whole_text='ИНН получателя: 7707083893\nКопия: 7707083893')
        two=self.cross_reader_sources();two.append(obs('second','7707083893',[10,90,180,115],model='cyrillic-ctc'))
        for source in (repeated,two):
            with self.subTest(source=source):
                self.assertFalse(any(x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader' for x in build_candidates(source,c).values()))

    def test_cross_reader_near_matches_cross_pages_sources_and_other_roles_abstain(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        variants=[]
        near=self.cross_reader_sources();near[1]['text']='7707083894';variants.append(near)
        page=self.cross_reader_sources();page[1]['page_id']='p2';variants.append(page)
        foreign=self.cross_reader_sources();foreign[1]['source_sha256']='b'*64;variants.append(foreign)
        role=self.cross_reader_sources(local_text='ИНН плательщика: 7707083893');variants.append(role)
        no_label=self.cross_reader_sources(whole_text='Реквизиты\n7707083893');variants.append(no_label)
        embedded=self.cross_reader_sources(local_text='X7707083893Y');variants.append(embedded)
        for source in variants:
            with self.subTest(source=source):
                self.assertFalse(any(x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader' for x in build_candidates(source,c).values()))

    def test_cross_reader_coincident_reads_allow_exact_but_preserve_disagreement(self):
        c=contract(('payee.inn','ИНН получателя','inn','payee'))
        same=self.cross_reader_sources();same[1]['reading_stream_id']='db'
        reread=obs('second','7707083893',[10,10,180,35],model='cyrillic-ctc');reread['reading_stream_id']='yolo'
        same.append(reread)
        found=build_candidates(same,c)
        self.assertEqual(len([x for x in found.values() if x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader']),2)
        conflict=deepcopy(same);conflict[2]['text']='7707083894'
        found=build_candidates(conflict,c)
        self.assertFalse(any(x.get('association_proof',{}).get('kind')=='unique_literal_cross_reader' for x in found.values()))
        self.assertTrue(any(x['raw_value']=='7707083894' for x in found.values()))

if __name__=='__main__':unittest.main()
