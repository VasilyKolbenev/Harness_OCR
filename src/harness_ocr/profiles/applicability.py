"""Explicit server-owned applicability for fixed service/document-version scopes.

Nothing is inferred from OCR absence or a client's flags. An attestation is a
reference to expert review, not cryptographic proof that the review was correct.
"""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path
from ..core.common import digest
from .compiler import ProfileError

SCHEMA='harness-applicability/1'

class TrustedApplicabilityRegistry:
    def __init__(self,configuration=None):
        value=copy.deepcopy(configuration if configuration is not None else {'schema_version':SCHEMA,'entries':[]})
        if not isinstance(value,dict) or set(value)!={'schema_version','entries'} or value['schema_version']!=SCHEMA:
            raise ProfileError('invalid_applicability_registry')
        if not isinstance(value['entries'],list) or len(value['entries'])>1000:raise ProfileError('invalid_applicability_entries')
        self.entries={};self.configuration_sha256=digest(value)
        keys={'profile_id','profile_version','input_scope_version','configuration_version',
              'condition_context','basis','attestation_sha256'}
        for entry in value['entries']:
            if not isinstance(entry,dict) or set(entry)!=keys:raise ProfileError('invalid_applicability_entry')
            if entry['basis']!='fixed_service_contract':raise ProfileError('untrusted_applicability_basis')
            for key in ('profile_id','profile_version','input_scope_version','configuration_version'):
                if not isinstance(entry[key],str) or not 1<=len(entry[key])<=128:raise ProfileError('invalid_applicability_'+key)
            if not isinstance(entry['attestation_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',entry['attestation_sha256']):raise ProfileError('applicability_attestation_sha256_required')
            context=entry['condition_context']
            if not isinstance(context,dict) or any(type(v) is not bool or not isinstance(k,str) for k,v in context.items()):
                raise ProfileError('applicability_requires_explicit_boolean_conditions')
            key=(entry['profile_id'],entry['profile_version'],entry['input_scope_version'])
            if key in self.entries:raise ProfileError('duplicate_applicability_scope')
            self.entries[key]=entry
    @classmethod
    def from_file(cls,path):
        path=Path(path)
        return cls(json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None)
    def compile(self,profile_registry,profile_id,*,subset=None,input_scope_version):
        declaration=profile_registry.get(profile_id)
        version=declaration.get('profile_version',declaration.get('version','1.0.0'))
        key=(declaration['profile_id'],version,input_scope_version)
        entry=self.entries.get(key)
        # Validate full context before selecting a subset so typos cannot hide.
        if entry:profile_registry.compile(profile_id,condition_context=entry['condition_context'])
        context=entry['condition_context'] if entry else None
        compiled=profile_registry.compile(profile_id,subset=subset,condition_context=context)
        if entry is None:return compiled
        compiled['applicability_configuration']={
            'registry_sha256':self.configuration_sha256,'matched':entry is not None,
            'input_scope_version':input_scope_version,'entry':copy.deepcopy(entry),
            'authority':'trusted_server_fixed_service_contract' if entry else 'unresolved'}
        compiled['config_sha256']=digest({k:v for k,v in compiled.items() if k!='config_sha256'})
        return compiled
