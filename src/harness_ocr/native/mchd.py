"""Offline, original-XSD validated unified MChD 003 / EMCHD_1 adapter.

Only the ordinary, non-notarial B2B document variant is registered. XSD
validation proves structure, never a signature, current status or authority.
See docs/mchd_native.md for the exact supported scope and source publication.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

from lxml import etree as ET

ADAPTER_ID = 'fns-emchd-003-v1'
BRIDGE = 'mchd_xml_xsd_validated'
SCHEMA_VERSION = 'EMCHD_1'
SUPPORTED_VERSIONS = (SCHEMA_VERSION,)
PROFILE_ID = 'power_of_attorney_evidence_v1'
NAMESPACE = 'urn://x-artefacts/EMCHD_1'
NS = {'m': NAMESPACE}
XSD_NAME = 'ON_EMCHD_1_928_00_01_01_01.xsd'
SCHEMA_SHA256 = 'c8b9a4547074ae9d2c43ad24c81bd1ff7fe82afa419d3f7de951ec9d702c8780'
DATA_DIR = Path(__file__).with_name('data')
MAX_BYTES = 32 * 1024 * 1024
MAX_NODES = 20000
MAX_DEPTH = 64
BASE = '/m:Доверенность[1]/m:Документ[1]/m:Довер[1]'
PRINCIPAL_BRANCHES = {'1': 'РосОргДовер', '2': 'ИнОргДовер',
                      '3': 'ИПДовер', '4': 'ФЛДовер'}
REPRESENTATIVE_BRANCHES = {'1': 'СведОрг', '2': 'СведИП', '3': 'СведФизЛ',
                           '4': 'СведФилиал', '5': 'СведИО'}
KIND_ATTRS = {'inn': ('ИННЮЛ', 'ИННФЛ'), 'ogrn': ('ОГРН',),
              'ogrnip': ('ОГРНИП',), 'rf_passport': ('СерНомДок',),
              'snils': ('СНИЛС',), 'kpp': ('КПП',)}
KIND_VERSIONS = {'inn': 'inn-label-v1', 'ogrn': 'ogrn-label-v1',
                 'ogrnip': 'ogrnip-label-v1', 'rf_passport': 'rf-passport-series-number-v1',
                 'snils': 'snils-label-v1', 'kpp': 'kpp-label-v1'}


class NativeMchdError(ValueError):
    """Stable fail-closed error code, without leaking source personal data."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':')).encode('utf-8'))


def _bytes(source) -> bytes:
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_BYTES:
            raise NativeMchdError('mchd_size_limit')
        data = path.read_bytes()
    else:
        raise NativeMchdError('mchd_source_requires_bytes_or_path')
    if not data or len(data) > MAX_BYTES:
        raise NativeMchdError('mchd_size_limit')
    return data


class _DenyResolver(ET.Resolver):
    def resolve(self, url, public_id, context):
        raise NativeMchdError('mchd_external_resource_forbidden')


def _parser():
    parser = ET.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                          dtd_validation=False, attribute_defaults=False,
                          remove_blank_text=False, remove_comments=False,
                          remove_pis=False, huge_tree=False, recover=False)
    parser.resolvers.add(_DenyResolver())
    return parser


def _parse(data):
    try:
        root = ET.fromstring(data, parser=_parser())
    except (ET.XMLSyntaxError, ValueError) as exc:
        if isinstance(exc, NativeMchdError):
            raise
        raise NativeMchdError('mchd_xml_invalid_or_unsafe') from None
    tree = root.getroottree()
    if tree.docinfo.doctype or tree.docinfo.internalDTD or tree.docinfo.externalDTD:
        raise NativeMchdError('mchd_dtd_forbidden')
    if tree.xpath('//processing-instruction()'):
        raise NativeMchdError('mchd_processing_instruction_forbidden')
    count = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise NativeMchdError('mchd_xml_complexity_limit')
        if isinstance(node, ET._Entity):
            raise NativeMchdError('mchd_entities_forbidden')
        if not isinstance(node.tag, str):
            continue  # Comments do not supply mapped values.
        qname = ET.QName(node)
        if qname.namespace in {'http://www.w3.org/2000/09/xmldsig#',
                              'http://www.w3.org/2001/XInclude'}:
            raise NativeMchdError('mchd_signature_or_xinclude_not_supported')
        # Never resolve or use input-provided schemas, including local paths.
        if any(ET.QName(key).namespace == 'http://www.w3.org/2001/XMLSchema-instance'
               for key in node.attrib):
            raise NativeMchdError('mchd_dynamic_schema_or_type_forbidden')
        stack.extend((child, depth + 1) for child in node)
    return root


def _load_schema():
    # Re-read and hash on every invocation. A cached compiled schema must never
    # hide a damaged installation or a replaced on-disk validation asset.
    try:
        data = (DATA_DIR / XSD_NAME).read_bytes()
    except OSError:
        raise NativeMchdError('mchd_schema_asset_missing') from None
    if _sha(data) != SCHEMA_SHA256:
        raise NativeMchdError('mchd_schema_integrity_failure')
    try:
        doc = ET.fromstring(data, parser=_parser())
        ns = {'x': 'http://www.w3.org/2001/XMLSchema'}
        if doc.xpath('x:import|x:include|x:redefine', namespaces=ns):
            raise NativeMchdError('mchd_unregistered_schema_dependency')
        return ET.XMLSchema(doc)
    except ET.XMLSchemaParseError:
        raise NativeMchdError('mchd_schema_compilation_failure') from None


def _one(node, relative):
    found = node.findall('/'.join('m:' + part for part in relative.split('/')), NS)
    if len(found) != 1:
        raise NativeMchdError('mchd_required_exact_path_missing')
    return found[0]


def _locator(node):
    """Prefix-independent, absolute XPath with exact namespace and sibling index."""
    parts = []
    while node is not None:
        name = ET.QName(node)
        if name.namespace != NAMESPACE:
            raise NativeMchdError('mchd_namespace_not_registered')
        parent = node.getparent()
        peers = [c for c in parent if c.tag == node.tag] if parent is not None else [node]
        parts.append('m:' + name.localname + '[' + str(peers.index(node) + 1) + ']')
        node = parent
    return '/' + '/'.join(reversed(parts))


def _checked_branch(wrapper, container_name, type_attr, branches):
    container = _one(wrapper, container_name)
    children = [c for c in container if isinstance(c.tag, str)]
    expected = branches.get(wrapper.get(type_attr))
    # The official XSD has optional sequences here; the accompanying official
    # format tables 4.4/4.5 and 4.20/4.21 supply the discriminator semantics.
    if expected is None or len(children) != 1 or children[0].tag != '{' + NAMESPACE + '}' + expected:
        raise NativeMchdError('mchd_party_type_branch_mismatch')
    return children[0]


def decode_mchd(source, *, profile_id=PROFILE_ID, allowed_versions=SUPPORTED_VERSIONS):
    """Return deterministic observations, candidates, claims and all party lists.

    A ValueError subclass rejects every unregistered input; this function never
    downgrades a rejected MChD to generic XML. ``allowed_versions`` may narrow
    the server registry but cannot add a new format.
    """
    if profile_id != PROFILE_ID:
        raise NativeMchdError('mchd_adapter_not_registered_for_profile')
    if (not isinstance(allowed_versions, (tuple, list)) or
            any(v not in SUPPORTED_VERSIONS for v in allowed_versions)):
        raise NativeMchdError('mchd_unregistered_allowed_version')
    data = _bytes(source)
    source_sha = _sha(data)
    root = _parse(data)
    if root.tag != '{' + NAMESPACE + '}Доверенность':
        raise NativeMchdError('mchd_root_namespace_or_envelope_not_supported')
    if root.get('ВерсФорм') not in allowed_versions:
        raise NativeMchdError('mchd_version_not_registered')
    schema = _load_schema()
    if not schema.validate(root):
        # Do not include libxml error messages: they can echo source values.
        raise NativeMchdError('mchd_xsd_validation_failed')
    if root.get('ПрЭлФорм') != '00000000':
        raise NativeMchdError('mchd_element_flags_variant_not_registered')
    document = _one(root, 'Документ')
    if len(document) != 1 or document[0].tag != '{' + NAMESPACE + '}Довер':
        raise NativeMchdError('mchd_delegation_variant_not_registered')
    power = document[0]
    if power.find('m:СвНотУд', NS) is not None:
        raise NativeMchdError('mchd_notarial_variant_not_registered')
    info = _one(power, 'СвДов')
    if info.get('ВидДовер') != '1':
        raise NativeMchdError('mchd_power_kind_variant_not_registered')
    proof = {'adapter_id': ADAPTER_ID, 'adapter_version': '1.0',
             'schema_version': SCHEMA_VERSION, 'schema_sha256': SCHEMA_SHA256,
             'source_sha256': source_sha, 'mapper_sha256': _sha(Path(__file__).read_bytes()),
             'validator': 'lxml.etree.XMLSchema',
             'validator_version': '.'.join(map(str, ET.LXML_VERSION)),
             'structural_validation_status': 'pass', 'schema_dependencies': [],
             'network_access': False, 'input_schema_locations_used': False}
    observations = []
    observations_by_location = {}

    def observe(node, attr=None, *, mapping_id='unmapped', role=None, party_id='document'):
        xpath = _locator(node) + ('/@' + attr if attr else '/text()')
        value = node.get(attr) if attr else node.text
        if value is None or value == '':
            return None
        # Parsed XML character data is the exact XML-infoset value. The source
        # bytes and SHA retain lexical entity escaping/attribute whitespace.
        native = {k: proof[k] for k in ('adapter_id', 'schema_version', 'schema_sha256', 'source_sha256')}
        native.update(mapping_id=mapping_id, xpath=xpath, attribute=attr or '#text', party_id=party_id)
        oid = 'mx_' + _digest([source_sha, xpath])[:24]
        obs = {'id': oid, 'source_sha256': source_sha, 'text': value,
               'original_value': value, 'raw_value': value, 'evidence_kind': 'structured_xml',
               'xpath': xpath, 'namespace_map': NS, 'schema_version': SCHEMA_VERSION,
               'schema_sha256': SCHEMA_SHA256, 'structural_validation_status': 'pass',
               'native_adapter': ADAPTER_ID, 'native_proof': native,
               'native_validation': proof, 'party_role': role,
               'completed': True, 'capture_id': ADAPTER_ID, 'model_id': 'none'}
        previous = observations_by_location.get(xpath)
        if previous is not None:
            if previous != obs:
                raise NativeMchdError('mchd_duplicate_mapping_locator')
            return previous
        observations.append(obs)
        if len(observations) > 5000:
            raise NativeMchdError('mchd_observation_limit_no_silent_truncation')
        observations_by_location[xpath] = obs
        return obs

    def span(obs):
        return {'observation_id': obs['id'], 'start': 0, 'end': len(obs['text'])}

    candidates = []

    def candidate(field, raw, source_observations, role, *, party_id='document', assembly=None):
        source_observations = [o for o in source_observations if o]
        if not source_observations:
            return
        ident = 'nc_' + _digest([source_sha, field, party_id, raw])[:24]
        spans = [span(o) for o in source_observations]
        if assembly == 'name_parts':
            spans = [{**s, 'separator_after': ' ' if i + 1 < len(spans) else ''}
                     for i, s in enumerate(spans)]
        candidates.append({'id': ident, 'field_key': field, 'raw_value': raw,
                           'evidence_ids': [o['id'] for o in source_observations],
                           'spans': spans, 'party_role': role,
                           'role_evidence_ids': [o['id'] for o in source_observations],
                           'role_spans': [span(o) for o in source_observations],
                           'role_verified': True, 'party_id': party_id,
                           'candidate_generator': ADAPTER_ID,
                           'association_proof': {'kind': 'validated_native_xml',
                                                 'native_validation': source_observations[0]['native_proof']},
                           'assembly': assembly})
        if len(candidates) > 700:
            raise NativeMchdError('mchd_candidate_limit_no_silent_truncation')

    def party(wrapper, role, index):
        if role == 'principal':
            branch = _checked_branch(wrapper, 'Доверит', 'ТипДоверит', PRINCIPAL_BRANCHES)
        else:
            branch = _checked_branch(wrapper, 'Пред', 'ТипПред', REPRESENTATIVE_BRANCHES)
        branch_name = ET.QName(branch).localname
        pid = role + ':' + str(index)
        if branch_name == 'РосОргДовер':
            entity = _one(branch, 'СвРосОрг'); name_attribute = 'НаимОрг'
        elif branch_name == 'ИнОргДовер':
            entity = _one(branch, 'СвИнОрг'); name_attribute = 'НаимИО'
        elif branch_name in {'СведОрг', 'СведФилиал', 'СведИО'}:
            entity = branch; name_attribute = 'НаимИО' if branch_name == 'СведИО' else 'НаимОрг'
        else:
            entity = branch; name_attribute = None
        names = []
        identity = {}
        if name_attribute:
            obs = observe(entity, name_attribute, mapping_id='party.identity.organization', role=role, party_id=pid)
            names.append(obs); identity[name_attribute] = obs['text']
        else:
            fio = _one(entity, 'СведФЛ/ФИО')
            for attr in ('Фамилия', 'Имя', 'Отчество'):
                if fio.get(attr) is not None:
                    obs = observe(fio, attr, mapping_id='party.identity.person.' + attr, role=role, party_id=pid)
                    names.append(obs); identity[attr] = obs['text']
        candidate(role + '.identity', ' '.join(o['text'] for o in names), names, role,
                  party_id=pid, assembly='name_parts' if not name_attribute else None)
        identifiers = []
        for attr in entity.attrib:
            kind = next((k for k, attrs in KIND_ATTRS.items() if attr in attrs and k != 'rf_passport'), None)
            if kind:
                identifiers.append((kind, observe(entity, attr, mapping_id='party.identifier.' + kind,
                                                  role=role, party_id=pid)))
        iddoc = entity.find('m:СведФЛ/m:УдЛичнФЛ', NS)
        code_observation = None
        if iddoc is not None and iddoc.get('КодВидДок') == '21':
            code_observation = observe(iddoc, 'КодВидДок', mapping_id='party.identifier.passport_document_type', role=role, party_id=pid)
            identifiers.append(('rf_passport', observe(iddoc, 'СерНомДок',
                                mapping_id='party.identifier.rf_passport', role=role, party_id=pid)))
        items = []
        for number, (kind, obs) in enumerate(identifiers, 1):
            item_span = [span(obs)]
            items.append({'item_id': 'i' + str(number), 'kind': kind, 'version': KIND_VERSIONS[kind],
                          'raw_value': obs['text'], 'party_role': role,
                          'spans': item_span, 'kind_spans': item_span, 'role_spans': item_span,
                          'version_spans': item_span,
                          'association': {'kind': 'validated_native_xml', 'native_validation': obs['native_proof']}})
            if kind == 'rf_passport':
                items[-1]['kind_spans'] = [*item_span, span(code_observation)]
        if items:
            raw = {'schema_version': 'typed-fields-v1', 'contract_id': 'party-identifiers-v1',
                   'contract_version': '1.0', 'value_type': 'typed_identifier', 'items': items}
            candidate(role + '.identifier', raw, [*[o for _, o in identifiers], code_observation], role, party_id=pid)
        return {'party_id': pid, 'party_role': role, 'type_code': wrapper.get('ТипДоверит' if role == 'principal' else 'ТипПред'),
                'branch': branch_name, 'xpath': _locator(wrapper), 'identity': identity,
                'identifiers': items, 'identity_evidence_ids': [o['id'] for o in names],
                'unmapped_identifier_documents': ([{'xpath': _locator(iddoc),
                    'attributes': dict(iddoc.attrib),
                    'children': [{'element': ET.QName(c).localname, 'raw_value': c.text}
                                 for c in iddoc if isinstance(c.tag, str)]}]
                    if iddoc is not None and iddoc.get('КодВидДок') != '21' else [])}

    principals = [party(node, 'principal', i) for i, node in enumerate(power.findall('m:СвДоверит', NS), 1)]
    representatives = [party(node, 'representative', i) for i, node in enumerate(power.findall('m:СвУпПред', NS), 1)]
    for attr, field in (('ДатаВыдДовер', 'power.issue_date'), ('СрокДейст', 'power.valid_until'), ('НомДовер', 'power.number')):
        obs = observe(info, attr, mapping_id=field, role='document')
        candidate(field, obs['text'], [obs], 'document')
    # Preserve internal and registry numbers as distinct source data. The D04
    # scalar power.number maps the unified registration number НомДовер only.
    internal = observe(info, 'ВнНомДовер', mapping_id='power.internal_number', role='document')
    authority = _one(power, 'СвПолн')
    kind = authority.get('ТипПолн')
    texts = authority.findall('m:ТекстПолн', NS)
    machine = authority.findall('m:МашПолн', NS)
    if (kind == '0' and (len(texts) != 1 or machine)) or (kind == '1' and (texts or not machine)):
        raise NativeMchdError('mchd_authority_type_branch_mismatch')
    machine_powers = []
    review_reasons = []
    unmapped_identifier_parties = {p['party_id'] for p in [*principals, *representatives]
                                   if p['unmapped_identifier_documents']}
    review_reasons.extend(pid + '.identifier:identity_document_kind_not_registered'
                          for pid in sorted(unmapped_identifier_parties))
    if kind == '0':
        obs = observe(texts[0], mapping_id='power.authority_literal', role='authority')
        candidate('power.authority_literal', obs['text'], [obs], 'authority')
    else:
        # A coded authority with restrictions is a structured list. Never turn
        # its title alone into a complete literal legal authority claim.
        for node in machine:
            record = {'xpath': _locator(node), 'attributes': dict(node.attrib), 'restrictions': []}
            for attr in node.attrib:
                observe(node, attr, mapping_id='power.machine.' + attr, role='authority')
            for restriction in node.findall('m:ОгрСвПолн', NS):
                children = [c for c in restriction if isinstance(c.tag, str)]
                record['restrictions'].append({'xpath': _locator(restriction), 'attributes': dict(restriction.attrib),
                                                'values': [{'element': ET.QName(c).localname, 'raw_value': c.text} for c in children]})
                for attr in restriction.attrib:
                    observe(restriction, attr, mapping_id='power.machine.restriction.' + attr, role='authority')
                for child in children:
                    observe(child, mapping_id='power.machine.restriction.value', role='authority')
            machine_powers.append(record)
        review_reasons.append('power.authority_literal:structured_machine_authority_requires_service_contract')
    # An entire role remains a structured ordered list in native_model. Scalar
    # D04 identity/identifier claims are emitted only for one party in that role.
    multiple = {role for role, parties in (('principal', principals), ('representative', representatives)) if len(parties) != 1}
    review_reasons.extend(role + ':multiple_parties_require_structured_service_contract' for role in sorted(multiple))
    # The source-replayed complete-value digest prevents a claimant with only
    # selected evidence IDs from omitting, duplicating or reordering identifiers.
    proof['identifier_value_sha256_by_party'] = {
        c['party_id']: _digest(c['raw_value']) for c in candidates
        if c['field_key'] in {'principal.identifier', 'representative.identifier'}}
    claims = []
    for c in candidates:
        if c['party_role'] in multiple:
            continue
        if c['field_key'].endswith('.identifier') and c['party_id'] in unmapped_identifier_parties:
            continue
        claims.append({'id': 'ncl_' + c['id'][3:], 'field_key': c['field_key'], 'raw_value': c['raw_value'],
                       'evidence_ids': c['evidence_ids'], 'spans': c['spans'], 'party_role': c['party_role'],
                       'role_evidence_ids': c['role_evidence_ids'], 'role_spans': c['role_spans'],
                       'association_proofs': [c['association_proof']], 'assembly': c['assembly'],
                       'origin': 'rule', 'depends_on': []})
    return {'source_sha256': source_sha, 'pages': [], 'observations': observations,
            'field_candidates': candidates, 'native_claims': claims, 'native_adapter': ADAPTER_ID,
            'native_bridge': BRIDGE, 'native_validation': proof, 'structural_validation_status': 'pass',
            'native_model': {'principals': principals, 'representatives': representatives,
                             'authority_type': kind, 'machine_powers': machine_powers,
                             'internal_number': internal['text'], 'joint_exercise_code': authority.get('ПрСовмПолн')},
            'review_reasons': review_reasons, 'signature_verified': False,
            'verification': {'signature_validity': 'not_verified', 'signer_trust': 'not_verified',
                             'revocation_status': 'not_verified', 'authorization': 'not_verified',
                             'document_current_validity': 'not_verified'}}


@lru_cache(maxsize=4)
def _replay(data, source_sha, mapper_sha):
    # Only private replay results enter this cache; public decode results remain
    # independent mutable values. Cache identity includes original bytes, code
    # and a SHA rechecked by _verified_replay on every call.
    return decode_mchd(data)


def _verified_replay(source):
    data = _bytes(source)
    _load_schema()
    return _replay(data, _sha(data), _sha(Path(__file__).read_bytes()))


def verify_native_result(result, source) -> bool:
    """Replay against server-owned original source; hashes alone are not proof."""
    try:
        return isinstance(result, dict) and result == _verified_replay(source)
    except (NativeMchdError, OSError, TypeError):
        return False


def verify_native_observation(observation, source) -> bool:
    """Replay exact evidence, ignoring only state-store managed revision flags."""
    try:
        clean = {k: v for k, v in observation.items() if k not in {'active', 'created_revision'}}
        return any(clean == issued for issued in _verified_replay(source)['observations'])
    except (NativeMchdError, OSError, TypeError, AttributeError):
        return False


def _identifier_locator_valid(proof, kind, role):
    """Exact namespace-bound path grammar, never local-name/substring inference."""
    if not isinstance(proof, dict) or set(proof) != {'adapter_id', 'schema_version', 'schema_sha256',
            'mapping_id', 'source_sha256', 'xpath', 'attribute', 'party_id'}:
        return False
    if (proof['adapter_id'] != ADAPTER_ID or proof['schema_version'] != SCHEMA_VERSION or
            proof['schema_sha256'] != SCHEMA_SHA256 or proof['mapping_id'] != 'party.identifier.' + kind or
            re.fullmatch(r'[0-9a-f]{64}', proof['source_sha256']) is None):
        return False
    if proof['attribute'] not in KIND_ATTRS.get(kind, ()):
        return False
    prefix = BASE + ('/m:СвДоверит' if role == 'principal' else '/m:СвУпПред')
    match = re.fullmatch(re.escape(prefix) + r'\[([1-9][0-9]*)\](.*)', proof['xpath'])
    if match is None or proof['party_id'] != role + ':' + match[1]:
        return False
    suffixes = []
    if role == 'principal':
        paths = ['Доверит/РосОргДовер/СвРосОрг', 'Доверит/ИнОргДовер/СвИнОрг',
                 'Доверит/ИПДовер', 'Доверит/ФЛДовер']
    elif role == 'representative':
        paths = ['Пред/' + v for v in REPRESENTATIVE_BRANCHES.values()]
    else:
        return False
    for path in paths:
        is_person = path.endswith(('ИПДовер', 'ФЛДовер', 'СведИП', 'СведФизЛ'))
        if kind in {'ogrnip', 'rf_passport'} and not path.endswith(('ИПДовер', 'ФЛДовер', 'СведИП', 'СведФизЛ')):
            continue
        if kind == 'ogrnip' and not path.endswith(('ИПДовер', 'СведИП')):
            continue
        if kind == 'ogrn' and not path.endswith(('СвРосОрг', 'СведОрг', 'СведФилиал')):
            continue
        if kind == 'snils' and not is_person or kind == 'kpp' and is_person:
            continue
        base = ''.join('/m:' + p + '[1]' for p in path.split('/'))
        if kind == 'rf_passport':
            base += '/m:СведФЛ[1]/m:УдЛичнФЛ[1]'
        attrs = (('ИННФЛ',) if is_person else ('ИННЮЛ',)) if kind == 'inn' else KIND_ATTRS[kind]
        for attr in attrs:
            suffixes.append(base + '/@' + attr)
    return match[2] in suffixes


def verify_typed_xml_item(item, observations) -> bool:
    """Validate native typed association after core source replay validation.

    This is a locator/kind/role check, not an authenticity claim. The state store
    must first replay every native observation against its server-owned source.
    """
    try:
        kind = item['kind']; role = item['party_role']
        proof = item['association']['native_validation']
        if item['version'] != KIND_VERSIONS.get(kind) or not _identifier_locator_valid(proof, kind, role):
            return False
        by_id = {o['id']: o for o in observations}
        if len(item['spans']) != 1:
            return False
        value_span = item['spans'][0]
        obs = by_id[value_span['observation_id']]
        if (obs.get('native_adapter') != ADAPTER_ID or obs.get('native_proof') != proof or
                obs.get('source_sha256') != proof['source_sha256'] or obs.get('namespace_map') != NS or
                obs.get('xpath') != proof['xpath'] or obs.get('structural_validation_status') != 'pass' or
                obs.get('text') != item['raw_value'] or value_span['start'] != 0 or
                value_span['end'] != len(obs['text']) or obs.get('party_role') != role):
            return False
        if kind == 'rf_passport':
            code_path = proof['xpath'].rsplit('/@', 1)[0] + '/@КодВидДок'
            code = next((o for o in observations if o.get('xpath') == code_path and o.get('text') == '21' and
                       o.get('native_adapter') == ADAPTER_ID and o.get('source_sha256') == proof['source_sha256']
                       and o.get('party_role') == role), None)
            if code is None:
                return False
            expected_kind = [*item['spans'], {'observation_id': code['id'], 'start': 0, 'end': 2}]
        else:
            expected_kind = item['spans']
        return (item['kind_spans'] == expected_kind and
                all(item[name] == item['spans'] for name in ('role_spans', 'version_spans')))
    except (KeyError, TypeError, AttributeError):
        return False


def verify_typed_xml_value(raw, observations) -> bool:
    """Check the whole ordered issued party value, including completeness."""
    try:
        if (set(raw) != {'schema_version', 'contract_id', 'contract_version', 'value_type', 'items'} or
                raw['schema_version'] != 'typed-fields-v1' or raw['contract_id'] != 'party-identifiers-v1' or
                raw['contract_version'] != '1.0' or raw['value_type'] != 'typed_identifier' or not raw['items']):
            return False
        if not all(verify_typed_xml_item(item, observations) for item in raw['items']):
            return False
        proofs = [item['association']['native_validation'] for item in raw['items']]
        bindings = {(p['source_sha256'], p['party_id']) for p in proofs}
        if len(bindings) != 1:
            return False
        source_sha, party_id = next(iter(bindings))
        expected = _digest(raw)
        value_observation_ids = {s['observation_id'] for item in raw['items'] for s in item['spans']}
        return all(o.get('native_validation', {}).get('identifier_value_sha256_by_party', {}).get(party_id) == expected
                   and o.get('source_sha256') == source_sha
                   for o in observations if o.get('id') in value_observation_ids)
    except (KeyError, TypeError, AttributeError):
        return False
