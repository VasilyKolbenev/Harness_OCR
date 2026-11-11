"""Versioned checksum consistency, not account truth or account classification.

The caller must establish where the account is opened before choosing a rule.
This function never tries both prefixes, guesses a location from a matching
checksum, repairs a digit, strips separators or substitutes look-alike letters.
The pinned sources are legal reproductions of the 1997 normative text, explicitly
identified as such. They are not presented as an original CBR-hosted PDF or as a
verification of present legal applicability.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ALGORITHM_ID = 'RU20-CBR515-1997-v1'
SCOPE = 'arithmetic_consistency_only'
REGISTRY_PATH = Path(__file__).with_name('data') / 'bank_key_sources.json'
REGISTRY_SHA256 = '2e3120170fedc37099140933b3f95b6d244ddb9df188b65bbfb3c65ce0d87c5d'
SOURCE_IDS = ('CBR515-KONTUR-19970908', 'CBR515-ZAKONBASE-19970908', 'CBR515-RULAWS-19970908')
LOCATIONS = {'credit_institution', 'bank_of_russia_unit'}
WEIGHTS = (7, 1, 3) * 7 + (7, 1)


def source_registry():
    """Return a copy only when the installed registry has its frozen bytes."""
    data = REGISTRY_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != REGISTRY_SHA256:
        raise ValueError('bank_key_source_registry_sha_mismatch')
    registry = json.loads(data)
    if (registry.get('algorithm_id') != ALGORITHM_ID
            or registry.get('weights') != list(WEIGHTS)
            or tuple(s['id'] for s in registry.get('sources', [])) != SOURCE_IDS):
        raise ValueError('bank_key_source_registry_contract_mismatch')
    return deepcopy(registry)


def validate_bank_key(account, bik, *, location=None):
    """Check literal ASCII20/BIC9 under one explicitly supplied opening location.

    ``location`` is ``credit_institution`` or ``bank_of_russia_unit``. It is a
    prerequisite supplied by a separate source-grounded context, not a verdict
    this arithmetic function can infer. Missing/unknown location abstains.
    """
    def result(status, reason=None):
        reasons = [] if reason is None else [reason]
        return {'validation_status': status, 'validation_reasons': reasons,
                'validator_id': ALGORITHM_ID, 'algorithm_id': ALGORITHM_ID,
                'validation_scope': SCOPE, 'source_ids': list(SOURCE_IDS),
                'source_registry_sha256': REGISTRY_SHA256,
                'transform_ids': [], 'normalization_performed': False,
                'automatic_acceptance_authority': False,
                'production_qualified': False,
                'checks': [{'check_id': ALGORITHM_ID, 'status': status,
                            'scope': SCOPE, 'reason_codes': reasons,
                            'source_ids': list(SOURCE_IDS)}]}
    try:
        source_registry()
    except (OSError, ValueError, TypeError, KeyError):
        return result('not_checked', 'bank_key_source_registry_unavailable_or_tampered')
    if not isinstance(account, str) or not isinstance(bik, str):
        return result('fail', 'bank_key_requires_literal_strings')
    if re.fullmatch(r'[0-9]{5}[АВСЕНКМРТХABCEHKMPTX][0-9]{14}', account):
        # Clause 8 and published example 4 describe a real historical family.
        # Its conversion is deliberately outside this ASCII20 contract.
        return result('not_checked', 'clearing_currency_literal_not_supported')
    if not re.fullmatch(r'[0-9]{20}', account):
        return result('fail', 'bank_account_requires_20_ascii_digits_without_normalization')
    if not re.fullmatch(r'[0-9]{9}', bik):
        return result('fail', 'bank_bik_requires_9_ascii_digits_without_normalization')
    if not any(c != '0' for c in account) or not any(c != '0' for c in bik):
        return result('fail', 'degenerate_account_or_bik')
    if account.startswith('0'):
        # A conservative scope exclusion, never a claim of account kind based
        # on length/prefix. Treasury identifiers must not receive a bank key.
        return result('not_checked', 'zero_prefixed_account_outside_registered_bank_scope')
    if location is None:
        return result('not_checked', 'account_opening_location_not_established')
    if not isinstance(location, str) or location not in LOCATIONS:
        return result('not_checked', 'unsupported_account_opening_location')
    prefix = bik[-3:] if location == 'credit_institution' else '0' + bik[4:6]
    # Sum of the products modulo 10 equals the sum of their low digits modulo
    # 10. The existing key stays in the input; no candidate key is generated.
    matches = sum(int(digit) * weight for digit, weight in zip(prefix + account, WEIGHTS)) % 10 == 0
    return result('pass') if matches else result('fail', 'bank_account_bik_checksum_mismatch')
