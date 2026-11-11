# Native MChD adapter

The server registers `fns-emchd-003-v1` only for D04
(`power_of_attorney_evidence_v1`). It validates the complete XML against the
original official unified003 schema `ON_EMCHD_1_928_00_01_01_01.xsd`, not a
locally approximated schema or selected field checks. The schema is a standalone
file with no imports/includes; its SHA256 is hardcoded in the adapter and checked
on every decode/replay. The packaged registry records the original archive hash,
official publication chain and normative mapping reference.

The [FNS official electronic signature page](https://www.nalog.gov.ru/rn77/related_activities/el_doc/use_electronic_sign/)
identifies unified003 as EMCHD_1. Its referenced
[Gosuslugi publication](https://partners.gosuslugi.ru/catalog/attorney) lists the
[original XSD archive](https://gu-st.ru/content/partners/ON_EMCHD_1_928_00_01_01_01.zip).
The [public Gosuslugi metadata](https://www.gosuslugi.ru/api/cms/v2/partners/products/attorney)
provides the actual attachment URL. Retrieved on2026-11-11.

## Explicit supported scope

The allowlist contains exactly `EMCHD_1`, namespace
`urn://x-artefacts/EMCHD_1`, root `Доверенность`, version attribute `ВерсФорм`.
This release maps ordinary non-notarial B2B powers: `ПрЭлФорм="00000000"`,
`Документ/Довер`, `СвДов/@ВидДовер="1"`. Other versions, element-flag variants,
delegations and notarial variants are rejected explicitly. Full XSD validation
precedes the variant and semantic mapping checks. An input cannot extend the
registry through `schemaLocation` or an alternate schema.

Within this scope all four principal types and five representative types have
explicit structural mappings. The original XSD represents some alternatives as
optional sequences; the adapter additionally checks the discriminator and its
one permitted branch against official format tables4.4/4.5 and4.20/4.21 in
[ON_DOVBB.docx](https://gu-st.ru/content/partners/ON_DOVBB.docx). Thus an XML file
can pass XSD and still be rejected for a wrong party type. An organization's
executive/signatory is never relabelled as the representative or principal.

## Exact evidence and field mapping

All paths are absolute namespace-bound XPaths with one-based same-QName sibling
indices. Attributes are unqualified as specified by the official XSD. Values
remain exact parsed XML-infoset strings, including leading/trailing whitespace
in string values. Input bytes and their SHA256 preserve lexical escaping,
encoding, XML-mandated attribute whitespace normalization and line endings.
No invented label strings, page numbers or coordinates are attached to XML.

| D04 field | Registered source |
| --- | --- |
| `principal.identity` | `СвДоверит/Доверит` branch specified by `ТипДоверит`; organization `НаимОрг`/`НаимИО` or personal `СведФЛ/ФИО` attributes in explicit Фамилия, Имя, Отчество order |
| `representative.identity` | `СвУпПред/Пред` branch specified by `ТипПред`; same explicit organization or personal name components |
| `principal.identifier`, `representative.identifier` | Direct party entity attributes `ИННЮЛ`, `ИННФЛ`, `ОГРН`, `ОГРНИП`, `СНИЛС`, `КПП`; personal `СведФЛ/УдЛичнФЛ/@СерНомДок` is a Russian passport only when its same element explicitly has `КодВидДок="21"` |
| `power.issue_date` | `СвДов/@ДатаВыдДовер` |
| `power.valid_until` | `СвДов/@СрокДейст` |
| `power.number` | `СвДов/@НомДовер`, unified registration number; `ВнНомДовер` remains separately named internal-number data |
| `power.authority_literal` | Entire exact `СвПолн/ТекстПолн/text()` only for `ТипПолн="0"` |

Personal-name assembly joins individually anchored original attributes with one
explicit separator space. Every component retains its own original observation
and locator; no whole-name source observation is fabricated. Typed identifiers
use `typed-fields-v1`, `party-identifiers-v1` with individual kind, role, version
and evidence spans. The passport's type21 observation is included as evidence.

All principals and representatives remain ordered structured lists in
`native_model`. Multiple parties are legitimate input. They produce all
per-party candidates, explicit review reasons and no scalar identity/identifier
claims for the affected role; no first-party fallback exists. Machine-coded
authorities (`ТипПолн="1"`) retain every code, title and restriction in a
structured list. A title alone never replaces the full authority. Because the
current D04 scalar literal contract cannot express that list, it yields review
instead of a partial authority claim.

## Runtime trust boundary

`decode_mchd(bytes_or_path)` returns deterministic `observations`,
`field_candidates`, `native_claims`, `native_model`, `review_reasons` and
`native_validation`. The exact discriminator is `native_adapter=
"fns-emchd-003-v1"`, bridge `mchd_xml_xsd_validated`. Structural status is `pass`.
The parent server replays child-process results against its own original source
using `verify_native_result(result, source)`. Core observation validation uses
`verify_native_observation(observation, source)`; source/schema hashes claimed
by a caller alone do not establish provenance. Only state-managed `active` and
`created_revision` are ignored in exact observation replay. The typed verifier
then checks exact registered locator, kind, role, party and source association.

Parsing disables DTD loading, entity resolution, XInclude processing, external
schema fetching, network access, recovery and huge trees. A deny-all resource
resolver supplies an additional boundary. DTDs, entity declarations, processing
instructions, xsi schema/type attributes, embedded XMLDSig and envelopes are
rejected. Standalone XML can be read without its detached signature; no signed
container, CAdES/ZIP, SOAP/XMLDSig envelope or detached-signature verification is
implemented. No signature is silently stripped or validated by schema checks.

`signature_verified` is always false. Signature validity, signer trust,
revocation status, current document validity and service authorization each
remain `not_verified`. XSD pass never supplies legal acceptance, authenticity,
current registry status or the sufficiency of a power. Generic XML remains the
separate diagnostic `generic_xml_unqualified` path and cannot inherit native
evidence or quality qualification.

## Validation and distribution

`tests/test_mchd.py` generates known-gold synthetic XML in code and validates it
against the original XSD. Tests cover attribute/text exactness, namespace and
version mismatches, unknown attributes, schema-valid party-role mismatch,
multiple parties, restricted machine authorities, entities/DTDs, dynamic
schemas, embedded signatures, envelopes, unsupported variants, tampered schema,
forged evidence and explicit D04 routing. These are structural/security adapter
tests, not measured OCR quality or cryptographic acceptance tests. No live real
power of attorney or signature corpus is included.

The XSD and provenance registry ship as package data and in source delivery;
the official DOCX/PDF and portal assets stay in ignored `validation/`. The
original normative government schema retains its own publication status and
attribution, described in `native/data/NOTICE.txt`; the project does not assign
its software license to that asset. Changes to format bytes, namespace,
variant scope or mapping require a new reviewed registry/adapter release.
