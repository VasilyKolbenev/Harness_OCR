import copy
import unittest

from harness_ocr.profiles import (ProfileError, ProfileRegistry, UnknownProfile,
                                  UnsupportedProfileVersion, assess_completion, compile_profile)


def declaration():
    return {"document_id": "DXX", "profile_id": "test_v1", "profile_version": "1.0.0", "name": "Тест",
            "fields": [
                {"key": "holder.id", "label": "Номер", "value_type": "identifier", "role": "holder",
                 "critical": True, "requirement": "required_for_profile_completion"},
                {"key": "holder.extra", "label": "Дополнительный номер", "value_type": "identifier", "role": "holder",
                 "critical": True, "requirement": "conditional", "condition": "Только если напечатан в данной версии"}]}


class ProfileCompilerTests(unittest.TestCase):
    def test_ten_profiles_96_fields_and_no_quality_claim(self):
        registry = ProfileRegistry()
        self.assertEqual(len(registry.list()), 10)
        count = 0
        for item in registry.list():
            profile = registry.compile(item["profile_id"])
            count += len(profile["field_definitions"])
            self.assertEqual(profile["qualification_status"], "not_qualified")
            self.assertFalse(profile["automatic_acceptance_authorized"])
            self.assertFalse(profile["candidate_contract"]["additionalProperties"])
        self.assertEqual(count, 96)

    def test_unknown_condition_cannot_disappear_with_missing_ocr(self):
        profile = compile_profile(declaration())
        fields = {"holder.id": {"raw_value": "001", "status": "supported", "party_role": "holder"}}
        result = assess_completion(profile, fields)
        self.assertFalse(result["contract_complete"])
        self.assertEqual(result["unknown_conditionals"], ["holder.extra"])
        explicit = compile_profile(declaration(), condition_context={"holder.extra": False})
        self.assertTrue(assess_completion(explicit, fields)["contract_complete"])
        fields["holder.extra"] = {"raw_value": "123", "value": "123"}
        self.assertFalse(assess_completion(explicit, fields)["contract_complete"])

    def test_true_conditional_required_and_wrong_role_blocks_completion(self):
        profile = compile_profile(declaration(), condition_context={"holder.extra": True})
        self.assertIn("holder.extra", profile["required_fields"])
        fields = {key: {"raw_value": "001", "status": "supported", "party_role": "issuer"} for key in profile["required_fields"]}
        result = assess_completion(profile, fields)
        self.assertFalse(result["contract_complete"])
        self.assertEqual(set(result["wrong_role_fields"]), set(profile["required_fields"]))

    def test_subset_bound_to_version_not_full_profile(self):
        full = compile_profile(declaration())
        subset = compile_profile(declaration(), subset=["holder.id"])
        self.assertNotEqual(full["subset_contract_id"], subset["subset_contract_id"])
        self.assertFalse(subset["is_full_profile"])
        self.assertEqual(subset["unknown_conditionals"], [])
        self.assertEqual(subset["json_schema"]["properties"]["fields"]["required"], ["holder.id"])
        with self.assertRaises(ProfileError):
            compile_profile(declaration(), subset=["unknown"])
        with self.assertRaises(ProfileError):
            compile_profile(declaration(), subset=[])

    def test_declaration_change_changes_config_hash_and_caller_cannot_bypass_required(self):
        source = declaration()
        first = compile_profile(source)
        self.assertEqual(first["config_sha256"], compile_profile(source)["config_sha256"])
        source["fields"][0]["role"] = "issuer"
        second = compile_profile(source)
        self.assertNotEqual(first["config_sha256"], second["config_sha256"])
        with self.assertRaises(ProfileError):
            compile_profile(declaration(), condition_context={"holder.id": False})
        with self.assertRaises(ProfileError):
            compile_profile(declaration(), condition_context=[])
        duplicate = declaration();duplicate["fields"].append(copy.deepcopy(duplicate["fields"][0]))
        with self.assertRaises(ProfileError):
            compile_profile(duplicate)

    def test_registry_unknown_or_ambiguous_version_fails_closed(self):
        one = declaration();two = declaration();two["profile_version"] = "2.0.0"
        registry = ProfileRegistry([one, two])
        with self.assertRaises(UnsupportedProfileVersion):
            registry.get("DXX")
        self.assertEqual(registry.compile("DXX", version="2.0.0")["profile_version"], "2.0.0")
        with self.assertRaises(UnsupportedProfileVersion):
            registry.get("DXX", version="9")
        with self.assertRaises(UnknownProfile):
            registry.get("nearest_profile_guess")

    def test_projection_roles_and_literal_contract_remain_aligned(self):
        profile = compile_profile(declaration(), subset=["holder.id"])
        self.assertEqual(profile["validator_configuration"]["holder.id"]["role"], "holder")
        self.assertEqual(profile["review_projection"][0]["role"], "holder")
        self.assertEqual(profile["candidate_contract"]["properties"]["proposals"]["items"]["properties"]["party_role"]["enum"], ["holder"])
        proposal_keys = profile["candidate_contract"]["properties"]["proposals"]["items"]["properties"]
        self.assertNotIn("raw_value", proposal_keys)
        self.assertIn("candidate_ids", proposal_keys)
        self.assertTrue(profile["mcp_projection"]["caller_cannot_enable_autoaccept"])


if __name__ == "__main__":
    unittest.main()
