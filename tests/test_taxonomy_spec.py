"""Tests for the versioned taxonomy specification (v1.0)."""

from __future__ import annotations

import re

import pytest

from loato_bench.data.taxonomy_spec import (
    CATEGORY_ID_TO_SLUG,
    LOATO_CATEGORIES,
    OLD_SLUG_TO_NEW,
    SLUG_TO_CATEGORY_ID,
    TAXONOMY_V1,
    TAXONOMY_VERSION,
    VALID_SLUGS,
    CategorySpec,
    get_category_by_id,
    get_category_by_slug,
    validate_slug,
)

# ---------------------------------------------------------------------------
# TestTaxonomyV1 — structural integrity
# ---------------------------------------------------------------------------


class TestTaxonomyV1:
    """Validate the top-level TAXONOMY_V1 registry."""

    def test_has_seven_categories(self) -> None:
        assert len(TAXONOMY_V1) == 7

    def test_ids_are_c_prefixed(self) -> None:
        for cid in TAXONOMY_V1:
            assert re.fullmatch(r"C[1-7]", cid), f"Bad ID: {cid}"

    def test_slugs_are_snake_case(self) -> None:
        for spec in TAXONOMY_V1.values():
            assert re.fullmatch(r"[a-z][a-z0-9_]*", spec.slug), f"Bad slug: {spec.slug}"

    def test_no_duplicate_slugs(self) -> None:
        slugs = [spec.slug for spec in TAXONOMY_V1.values()]
        assert len(slugs) == len(set(slugs))

    def test_no_duplicate_ids(self) -> None:
        ids = [spec.id for spec in TAXONOMY_V1.values()]
        assert len(ids) == len(set(ids))

    def test_all_fields_present(self) -> None:
        for cid, spec in TAXONOMY_V1.items():
            assert spec.id == cid
            assert spec.name
            assert spec.slug
            assert spec.mechanism
            assert isinstance(spec.signal_phrases, tuple)
            assert isinstance(spec.exclusions, tuple)
            assert isinstance(spec.examples_positive, tuple)
            assert isinstance(spec.examples_negative, tuple)
            assert isinstance(spec.loato_eligible, bool)

    def test_version_string(self) -> None:
        assert TAXONOMY_VERSION == "1.0"

    def test_category_spec_is_frozen(self) -> None:
        spec = TAXONOMY_V1["C1"]
        with pytest.raises(AttributeError):
            spec.slug = "something_else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLoatoCategories
# ---------------------------------------------------------------------------


class TestLoatoCategories:
    """Validate LOATO-eligible category list."""

    def test_has_five_loato_categories(self) -> None:
        """C1–C5 eligible, C6 below threshold (2A-04), C7 excluded."""
        assert len(LOATO_CATEGORIES) == 5

    def test_c6_c7_excluded(self) -> None:
        """C6 (context_manipulation) and C7 (other) are not LOATO-eligible."""
        assert TAXONOMY_V1["C6"].slug not in LOATO_CATEGORIES
        assert TAXONOMY_V1["C7"].slug not in LOATO_CATEGORIES

    def test_all_loato_slugs_valid(self) -> None:
        for slug in LOATO_CATEGORIES:
            assert slug in VALID_SLUGS


# ---------------------------------------------------------------------------
# TestMigrationMap
# ---------------------------------------------------------------------------


class TestMigrationMap:
    """Validate OLD_SLUG_TO_NEW bridge mapping."""

    OLD_SLUGS = {
        "instruction_override",
        "jailbreak_roleplay",
        "obfuscation_encoding",
        "context_manipulation",
        "payload_splitting",
        "information_extraction",
        "indirect_injection",
        "social_engineering",
    }

    def test_all_eight_old_slugs_mapped(self) -> None:
        assert set(OLD_SLUG_TO_NEW.keys()) == self.OLD_SLUGS

    def test_all_targets_are_valid_slugs(self) -> None:
        for target in OLD_SLUG_TO_NEW.values():
            assert target in VALID_SLUGS, f"Invalid target slug: {target}"

    def test_context_manipulation_remapped_to_information_extraction(self) -> None:
        assert OLD_SLUG_TO_NEW["context_manipulation"] == "information_extraction"

    def test_indirect_injection_remapped_to_context_manipulation(self) -> None:
        assert OLD_SLUG_TO_NEW["indirect_injection"] == "context_manipulation"

    def test_payload_splitting_remapped_to_other(self) -> None:
        assert OLD_SLUG_TO_NEW["payload_splitting"] == "other"

    def test_identity_mappings(self) -> None:
        identity = {
            "instruction_override",
            "jailbreak_roleplay",
            "obfuscation_encoding",
            "social_engineering",
            "information_extraction",
        }
        for slug in identity:
            assert OLD_SLUG_TO_NEW[slug] == slug


# ---------------------------------------------------------------------------
# TestLookupFunctions
# ---------------------------------------------------------------------------


class TestLookupFunctions:
    """Validate get_category_by_slug, get_category_by_id, validate_slug."""

    def test_get_by_slug_returns_correct_spec(self) -> None:
        spec = get_category_by_slug("instruction_override")
        assert spec.id == "C1"
        assert isinstance(spec, CategorySpec)

    def test_get_by_slug_raises_on_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown slug"):
            get_category_by_slug("nonexistent")

    def test_get_by_id_returns_correct_spec(self) -> None:
        spec = get_category_by_id("C2")
        assert spec.slug == "jailbreak_roleplay"

    def test_get_by_id_raises_on_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown category ID"):
            get_category_by_id("C99")

    def test_validate_slug_true(self) -> None:
        assert validate_slug("instruction_override") is True

    def test_validate_slug_false(self) -> None:
        assert validate_slug("made_up") is False


# ---------------------------------------------------------------------------
# TestDerivedConstants
# ---------------------------------------------------------------------------


class TestDerivedConstants:
    """Validate VALID_SLUGS and bidirectional mappings."""

    def test_valid_slugs_is_frozenset(self) -> None:
        assert isinstance(VALID_SLUGS, frozenset)

    def test_valid_slugs_has_seven(self) -> None:
        assert len(VALID_SLUGS) == 7

    def test_bidirectional_mapping_consistency(self) -> None:
        for cid, slug in CATEGORY_ID_TO_SLUG.items():
            assert SLUG_TO_CATEGORY_ID[slug] == cid

    def test_all_ids_in_id_to_slug(self) -> None:
        assert set(CATEGORY_ID_TO_SLUG.keys()) == set(TAXONOMY_V1.keys())

    def test_all_slugs_in_slug_to_id(self) -> None:
        assert set(SLUG_TO_CATEGORY_ID.keys()) == VALID_SLUGS


# ---------------------------------------------------------------------------
# TestExamplesPresent
# ---------------------------------------------------------------------------


class TestExamplesPresent:
    """Ensure C1--C6 have sufficient examples for LLM prompt building."""

    @pytest.mark.parametrize("cid", ["C1", "C2", "C3", "C4", "C5", "C6"])
    def test_at_least_two_positive_examples(self, cid: str) -> None:
        spec = TAXONOMY_V1[cid]
        assert len(spec.examples_positive) >= 2, f"{cid} needs >=2 positive examples"

    @pytest.mark.parametrize("cid", ["C1", "C2", "C3", "C4", "C5", "C6"])
    def test_at_least_one_negative_example(self, cid: str) -> None:
        spec = TAXONOMY_V1[cid]
        assert len(spec.examples_negative) >= 1, f"{cid} needs >=1 negative example"
