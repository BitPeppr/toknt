"""Canonical model identity across providers and harnesses."""

from __future__ import annotations

from toknt.analytics.model_names import (
    CanonicalName,
    canonical_model,
    clean_label,
    group_label,
)


def identity(name: str) -> str:
    return canonical_model(name).identity


class TestPrefixes:
    def test_vendor_prefix_is_dropped(self) -> None:
        assert identity("z-ai/glm-5.3-flash") == identity("glm-5.3-flash")

    def test_harness_prefix_is_dropped(self) -> None:
        assert identity("cline-free/muse-spark-1.3") == identity("muse-spark-1.3")

    def test_provider_family_prefix_is_dropped(self) -> None:
        assert identity("deepseek-ai/deepseek-v4.1-flash") == identity(
            "deepseek-v4.1-flash"
        )

    def test_only_the_first_segment_is_dropped(self) -> None:
        assert identity("a/b") == identity("b")


class TestDecorations:
    def test_free_tier_suffix_is_dropped(self) -> None:
        assert identity("glm-5.3-flash-free") == identity("glm-5.3-flash")

    def test_vendor_tier_marker_is_dropped(self) -> None:
        assert identity("muse-spark-1.3-contributor") == identity(
            "muse-spark-1.3-contributor-free"
        )

    def test_routing_tag_is_dropped(self) -> None:
        assert identity("thinkingmachines/inkling:free") == identity("inkling")

    def test_quantization_suffix_is_dropped(self) -> None:
        assert identity("llama-3-70b-q4_k_m") == identity("llama-3-70b")
        assert identity("llama-3-70b-fp8") == identity("llama-3-70b")

    def test_decoration_inside_a_word_is_kept(self) -> None:
        assert identity("freeway-2") == identity("freeway-2")

    def test_underscores_and_spaces_are_normalized(self) -> None:
        assert identity("glm_5_3_flash") == identity("glm-5.3-flash")
        assert identity("GLM 5.3 Flash") == identity("glm-5.3-flash")


class TestKeepsDifferentModelsApart:
    def test_size_tier_is_part_of_the_identity(self) -> None:
        assert identity("glm-5.3-flash") != identity("glm-5.3")

    def test_version_is_part_of_the_identity(self) -> None:
        assert identity("muse-spark-1.3") != identity("muse-spark-1.2")

    def test_capability_marker_is_part_of_the_identity(self) -> None:
        assert identity("nemotron-3.5-super-vl") != identity("nemotron-3.5-super")

    def test_release_stage_is_part_of_the_identity(self) -> None:
        assert identity("gemini-3-flash-preview") != identity("gemini-3-flash")

    def test_model_build_suffix_is_kept(self) -> None:
        assert identity("nemotron-3-ultra-550b-a55b") != identity("nemotron-3-ultra")


class TestNormalisation:
    def test_digit_boundaries_do_not_matter(self) -> None:
        assert identity("gpt4o") == identity("gpt-4o")
        assert identity("llama3-70b") == identity("llama-3-70b")

    def test_version_dots_and_dashes_agree(self) -> None:
        assert identity("glm-5.3-flash") == identity("glm-5-3-flash")

    def test_vendor_alias_maps_to_the_family(self) -> None:
        assert identity("sglm-5.3-flash") == identity("glm-5.3-flash")


class TestLabels:
    def test_label_keeps_the_readable_spelling(self) -> None:
        assert clean_label("z-ai/glm-5.3-flash") == "glm-5.3-flash"
        assert clean_label("muse-spark-1.3-contributor-free") == "muse-spark-1.3"
        assert clean_label("dots-studio/dots-3-note-preview:free") == (
            "dots-3-note-preview"
        )

    def test_label_keeps_multi_part_versions_readable(self) -> None:
        assert clean_label("nemotron-3.5-super-vl-free") == "nemotron-3.5-super-vl"

    def test_empty_name_is_harmless(self) -> None:
        assert clean_label("") == ""
        assert canonical_model("").identity == ""


class TestGroupLabel:
    def test_most_common_spelling_wins(self) -> None:
        assert group_label(["glm-5.3-flash", "glm-5.3-flash", "glm-5-3-flash"]) == (
            "glm-5.3-flash"
        )

    def test_ties_prefer_the_shorter_label(self) -> None:
        assert group_label(["aaa", "bbbb"]) == "aaa"

    def test_empty_group(self) -> None:
        assert group_label([]) == ""


class TestRealWorldNames:
    """The names a live corpus actually contains."""

    def test_muse_spark_across_three_sources(self) -> None:
        names = [
            "cline-free/muse-spark-1.3-contributor",
            "muse-spark-1.3-contributor-free",
            "pi/muse-spark-1.3-contributor-free",
        ]
        assert len({identity(name) for name in names}) == 1

    def test_glm_flash_merges_but_plain_glm_does_not(self) -> None:
        merged = {
            identity(name)
            for name in ("z-ai/glm-5.3-flash", "glm-5.3-flash-free", "cline/glm-5.3-flash")
        }
        assert len(merged) == 1
        assert identity("z-ai/glm-5.3") not in merged

    def test_canonical_name_dataclass(self) -> None:
        name = canonical_model("z-ai/glm-5.3-flash")
        assert isinstance(name, CanonicalName)
        assert name.label == "glm-5.3-flash"
        assert name.identity == "glm-5-3-flash"
