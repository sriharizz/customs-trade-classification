import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import (
    _grade_chapter,
    _grade_heading,
    _grade_subheading,
    _grade_duty,
    is_sanctioned,
    get_duty_rate,
    _hs_lookup_results,
)


class TestGradeChapter:
    def test_exact_match(self):
        assert _grade_chapter("39", "39") == 1.0

    def test_same_section_partial(self):
        # Both in 30-39 range — same first digit
        assert _grade_chapter("38", "39") == 0.4

    def test_completely_wrong(self):
        assert _grade_chapter("85", "39") == 0.0

    def test_leading_zero(self):
        assert _grade_chapter("09", "09") == 1.0


class TestGradeHeading:
    def test_exact_match(self):
        assert _grade_heading("3903", "3903", "39") == 1.0

    def test_chapter_prefix_match(self):
        assert _grade_heading("3905", "3903", "39") == 0.4

    def test_wrong_chapter(self):
        assert _grade_heading("8501", "3903", "39") == 0.0


class TestGradeSubheading:
    def test_exact_match(self):
        assert _grade_subheading("3903.20.00.00", "3903.20.00.00") == 1.0

    def test_partial_match_high(self):
        score = _grade_subheading("3903.20.00", "3903.20.00.00")
        assert 0.5 < score < 1.0

    def test_partial_match_low(self):
        score = _grade_subheading("3903.30.00.00", "3903.20.00.00")
        assert 0.0 < score < 1.0

    def test_wrong(self):
        score = _grade_subheading("8501.10.20.00", "3903.20.00.00")
        assert score == 0.0

    def test_caps_at_0_95(self):
        # Near-exact but not exact should cap at 0.95
        score = _grade_subheading("3903.20.00.0", "3903.20.00.00")
        assert score <= 0.95


class TestGradeDuty:
    def test_exact_free(self):
        assert _grade_duty("Free", "Free") == 1.0

    def test_case_insensitive(self):
        assert _grade_duty("free", "Free") == 1.0

    def test_numeric_exact(self):
        assert _grade_duty("3.5%", "3.5%") == 1.0

    def test_numeric_close(self):
        score = _grade_duty("3.5%", "3.7%")
        assert score >= 0.3

    def test_numeric_far(self):
        score = _grade_duty("10%", "Free")
        assert score == 0.0

    def test_substring_match(self):
        score = _grade_duty("3.5", "3.5%")
        assert score >= 0.7


class TestIsSanctioned:
    def test_russia_sanctioned(self):
        assert is_sanctioned("Russia") is True

    def test_iran_sanctioned(self):
        assert is_sanctioned("Iran") is True

    def test_north_korea_sanctioned(self):
        assert is_sanctioned("North Korea") is True

    def test_syria_sanctioned(self):
        assert is_sanctioned("Syria") is True

    def test_germany_clear(self):
        assert is_sanctioned("Germany") is False

    def test_japan_clear(self):
        assert is_sanctioned("Japan") is False

    def test_case_insensitive(self):
        assert is_sanctioned("IRAN") is True
        assert is_sanctioned("iran") is True


class TestHtsLookup:
    def test_returns_results_for_valid_prefix(self):
        results = _hs_lookup_results("39")
        assert len(results.splitlines()) > 0
        assert "39" in results

    def test_returns_not_found_for_invalid(self):
        results = _hs_lookup_results("99999")
        assert "No HTS entries found" in results

    def test_heading_prefix(self):
        results = _hs_lookup_results("3903")
        assert "3903" in results


class TestGetDutyRate:
    def test_known_subheading(self):
        # All 15 shipments are covered — spot check a few
        rate = get_duty_rate("3903.20.00.00")
        assert isinstance(rate, str)
        assert len(rate) > 0

    def test_fallback_to_heading(self):
        rate = get_duty_rate("3903.99.99.99")
        assert isinstance(rate, str)
