import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ytrag.frontend import format_result


def test_format_result_adds_large_readable_text():
    result = format_result(
        answer="This is a sample answer.",
        sources=[{"source": "sample.pdf", "score": 0.88}],
        confidence=0.88,
    )

    assert "This is a sample answer." in result["answer"]
    assert result["source_count"] == 1
    assert result["font_scale"] >= 1.4
