import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/research_master_comprehensive.ipynb")


class ResearchNotebookDeepAnalysisSectionsTest(unittest.TestCase):
    def _load_notebook(self):
        return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_has_new_deep_analysis_markdown_sections(self):
        nb = self._load_notebook()
        markdown_blob = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )

        required_sections = [
            "## 2.4 Dataset Provenance and Integrity",
            "## 2.5 Label Distribution and Data Quality Deep Dive",
            "## 2.6 Cross-Dataset Shift and Feature Engineering Analysis",
            "## 2.7 Mini Baseline Importance and Ablation",
            "## 2.8 Research Readiness Notes and Limitations",
        ]
        for section in required_sections:
            self.assertIn(section, markdown_blob)

    def test_has_deep_analysis_control_flags_and_outputs(self):
        nb = self._load_notebook()
        code_blob = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )

        required_tokens = [
            "RUN_DEEP_AUDIT",
            "RUN_DRIFT_ANALYSIS",
            "RUN_BASELINE_IMPORTANCE_ANALYSIS",
            "ANALYSIS_OUTPUT_DIR",
            "compute_psi",
            "drift_report.csv",
            "ablation_results.csv",
            "research_notes.md",
        ]
        for token in required_tokens:
            self.assertIn(token, code_blob)


if __name__ == "__main__":
    unittest.main()
