import json
from pathlib import Path
import tempfile
import unittest
from deploy.paper_assets import PUBLICATIONS, verify_assets


class PaperReleaseTests(unittest.TestCase):
    def test_three_distinct_main_papers_and_one_supplement_are_required(self):
        self.assertEqual(sum(row[1] == 'main' for row in PUBLICATIONS), 3)
        self.assertEqual(len({row[3] for row in PUBLICATIONS}), 4)

    def test_incomplete_paper_release_is_rejected_before_pdf_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'papers-manifest.json').write_text(json.dumps({
                'schema': 1, 'kind': 'research_manuscript_pdfs', 'main_papers': 3, 'papers': []}))
            with self.assertRaisesRegex(ValueError, 'all three'):
                verify_assets(root)


if __name__ == '__main__':
    unittest.main()
