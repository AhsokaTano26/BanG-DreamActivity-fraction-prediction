from pathlib import Path
import tempfile
import unittest
from deploy.paper_assets import PUBLICATIONS, build_assets, sync_repository_pdfs, verify_assets


def write_pdf(path, title):
    """Small valid PDF fixture; metadata and visible author are independently encoded."""
    content = b'BT /F1 12 Tf 20 100 Td (Tano) Tj ET'
    title_hex = (b'\xfe\xff' + title.encode('utf-16-be')).hex().encode()
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 150] '
        b'/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length ' + str(len(content)).encode() + b' >>\nstream\n' + content + b'\nendstream',
        b'<< /Title <' + title_hex + b'> /Author (Tano) >>',
    ]
    result = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(str(index).encode() + b' 0 obj\n' + obj + b'\nendobj\n')
    start = len(result)
    result.extend(b'xref\n0 7\n0000000000 65535 f \n')
    for offset in offsets[1:]:
        result.extend(f'{offset:010d} 00000 n \n'.encode())
    result.extend(b'trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n')
    result.extend(str(start).encode() + b'\n%%EOF\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result)


def prepare_sources(repo):
    for _, _, source, filename in PUBLICATIONS:
        write_pdf(repo / source, filename[:-4])


class PaperReleaseTests(unittest.TestCase):
    def test_three_distinct_main_papers_and_one_supplement_are_required(self):
        self.assertEqual(sum(row[1] == 'main' for row in PUBLICATIONS), 3)
        self.assertEqual(len({row[3] for row in PUBLICATIONS}), 4)

    def test_incomplete_paper_release_is_rejected_before_pdf_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'all three'):
                verify_assets(root)

    def test_build_contains_only_four_pdfs_and_keeps_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            prepare_sources(repo)
            output = Path(tmp) / 'release'
            build_assets(repo, output)
            self.assertEqual(len(list(output.iterdir())), 4)
            self.assertTrue(all(path.suffix == '.pdf' for path in output.iterdir()))
            for _, _, source, filename in PUBLICATIONS:
                self.assertEqual((repo / source).read_bytes(), (output / filename).read_bytes())

    def test_source_file_in_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            prepare_sources(repo)
            output = Path(tmp) / 'release'
            build_assets(repo, output)
            (output / 'main.tex').write_text('private local manuscript')
            with self.assertRaisesRegex(ValueError, 'only'):
                verify_assets(output)

    def test_modified_pdf_is_rejected_against_checked_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            prepare_sources(repo)
            reference = Path(tmp) / 'reference'
            output = Path(tmp) / 'release'
            build_assets(repo, reference)
            build_assets(repo, output)
            pdf = output / PUBLICATIONS[0][3]
            with pdf.open('ab') as stream:
                stream.write(b'\n% changed after review\n')
            with self.assertRaisesRegex(ValueError, 'reference'):
                verify_assets(output, reference=reference)

    def test_sync_provides_a_pdf_only_repository_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            prepare_sources(repo)
            sync_repository_pdfs(repo)
            target = repo / 'papers/published'
            self.assertEqual(len(list(target.iterdir())), 4)
            self.assertTrue(all(path.suffix == '.pdf' for path in target.iterdir()))


if __name__ == '__main__':
    unittest.main()
