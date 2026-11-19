"""Prepare and verify the three existing manuscripts without recompiling them."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PUBLICATIONS = (
    ('forecast-original', 'main', 'papers/20261118_sci_cn/output/main.pdf', 'forecast-original-cn.pdf'),
    ('forecast-round2', 'main', 'papers/20261118_entertainment_computing_cn/output/main.pdf', 'forecast-entertainment-computing-cn.pdf'),
    ('game-data-analysis', 'main', 'papers/20261118_game_data_analysis_cn/output/main.pdf', 'game-data-analysis-cn.pdf'),
    ('forecast-round2-supplement', 'supplement', 'papers/20261118_entertainment_computing_cn/output/supplement.pdf', 'forecast-supplement-cn.pdf'),
)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pdf_details(path):
    path = Path(path)
    with path.open('rb') as source:
        if source.read(5) != b'%PDF-':
            raise ValueError('Not a PDF: ' + path.name)
    result = subprocess.run(['pdfinfo', str(path)], check=True, capture_output=True,
                            text=True, env=os.environ | {'LC_ALL': 'C'})
    fields = dict(line.split(':', 1) for line in result.stdout.splitlines() if ':' in line)
    pages = int(fields.get('Pages', '0'))
    if pages < 1 or fields.get('Encrypted', '').strip().startswith('yes'):
        raise ValueError('Unreadable or encrypted PDF: ' + path.name)
    first_page = subprocess.run(['pdftotext', '-f', '1', '-l', '1', str(path), '-'],
                                check=True, capture_output=True, text=True).stdout
    if fields.get('Author', '').strip() != 'Tano' or 'Tano' not in first_page:
        raise ValueError('PDF must visibly credit Tano and include author metadata: ' + path.name)
    return {'pages': pages, 'title': fields.get('Title', '').strip(),
            'author_metadata': fields.get('Author', '').strip(), 'bytes': path.stat().st_size,
            'sha256': digest(path)}


def build_assets(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Paper release assets cannot be overwritten')
    records = []
    for paper_id, role, relative, filename in PUBLICATIONS:
        records.append({'paper_id': paper_id, 'role': role, 'source': relative,
                        'asset': filename, **pdf_details(repo / relative)})
    output.mkdir(parents=True)
    for row in records:
        shutil.copyfile(repo / row['source'], output / row['asset'])
    manifest = {'schema': 1, 'kind': 'research_manuscript_pdfs', 'main_papers': 3,
                'papers': records, 'scope': 'Existing Chinese manuscripts, copied byte-for-byte; no new experiments or claims.'}
    (output / 'papers-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    files = [row['asset'] for row in records] + ['papers-manifest.json']
    (output / 'SHA256SUMS').write_text(''.join(digest(output / name) + '  ' + name + '\n' for name in files))
    return verify_assets(output)


def verify_assets(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'papers-manifest.json').read_text())
    expected = {item[3] for item in PUBLICATIONS}
    rows = manifest.get('papers', [])
    if (manifest.get('schema') != 1 or manifest.get('kind') != 'research_manuscript_pdfs'
            or manifest.get('main_papers') != 3 or len(rows) != 4
            or {row['asset'] for row in rows} != expected
            or {row['paper_id'] for row in rows} != {item[0] for item in PUBLICATIONS}
            or sum(row.get('role') == 'main' for row in rows) != 3):
        raise ValueError('Release requires all three manuscripts and the existing supplement')
    expected |= {'papers-manifest.json'}
    lines = (root / 'SHA256SUMS').read_text().splitlines()
    checksums = {}
    for line in lines:
        value, name = line.split('  ', 1)
        if name not in expected or name in checksums or len(value) != 64:
            raise ValueError('Unexpected paper checksum member')
        checksums[name] = value
    if set(checksums) != expected:
        raise ValueError('Missing paper checksums')
    for name, value in checksums.items():
        path = root / name
        if path.is_symlink() or digest(path) != value:
            raise ValueError('Paper asset changed: ' + name)
    for row in rows:
        actual = pdf_details(root / row['asset'])
        if any(actual[key] != row[key] for key in actual):
            raise ValueError('PDF metadata differs: ' + row['asset'])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    result = build_assets(args.repo_root, args.output) if args.command == 'build' else verify_assets(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
