"""Prepare and verify the three existing manuscripts without recompiling them."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PUBLICATIONS = (
    ('forecast-original', 'main', 'papers/20261118_sci_cn/output/main.pdf', '非平稳音乐游戏活动排名终分预测的时间一致评测与结构化在线误差校正.pdf'),
    ('forecast-round2', 'main', 'papers/20261118_entertainment_computing_cn/output/main.pdf', '音乐游戏活动终分预测：延迟反馈下的跨排名共享与失效诊断.pdf'),
    ('game-data-analysis', 'main', 'papers/20261118_game_data_analysis_cn/output/main.pdf', 'BanG Dream国服活动排名分数的时序结构与活动条件关联.pdf'),
    ('forecast-round2-supplement', 'supplement', 'papers/20261118_entertainment_computing_cn/output/supplement.pdf', '补充材料：音乐游戏活动终分预测.pdf'),
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
        details = pdf_details(repo / relative)
        if filename != details['title'] + '.pdf':
            raise ValueError('PDF filename must match its title: ' + filename)
        records.append({'paper_id': paper_id, 'role': role, 'source': relative,
                        'asset': filename, **details})
    output.mkdir(parents=True)
    for row in records:
        shutil.copyfile(repo / row['source'], output / row['asset'])
    return verify_assets(output)


def verify_assets(directory, reference=None):
    """Accept exactly the reviewed PDFs; return audit metadata on stdout, not as assets."""
    root = Path(directory).resolve()
    expected = {item[3] for item in PUBLICATIONS}
    names = {path.name for path in root.iterdir()}
    if expected - names:
        raise ValueError('Release requires all three manuscripts and the existing supplement')
    if names != expected:
        raise ValueError('Paper releases may contain only the four title-named PDFs')
    rows = []
    for paper_id, role, source, filename in PUBLICATIONS:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError('Paper asset must be a regular PDF: ' + filename)
        actual = pdf_details(path)
        if filename != actual['title'] + '.pdf':
            raise ValueError('PDF filename must match its title: ' + filename)
        if reference is not None:
            approved = Path(reference) / filename
            if approved.is_symlink() or not approved.is_file() or digest(approved) != actual['sha256']:
                raise ValueError('PDF differs from the checked reference: ' + filename)
        rows.append({'paper_id': paper_id, 'role': role, 'source': source,
                     'asset': filename, **actual})
    return {'schema': 1, 'kind': 'research_manuscript_pdfs', 'main_papers': 3,
            'papers': rows, 'scope': 'Existing Chinese manuscripts, copied byte-for-byte; no new experiments or claims.'}


def sync_repository_pdfs(repo):
    """Refresh reviewable, title-named PDFs in the repository; never use Git or a network."""
    repo = Path(repo).resolve()
    target = repo / 'papers/published'
    with tempfile.TemporaryDirectory(prefix='bangdream-paper-titles-') as tmp:
        stage = Path(tmp) / 'assets'
        manifest = build_assets(repo, stage)
        target.mkdir(parents=True, exist_ok=True)
        # Validate all source PDFs before replacing any review copy.
        for source in sorted(stage.iterdir()):
            temporary = target / ('.' + source.name + '.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(target / source.name)
    verify_assets(target)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('directory', type=Path)
    verify.add_argument('--reference', type=Path, help='逐字节对照已审阅的PDF目录')
    sync = commands.add_parser('sync', help='同步标题命名PDF到本地papers/published目录')
    sync.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.command == 'sync':
        result = sync_repository_pdfs(args.repo_root)
    elif args.command == 'build':
        result = build_assets(args.repo_root, args.output)
    else:
        result = verify_assets(args.directory, reference=args.reference)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
