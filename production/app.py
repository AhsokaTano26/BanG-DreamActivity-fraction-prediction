"""Web-only operational entry; fitting is an explicit training command."""
import argparse
from pathlib import Path

from production.deployment import configure, DeploymentWorker
from production.web import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description='加载已训练模型并启动活动终分预测网页')
    parser.add_argument('--state', type=Path, default=Path(__file__).resolve().parents[1] / 'runtime/live')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=int, default=None)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('端口应在1至65535之间')
    if args.interval is not None and args.interval < 60:
        parser.error('更新间隔至少60秒')
    config = configure(args.state, auto_train=False, interval=args.interval)
    config_path = Path(config['root']) / 'config.json'
    return serve(config, config_path, host=args.host, port=args.port,
                 interval=args.interval, offline=args.offline, worker_factory=DeploymentWorker)
