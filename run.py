"""FL Platform entry point.

Three operating modes:

  dashboard   Launch the simulation + monitoring web UI (default)
              python run.py
              python run.py --port 5000 --db ./tracking.db

  server      Launch the FL coordination server for distributed deployment
              python run.py server
              python run.py server --port 8080 --model simple_cnn --rounds 20

  client      Launch an FL client node for distributed deployment
              python run.py client --server https://<server-ip>:8080
              python run.py client --server https://<server-ip>:8080 \\
                                   --client-id client_0 --port 8081

For full help on any mode:
    python run.py dashboard --help
    python run.py server    --help
    python run.py client    --help

Quick-start distributed example (3 terminals):
    # Terminal 1 - server
    python run.py server --port 8080 --min-clients 2 --rounds 10

    # Terminal 2 - client 0
    python run.py client --server https://localhost:8080 --client-id client_0 --port 8081

    # Terminal 3 - client 1
    python run.py client --server https://localhost:8080 --client-id client_1 --port 8082
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Dashboard mode
# ---------------------------------------------------------------------------

def run_dashboard(argv):
    parser = argparse.ArgumentParser(
        prog="run.py dashboard",
        description="FL Platform simulation + monitoring dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",           type=int, default=5000)
    parser.add_argument("--host",           default="0.0.0.0")
    parser.add_argument("--db",             default="./tracking.db",
                        help="SQLite metrics database path")
    parser.add_argument("--fl-server-url",  default=os.environ.get("FL_SERVER_URL", ""),
                        help="FL server URL for deployment monitor (e.g. https://localhost:8080)")
    parser.add_argument("--fl-client-urls", default=os.environ.get("FL_CLIENT_URLS", ""),
                        help="Comma-separated client URLs (e.g. https://localhost:8081,https://localhost:8082)")
    args = parser.parse_args(argv)

    dashboard_user = os.environ.get("FL_DASHBOARD_USER")
    dashboard_password = os.environ.get("FL_DASHBOARD_PASSWORD")
    api_token = os.environ.get("FL_API_TOKEN")
    if not dashboard_user or not dashboard_password:
        raise RuntimeError("Set FL_DASHBOARD_USER and FL_DASHBOARD_PASSWORD before starting the dashboard")
    if not api_token:
        raise RuntimeError("Set FL_API_TOKEN before starting the dashboard")

    _check_flask()
    from fl_platform.dashboard.app import create_app

    client_urls = [u.strip() for u in args.fl_client_urls.split(",") if u.strip()]
    app = create_app(
        db_path=args.db,
        fl_server_url=args.fl_server_url or None,
        fl_client_urls=client_urls or None,
        api_token=api_token,
        dashboard_user=dashboard_user,
        dashboard_password=dashboard_password,
    )

    _banner("FL Platform Dashboard", [
        f"Open in browser : https://localhost:{args.port}",
        f"Database        : {args.db}",
        f"Deployment tab  : https://localhost:{args.port}/deployment",
        f"FL Server       : {args.fl_server_url or '(not set)'}",
        f"FL Clients      : {args.fl_client_urls or '(not set)'}",
    ])
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            ssl_context=_tls_context())


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------

def run_server(argv):
    parser = argparse.ArgumentParser(
        prog="run.py server",
        description="FL Coordination Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",              default="0.0.0.0")
    parser.add_argument("--port",              type=int, default=8080)
    parser.add_argument("--model",             default="simple_cnn")
    parser.add_argument("--num-classes",       type=int, default=10)
    parser.add_argument("--dataset",           default="cifar10")
    parser.add_argument("--rounds",            type=int, default=20)
    parser.add_argument("--min-clients",       type=int, default=2)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--strategy",          default="fedavg",
                        choices=["fedavg","fedprox","fedadam","fedyogi",
                                 "fedavgm","fedmedian","krum","dp_fedavg"])
    parser.add_argument("--db",                default="./server_tracking.db")
    parser.add_argument("--task-id",           default="fl_deploy")
    args = parser.parse_args(argv)

    _check_flask()
    from fl_platform.deploy.fl_server import create_fl_server_app

    conf = {
        "model":             args.model,
        "num_classes":       args.num_classes,
        "dataset":           args.dataset,
        "rounds":            args.rounds,
        "min_clients":       args.min_clients,
        "clients_per_round": args.clients_per_round,
        "strategy":          args.strategy,
        "db_path":           args.db,
        "task_id":           args.task_id,
        "api_token":         os.environ.get("FL_API_TOKEN"),
    }
    app, _ = create_fl_server_app(conf)

    _banner("FL Coordination Server", [
        f"Address   : https://{args.host}:{args.port}",
        f"Model     : {args.model}  ({args.num_classes} classes)",
        f"Dataset   : {args.dataset}",
        f"Strategy  : {args.strategy}",
        f"Rounds    : {args.rounds}",
        f"Min clients to start       : {args.min_clients}",
        f"Clients per round          : {args.clients_per_round}",
        "",
        "Connect clients with:",
        f"  python run.py client --server https://<this-ip>:{args.port} \\",
        f"                       --model {args.model} --dataset {args.dataset}",
    ])
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            ssl_context=_tls_context())


# ---------------------------------------------------------------------------
# Client mode
# ---------------------------------------------------------------------------

def run_client(argv):
    parser = argparse.ArgumentParser(
        prog="run.py client",
        description="FL Client Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--client-id",         default="client_0")
    parser.add_argument("--server",            default="https://localhost:8080")
    parser.add_argument("--my-address",        default="localhost")
    parser.add_argument("--host",              default="0.0.0.0")
    parser.add_argument("--port",              type=int, default=8081)
    parser.add_argument("--model",             default="simple_cnn")
    parser.add_argument("--num-classes",       type=int, default=10)
    parser.add_argument("--dataset",           default="cifar10")
    parser.add_argument("--partition-id",      type=int, default=0)
    parser.add_argument("--num-clients-total", type=int, default=5)
    parser.add_argument("--split-type",        default="iid",
                        choices=["iid","dirichlet","class_based"])
    parser.add_argument("--batch-size",        type=int, default=32)
    parser.add_argument("--local-epoch",       type=int, default=5)
    parser.add_argument("--lr",                type=float, default=0.01)
    parser.add_argument("--data-dir",           default="./data")
    parser.add_argument("--alpha",              type=float, default=0.5,
                        help="Dirichlet concentration for non-IID partitioning")
    parser.add_argument("--optimizer",         default="sgd",
                        choices=["sgd","adam"])
    parser.add_argument("--gpu",               type=int, default=0)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--no-auto-start",     action="store_true",
                        help="Do not auto-start the training loop on launch")
    args = parser.parse_args(argv)

    _check_flask()
    from fl_platform.deploy.fl_client import create_fl_client_app

    conf = {
        "client_id":         args.client_id,
        "server_url":        args.server,
        "my_address":        args.my_address,
        "model":             args.model,
        "num_classes":       args.num_classes,
        "dataset":           args.dataset,
        "data_dir":          args.data_dir,
        "partition_id":      args.partition_id,
        "num_clients_total": args.num_clients_total,
        "split_type":        args.split_type,
        "batch_size":        args.batch_size,
        "local_epoch":       args.local_epoch,
        "lr":                args.lr,
        "optimizer":         args.optimizer,
        "gpu":               args.gpu,
        "seed":              args.seed,
        "alpha":             args.alpha,
        "api_token":         os.environ.get("FL_API_TOKEN"),
    }
    app, node = create_fl_client_app(conf)

    _banner("FL Client Node", [
        f"Client ID : {args.client_id}",
        f"Server    : {args.server}",
        f"Listening : https://{args.host}:{args.port}",
        f"Model     : {args.model}  ({args.num_classes} classes)",
        f"Dataset   : {args.dataset}  (partition {args.partition_id}/{args.num_clients_total})",
        f"Split     : {args.split_type}",
        f"LR={args.lr}  epochs={args.local_epoch}  batch={args.batch_size}  opt={args.optimizer}",
    ])

    if not args.no_auto_start:
        print("  Auto-training loop starting...\n")
        node.start_loop()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            ssl_context=_tls_context())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_flask():
    try:
        import flask
    except ImportError:
        print("Flask not installed.  Run:  pip install flask")
        sys.exit(1)


def _tls_context():
    from fl_platform.security import ssl_context_from_env
    return ssl_context_from_env(required=True)


def _banner(title: str, lines: list):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("-" * width)
    for line in lines:
        print(f"  {line}" if line else "")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def main():
    modes = {"dashboard", "server", "client"}

    # No args or first arg is a flag -> dashboard mode
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        run_dashboard(sys.argv[1:])
        return

    mode = sys.argv[1].lower()
    if mode not in modes:
        # Might be --port etc passed directly -> dashboard
        run_dashboard(sys.argv[1:])
        return

    rest = sys.argv[2:]
    if mode == "dashboard":
        run_dashboard(rest)
    elif mode == "server":
        run_server(rest)
    elif mode == "client":
        run_client(rest)


if __name__ == "__main__":
    main()
