"""FL Coordination Server entry point.

Manages FL rounds, aggregates client weights, and serves global predictions.

Usage:
    python server.py
    python server.py --port 8080 --model simple_cnn --dataset cifar10
    python server.py --port 8080 --rounds 20 --min-clients 2 --clients-per-round 5

API endpoints:
    GET  /status              -- server state, round info
    POST /register            -- client registers
    GET  /global-model        -- download current global model
    POST /upload-weights      -- client uploads trained weights
    POST /predict             -- inference with global model
    POST /admin/next-round    -- force advance round
    POST /admin/reset         -- reset server state
    GET  /admin/clients       -- list registered clients
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="FL Coordination Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",             default="0.0.0.0")
    parser.add_argument("--port",             type=int, default=8080)
    parser.add_argument("--model",            default="simple_cnn",
                        help="Model architecture name")
    parser.add_argument("--num-classes",      type=int, default=10)
    parser.add_argument("--dataset",          default="cifar10")
    parser.add_argument("--rounds",           type=int, default=20)
    parser.add_argument("--min-clients",      type=int, default=2,
                        help="Minimum clients before training starts")
    parser.add_argument("--clients-per-round",type=int, default=5,
                        help="Clients needed to trigger aggregation each round")
    parser.add_argument("--strategy",         default="fedavg",
                        choices=["fedavg","fedprox","fedadam","fedyogi",
                                 "fedavgm","fedmedian","krum","dp_fedavg"])
    parser.add_argument("--db",               default="./server_tracking.db",
                        help="SQLite metrics database path")
    parser.add_argument("--task-id",          default="fl_deploy")
    args = parser.parse_args()

    try:
        from flask import Flask
    except ImportError:
        print("Flask not installed.  Run:  pip install flask")
        sys.exit(1)

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

    app, state = create_fl_server_app(conf)

    print()
    print("=" * 60)
    print("  FL Coordination Server")
    print(f"  Address  :  https://{args.host}:{args.port}")
    print(f"  Model    :  {args.model}  ({args.num_classes} classes)")
    print(f"  Dataset  :  {args.dataset}")
    print(f"  Strategy :  {args.strategy}")
    print(f"  Rounds   :  {args.rounds}")
    print(f"  Min clients before start : {args.min_clients}")
    print(f"  Clients per round        : {args.clients_per_round}")
    print()
    print("  Clients connect with:")
    print(f"    python client.py --server https://<this-ip>:{args.port} \\")
    print(f"                     --model {args.model} --dataset {args.dataset}")
    print("=" * 60)
    print()

    from fl_platform.security import ssl_context_from_env
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            ssl_context=ssl_context_from_env(required=True))


if __name__ == "__main__":
    main()
