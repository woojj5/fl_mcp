"""FL Client Node entry point.

Trains on local data, uploads weights to the FL server, and serves
local predictions at /predict.

Usage:
    python client.py --server https://192.168.1.10:8080 --client-id client_0
    python client.py --server https://192.168.1.10:8080 \\
                     --client-id client_1 --port 8081 \\
                     --model simple_cnn --dataset cifar10 \\
                     --partition-id 1 --num-clients-total 5

API endpoints exposed on this node:
    GET  /status          -- client state
    POST /train           -- trigger one training round manually
    POST /predict         -- inference with local model
    POST /admin/start     -- start auto-training loop
    POST /admin/stop      -- stop auto-training loop
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="FL Client Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Identity
    parser.add_argument("--client-id",         default="client_0",
                        help="Unique identifier for this client")
    parser.add_argument("--server",            default="https://localhost:8080",
                        help="FL server URL")
    parser.add_argument("--my-address",        default="localhost",
                        help="Address this client advertises to the server")
    # Listening
    parser.add_argument("--host",              default="0.0.0.0")
    parser.add_argument("--port",              type=int, default=8081,
                        help="Port for this client's Flask server")
    # Model / data
    parser.add_argument("--model",             default="simple_cnn")
    parser.add_argument("--num-classes",       type=int, default=10)
    parser.add_argument("--dataset",           default="cifar10")
    parser.add_argument("--partition-id",      type=int, default=0,
                        help="Data partition index for this client")
    parser.add_argument("--num-clients-total", type=int, default=5,
                        help="Total number of clients (used to partition data)")
    parser.add_argument("--split-type",        default="iid",
                        choices=["iid","dirichlet","class_based"])
    # Training
    parser.add_argument("--batch-size",        type=int, default=32)
    parser.add_argument("--local-epoch",       type=int, default=5)
    parser.add_argument("--lr",                type=float, default=0.01)
    parser.add_argument("--data-dir",           default="./data")
    parser.add_argument("--alpha",              type=float, default=0.5,
                        help="Dirichlet concentration for non-IID partitioning")
    parser.add_argument("--optimizer",         default="sgd",
                        choices=["sgd","adam"])
    parser.add_argument("--gpu",               type=int, default=0,
                        help="GPU count (0 = CPU)")
    parser.add_argument("--seed",              type=int, default=42)
    # Behaviour
    parser.add_argument("--no-auto-start",     action="store_true",
                        help="Do not auto-start the training loop on launch")
    args = parser.parse_args()

    try:
        from flask import Flask
    except ImportError:
        print("Flask not installed.  Run:  pip install flask")
        sys.exit(1)

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

    print()
    print("=" * 60)
    print("  FL Client Node")
    print(f"  Client ID :  {args.client_id}")
    print(f"  Server    :  {args.server}")
    print(f"  Listening :  https://{args.host}:{args.port}")
    print(f"  Model     :  {args.model}  ({args.num_classes} classes)")
    print(f"  Dataset   :  {args.dataset}  (partition {args.partition_id}/{args.num_clients_total})")
    print(f"  Split     :  {args.split_type}")
    print(f"  LR={args.lr}  epochs={args.local_epoch}  batch={args.batch_size}  opt={args.optimizer}")
    print("=" * 60)
    print()

    if not args.no_auto_start:
        print("  Auto-training loop starting...")
        node.start_loop()

    from fl_platform.security import ssl_context_from_env
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            ssl_context=ssl_context_from_env(required=True))


if __name__ == "__main__":
    main()
