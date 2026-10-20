"""Flask dashboard for SQLite experiment history and remote node status."""

from __future__ import annotations

from typing import Any

import requests
from flask import Flask, jsonify, render_template_string

from fl_platform.coordinator import SimulationManager
from fl_platform.tracking import get_clients, get_rounds, get_runs, initialize
from fl_platform.security import install_api_token_auth, install_basic_auth, tls_verify_setting

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FL Platform</title><style>
body{font:15px system-ui,sans-serif;margin:0;background:#f3f5f8;color:#182230}header{background:#16243a;color:white;padding:22px 5vw}
main{max-width:1120px;margin:24px auto;padding:0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
section{background:white;border-radius:10px;padding:18px;margin:14px 0;box-shadow:0 1px 4px #0001}h1,h2{margin:0 0 12px}h2{font-size:18px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e8edf2}th{color:#596579}
.muted{color:#687588}.pill{padding:3px 8px;border-radius:12px;background:#e7f5ec;color:#24643b}.error{color:#a42727}
a{color:#2260a8}code{background:#edf1f5;padding:2px 5px;border-radius:4px}
</style></head><body><header><h1>FL Platform</h1><div>Federated learning experiment and deployment monitor</div></header><main>
<div class="grid"><section><h2>Experiments</h2><strong>{{ runs|length }}</strong><div class="muted">tracked runs</div></section>
<section><h2>Latest round</h2><strong>{{ rounds[0].round if rounds else '—' }}</strong><div class="muted">recorded aggregation round</div></section>
<section><h2>Deployment</h2><a href="/deployment">Open deployment monitor</a></section></div>
<section><h2>Run local simulation</h2><form id="simulation-form" class="grid">
<label>Dataset <select name="dataset"><option>cifar10</option><option>cifar100</option><option>mnist</option><option>fashionmnist</option><option>svhn</option></select></label>
<label>Model <select name="model"><option>simple_cnn</option><option>lenet</option><option>mlp</option><option>resnet18</option></select></label>
<label>Rounds <input name="rounds" type="number" min="1" max="100" value="3"></label>
<label>Clients <input name="num_clients" type="number" min="1" max="32" value="2"></label>
<label>Local epochs <input name="local_epoch" type="number" min="1" max="20" value="1"></label>
<label>Batch size <input name="batch_size" type="number" min="1" max="1024" value="32"></label>
<label>Learning rate <input name="lr" type="number" min="0.00001" step="any" value="0.01"></label>
<label>Split <select name="split_type"><option>iid</option><option>dirichlet</option><option>class_based</option></select></label>
<label>Strategy <select name="strategy"><option>fedavg</option><option>fedprox</option><option>fedmedian</option></select></label>
<label>Seed <input name="seed" type="number" value="42"></label>
<button type="submit">Start simulation</button><output id="simulation-status" class="muted">Idle</output>
</form></section>
<section><h2>Tracked experiments</h2>{% if runs %}<table><tr><th>Task</th><th>Model</th><th>Dataset</th><th>Strategy</th><th>Status</th><th>Started</th></tr>
{% for run in runs %}<tr><td>{{ run.task_id }}</td><td>{{ run.model }}</td><td>{{ run.dataset }}</td><td>{{ run.strategy }}</td><td>{{ run.status }}</td><td>{{ run.created_at }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted">No runs recorded yet. Start a distributed server to create a run record.</p>{% endif %}</section>
<section><h2>Recent rounds</h2>{% if rounds %}<table><tr><th>Task</th><th>Round</th><th>Clients</th><th>Loss</th><th>Accuracy</th><th>Recorded</th></tr>
{% for row in rounds %}<tr><td>{{ row.task_id }}</td><td>{{ row.round }}</td><td>{{ row.clients }}</td><td>{{ row.loss if row.loss is not none else '—' }}</td><td>{{ row.accuracy if row.accuracy is not none else '—' }}</td><td>{{ row.created_at }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted">Round metrics will appear after clients submit updates.</p>{% endif %}</section>
<p class="muted">SQLite: <code>{{ db_path }}</code></p></main><script>
const form=document.getElementById('simulation-form'),out=document.getElementById('simulation-status');
form.addEventListener('submit',async(e)=>{e.preventDefault();const values=Object.fromEntries(new FormData(form));
for(const key of ['rounds','num_clients','local_epoch','batch_size','seed'])values[key]=Number(values[key]);values.lr=Number(values.lr);
const response=await fetch('/api/simulation/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
const data=await response.json();out.textContent=response.ok?'Starting '+data.task_id:(data.error||'Could not start');});
setInterval(async()=>{const r=await fetch('/api/simulation/status'),s=await r.json();if(s.status!=='idle')out.textContent=`${s.status}: round ${s.round}/${s.total_rounds} ${s.message||s.error||''}`;},2000);
</script></body></html>"""

DEPLOYMENT_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deployment Monitor · FL Platform</title><style>body{font:15px system-ui,sans-serif;margin:0;background:#f3f5f8;color:#182230}header{background:#16243a;color:white;padding:22px 5vw}
main{max-width:1100px;margin:25px auto;padding:0 18px}section{background:white;padding:18px;margin:14px 0;border-radius:10px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #e8edf2}.error{color:#a42727}.muted{color:#687588}a{color:#b9d8ff}</style></head><body>
<header><h1>Deployment Monitor</h1><a href="/">← Dashboard</a></header><main>
<section><h2>FL server</h2>{% if server %}<pre>{{ server|tojson(indent=2) }}</pre>{% else %}<p class="error">{{ server_error or 'Server URL is not configured.' }}</p>{% endif %}</section>
<section><h2>Client nodes</h2>{% if clients %}<table><tr><th>URL</th><th>State</th><th>Client</th><th>Round</th><th>Details</th></tr>
{% for node in clients %}<tr><td>{{ node.url }}</td><td>{{ node.data.status if node.data else 'unavailable' }}</td><td>{{ node.data.client_id if node.data else '—' }}</td><td>{{ node.data.round if node.data else '—' }}</td><td>{{ node.error or '' }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted">No client URLs configured.</p>{% endif %}</section>
<p class="muted">This page reads node status endpoints. It does not expose node control actions.</p></main></body></html>"""


def create_app(db_path: str = "./tracking.db", fl_server_url: str | None = None,
               fl_client_urls: list[str] | None = None, api_token: str | None = None,
               dashboard_user: str | None = None, dashboard_password: str | None = None):
    app = Flask("fl_dashboard")
    app.config.update(DB_PATH=db_path, FL_SERVER_URL=fl_server_url,
                      FL_CLIENT_URLS=fl_client_urls or [])
    install_basic_auth(app, dashboard_user, dashboard_password)
    initialize(db_path)
    simulation = SimulationManager(db_path)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "fl-dashboard"})

    @app.get("/")
    def index():
        return render_template_string(PAGE, runs=get_runs(db_path), rounds=get_rounds(db_path), db_path=db_path)

    @app.get("/api/metrics")
    def metrics():
        return jsonify({"runs": get_runs(db_path), "rounds": get_rounds(db_path),
                        "clients": get_clients(db_path)})

    @app.post("/api/simulation/start")
    def start_simulation():
        try:
            return jsonify(simulation.start(request_json())), 202
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/simulation/status")
    def simulation_status():
        return jsonify(simulation.snapshot())

    @app.get("/deployment")
    def deployment():
        server, server_error = _fetch_json(fl_server_url, api_token)
        nodes = []
        for url in app.config["FL_CLIENT_URLS"]:
            data, error = _fetch_json(url, api_token)
            nodes.append({"url": url, "data": data, "error": error})
        return render_template_string(DEPLOYMENT_PAGE, server=server, server_error=server_error,
                                      clients=nodes)

    @app.get("/api/deployment")
    def deployment_json():
        server, server_error = _fetch_json(fl_server_url, api_token)
        nodes = []
        for url in app.config["FL_CLIENT_URLS"]:
            data, error = _fetch_json(url, api_token)
            nodes.append({"url": url, "data": data, "error": error})
        return jsonify({"server": server, "server_error": server_error, "clients": nodes})

    return app


def _fetch_json(base_url: str | None, api_token: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    if not base_url:
        return None, None
    url = base_url.rstrip("/") + "/status"
    try:
        response = requests.get(url, timeout=2,
                                headers={"Authorization": f"Bearer {api_token}"} if api_token else {},
                                verify=tls_verify_setting())
        response.raise_for_status()
        return response.json(), None
    except (requests.RequestException, ValueError) as exc:
        return None, f"{url}: {exc}"


def request_json() -> dict[str, Any]:
    from flask import request
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("A JSON simulation configuration is required")
    return data
