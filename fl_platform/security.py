"""Authentication and TLS helpers shared by the platform services."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from flask import Flask, Response, request


def install_api_token_auth(app: Flask, token: str | None) -> None:
    if not token or len(token) < 16:
        raise RuntimeError("FL_API_TOKEN must contain at least 16 characters")

    @app.before_request
    def require_bearer_token():
        if request.endpoint == "health":
            return None
        supplied = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not supplied.startswith(prefix) or not hmac.compare_digest(supplied[len(prefix):], token):
            return Response("Unauthorized", status=401, headers={"WWW-Authenticate": "Bearer"})
        return None


def install_basic_auth(app: Flask, username: str | None, password: str | None) -> None:
    if bool(username) != bool(password):
        raise RuntimeError("Set both FL_DASHBOARD_USER and FL_DASHBOARD_PASSWORD")
    if not username:
        return
    if len(password or "") < 12:
        raise RuntimeError("FL_DASHBOARD_PASSWORD must contain at least 12 characters")

    @app.before_request
    def require_dashboard_login():
        auth = request.authorization
        valid = (auth is not None and auth.username is not None and auth.password is not None
                 and hmac.compare_digest(auth.username, username)
                 and hmac.compare_digest(auth.password, password or ""))
        if not valid:
            return Response("Authentication required", status=401,
                            headers={"WWW-Authenticate": 'Basic realm="FL Platform"'})
        return None


def ssl_context_from_env(required: bool = True):
    cert = os.environ.get("FL_TLS_CERT")
    key = os.environ.get("FL_TLS_KEY")
    if not cert and not key and not required:
        return None
    if not cert or not key:
        raise RuntimeError("Set FL_TLS_CERT and FL_TLS_KEY to start the HTTPS service")
    if not Path(cert).is_file() or not Path(key).is_file():
        raise RuntimeError("TLS certificate or key file does not exist")
    return cert, key


def tls_verify_setting() -> str | bool:
    ca_file = os.environ.get("FL_TLS_CA")
    return ca_file if ca_file else True
