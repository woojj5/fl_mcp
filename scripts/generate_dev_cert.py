#!/usr/bin/env python3
"""Create a self-signed development certificate for the Compose service names."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "certs"
CERT_PATH = CERT_DIR / "server.crt"
KEY_PATH = CERT_DIR / "server.key"


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "FL Platform Development"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Local Development"),
    ])
    now = datetime.now(timezone.utc)
    sans = [x509.DNSName(name) for name in (
        "localhost", "fl-server", "fl-client-0", "fl-client-1", "fl-dashboard",
        "host.docker.internal",
    )]
    sans.append(x509.IPAddress(ip_address("127.0.0.1")))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(private_key, hashes.SHA256())
    )
    KEY_PATH.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    CERT_PATH.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    try:
        KEY_PATH.chmod(0o600)
    except OSError:
        pass
    print(f"Created {CERT_PATH.relative_to(ROOT)} and {KEY_PATH.relative_to(ROOT)}")
    print("This certificate is for local development only.")


if __name__ == "__main__":
    main()
