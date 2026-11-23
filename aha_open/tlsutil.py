"""Self-signed certificate for the `lan` profile and its fingerprint (clients pin the fingerprint; docs/06_deployment.md)."""

from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from .auth import restrict_to_current_user
from .config import WILDCARD_HOSTS, Config


def _san_entries(cfg: Config) -> list[x509.GeneralName]:
    names = {"localhost", socket.gethostname(), *cfg.hosts}
    if cfg.listen_host not in WILDCARD_HOSTS:
        names.add(cfg.listen_host)
    entries: list[x509.GeneralName] = []
    for name in sorted(n for n in names if n):
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            entries.append(x509.DNSName(name))
    entries.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    return entries


def init_tls(cfg: Config, force: bool = False) -> Path:
    cert_path, key_path = cfg.tls_cert_path, cfg.tls_key_path
    if cert_path.exists() and key_path.exists() and not force:
        return cert_path
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "aha-open")])
    start = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(start + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(_san_entries(cfg)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    restrict_to_current_user(key_path)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path


def fingerprint(cert_path: Path) -> str:
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    digest = cert.fingerprint(hashes.SHA256()).hex().upper()
    return ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))
