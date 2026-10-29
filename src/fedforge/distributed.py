"""Local multi-process federation protocol.

This module provides a narrow HTTP coordinator/client contract around the
existing ML engine. It intentionally transports only signed model envelopes;
raw samples and labels remain client-local. The coordinator can aggregate
pairwise-masked vectors when every expected client participates.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from .security import canonical_json, mask_update, sign_envelope, unmask_sum, verify_envelope


@dataclass(frozen=True)
class UpdateEnvelope:
    client_id: str
    round_id: int
    parameters: tuple[float, ...]
    num_examples: int
    masked: bool
    signature: str

    def payload(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "parameters": list(self.parameters),
            "num_examples": self.num_examples,
            "masked": self.masked,
        }

    @classmethod
    def create(
        cls, client_id: str, round_id: int, parameters: Sequence[float], num_examples: int,
        secret: str, *, peers: Sequence[str] = (), secure_aggregation: bool = False,
    ) -> "UpdateEnvelope":
        values = tuple(float(value) for value in parameters)
        if secure_aggregation:
            values = mask_update(client_id, values, peers, secret)
        payload = {
            "client_id": client_id, "round_id": round_id, "parameters": list(values),
            "num_examples": int(num_examples), "masked": secure_aggregation,
        }
        return cls(client_id, round_id, values, int(num_examples), secure_aggregation, sign_envelope(payload, secret))

    def verify(self, secret: str) -> bool:
        return verify_envelope(self.payload(), self.signature, secret)


def aggregate_masked(envelopes: Sequence[UpdateEnvelope], expected_clients: Sequence[str], secret: str) -> tuple[float, ...]:
    if set(expected_clients) != {envelope.client_id for envelope in envelopes}:
        raise ValueError("secure aggregation requires exactly one update from every expected client")
    if not envelopes or not all(envelope.masked for envelope in envelopes):
        raise ValueError("all updates must use secure aggregation masking")
    if any(not envelope.verify(secret) for envelope in envelopes):
        raise ValueError("one or more update signatures are invalid")
    return unmask_sum(
        [envelope.parameters for envelope in envelopes],
        [envelope.client_id for envelope in envelopes],
        secret,
    )


def post_update(url: str, envelope: UpdateEnvelope, *, bearer_token: str | None = None, tls_context: ssl.SSLContext | None = None) -> dict[str, Any]:
    body = json.dumps({**envelope.payload(), "signature": envelope.signature}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, context=tls_context, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


class CoordinatorHandler(BaseHTTPRequestHandler):
    """Minimal handler intended for local development and integration tests."""

    server: "CoordinatorServer"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/update":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(size))
            envelope = UpdateEnvelope(
                str(body["client_id"]), int(body["round_id"]), tuple(float(v) for v in body["parameters"]),
                int(body["num_examples"]), bool(body["masked"]), str(body["signature"]),
            )
            if not envelope.verify(self.server.shared_secret):
                raise ValueError("invalid signature")
            self.server.envelopes.append(envelope)
            payload = {"accepted": True, "received": len(self.server.envelopes)}
            self.send_response(200)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            payload = {"accepted": False, "detail": str(exc)}
            self.send_response(400)
        encoded = json.dumps(payload).encode("utf-8")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class CoordinatorServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], shared_secret: str, expected_clients: Sequence[str]) -> None:
        super().__init__(address, CoordinatorHandler)
        self.shared_secret = shared_secret
        self.expected_clients = tuple(expected_clients)
        self.envelopes: list[UpdateEnvelope] = []

    def aggregate(self) -> tuple[float, ...]:
        result = aggregate_masked(self.envelopes, self.expected_clients, self.shared_secret)
        self.envelopes.clear()
        return result
