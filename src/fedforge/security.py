"""Security primitives for the local federated-process prototype.

The primitives in this module are deliberately small and auditable:

* HMAC-signed JSON envelopes provide peer authentication and integrity.
* Pairwise additive masks let a coordinator aggregate masked vectors without
  receiving an individual client's unmasked update.
* Differential-privacy preprocessing clips a client update and optionally
  adds Gaussian noise.

Transport confidentiality is supplied by TLS at the HTTP layer; HMAC is not
encryption.  A production deployment still needs managed keys, rotation,
dropout recovery, and a reviewed secure-aggregation protocol.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
from typing import Any, Mapping, Sequence


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sign_envelope(payload: Mapping[str, Any], secret: str | bytes) -> str:
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(key, canonical_json(payload), hashlib.sha256).hexdigest()


def verify_envelope(payload: Mapping[str, Any], signature: str, secret: str | bytes) -> bool:
    return hmac.compare_digest(sign_envelope(payload, secret), signature)


def derive_pairwise_seed(client_a: str, client_b: str, shared_secret: str | bytes) -> bytes:
    first, second = sorted((client_a, client_b))
    key = shared_secret.encode("utf-8") if isinstance(shared_secret, str) else shared_secret
    return hmac.new(key, f"pair:{first}:{second}".encode("utf-8"), hashlib.sha256).digest()


def pairwise_mask(client_id: str, peer_id: str, dimension: int, shared_secret: str | bytes) -> tuple[float, ...]:
    """Create a deterministic pairwise mask with opposite signs per client."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    seed = derive_pairwise_seed(client_id, peer_id, shared_secret)
    values = []
    counter = 0
    while len(values) < dimension:
        digest = hmac.new(seed, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        for offset in range(0, len(digest), 8):
            if len(values) == dimension:
                break
            raw = int.from_bytes(digest[offset:offset + 8], "big")
            values.append((raw / 2**64 - 0.5) * 2.0)
        counter += 1
    sign = 1.0 if client_id < peer_id else -1.0
    return tuple(sign * value for value in values)


def mask_update(
    client_id: str,
    parameters: Sequence[float],
    peers: Sequence[str],
    shared_secret: str | bytes,
) -> tuple[float, ...]:
    masked = [float(value) for value in parameters]
    for peer_id in peers:
        if peer_id == client_id:
            continue
        mask = pairwise_mask(client_id, peer_id, len(masked), shared_secret)
        masked = [value + delta for value, delta in zip(masked, mask)]
    return tuple(masked)


def unmask_sum(
    masked_updates: Sequence[Sequence[float]],
    client_ids: Sequence[str],
    shared_secret: str | bytes,
) -> tuple[float, ...]:
    """Recover the sum when all expected clients submitted masked updates."""
    if len(masked_updates) != len(client_ids) or not masked_updates:
        raise ValueError("masked update and client lists must be non-empty and have equal length")
    dimension = len(masked_updates[0])
    if any(len(update) != dimension for update in masked_updates):
        raise ValueError("all masked updates must have the same dimension")
    expected = set(client_ids)
    if len(expected) != len(client_ids):
        raise ValueError("client ids must be unique")
    peers = tuple(client_ids)
    recovered = [0.0] * dimension
    for client_id, update in zip(client_ids, masked_updates):
        unmasked = mask_update(client_id, update, peers, shared_secret)
        recovered = [left + right for left, right in zip(recovered, unmasked)]
    return tuple(recovered)


def clip_and_noise(
    parameters: Sequence[float],
    reference: Sequence[float],
    *,
    clip_norm: float,
    noise_multiplier: float = 0.0,
    rng: Any = None,
) -> tuple[float, ...]:
    """Clip an update delta and optionally add Gaussian noise.

    The returned vector remains in model-parameter space.  This is a
    client-level DP building block; callers must account for sampling,
    composition, and privacy amplification separately.
    """
    if clip_norm <= 0 or noise_multiplier < 0:
        raise ValueError("clip_norm must be positive and noise_multiplier non-negative")
    if len(parameters) != len(reference):
        raise ValueError("parameters and reference must have equal length")
    delta = [float(value) - float(base) for value, base in zip(parameters, reference)]
    norm = math.sqrt(sum(value * value for value in delta))
    scale = min(1.0, clip_norm / (norm + 1e-12))
    if rng is None:
        import random
        gaussian = random.gauss
    else:
        gaussian = rng.normal
    sigma = clip_norm * noise_multiplier
    return tuple(
        float(base) + value * scale + (float(gaussian(0.0, sigma)) if sigma else 0.0)
        for base, value in zip(reference, delta)
    )


def random_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
