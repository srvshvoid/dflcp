# Docker status

`compose.yml` contains only the `dev` static replay viewer profile. It is a
safe packaging check, not the requested multi-container federated deployment.
The `sim` and `deploy` profiles remain pending until Flower's pinned app model,
private client volumes, network policy, database migrations, Redis eventing,
and GPU resource limits have been implemented and tested.
