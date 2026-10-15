# Ethics and responsible-use notes

FedForge concerns media and federated learning, which can involve biometric
information, sensitive identities, and high-impact accusations of
manipulation. A successful software check is not evidence that a proposed use
is ethical or that a detector is reliable in the field.

## Required principles

- **Consent and governance:** use only data with documented authority,
  appropriate consent or legal basis, and a named data steward. Follow the
  retention and deletion decision in the dataset verification record.
- **Data minimization:** keep raw media, face crops, identities, and labels out
  of logs and replays unless their presence is necessary, approved, and
  protected.
- **Fairness and validity:** assess performance across relevant conditions and
  populations. Do not infer fairness from the synthetic fixture or from an
  aggregate number without subgroup evidence.
- **Human review:** treat a detector output as a research signal, never as the
  sole basis for moderation, employment, education, legal, medical, or safety
  decisions.
- **Uncertainty:** expose calibration limits, abstention behavior, data shift,
  and false-positive/false-negative costs. Do not turn an unverified replay
  into a confidence claim.
- **Security and privacy:** document who can inspect client updates, labels,
  checkpoints, and run artifacts. Secure aggregation does not by itself solve
  every inference or governance risk.
- **Transparency:** preserve configs, seeds, provenance, code revisions, and
  limitations so an independent reviewer can reproduce or challenge a claim.

## Misuse boundaries

The project must not be presented as a forensic adjudicator, identity
verifier, or proof that a person created or distributed manipulated media. The
static viewer is intentionally unsuitable for handling production secrets or
uploading sensitive media. Do not publish a result without the dataset license,
split audit, threat assumptions, and uncertainty evidence required by the
project records.

## Reporting and correction

Record material failures, excluded examples, data leakage, attack success,
privacy incidents, and changed assumptions. If a result is later found to be
wrong or overstated, preserve the original artifact, mark it withdrawn, and
publish the correction with the affected claim. The results document should
link to evidence rather than silently replacing a number.
