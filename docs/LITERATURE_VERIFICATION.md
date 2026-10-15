# Literature verification protocol

**Status: no literature claim is verified by this repository.** This document
prevents placeholder citations or remembered paper titles from being used as
evidence. Add a citation only after a reviewer can inspect the source and the
claim actually supported by it.

## Claim register

| Claim that might be made | Primary source to verify | Exact passage / artifact | Status |
| --- | --- | --- | --- |
| Federated optimization behavior | Original algorithm paper or authoritative specification | To be recorded with page/section | Pending |
| Secure aggregation threat guarantees | Protocol paper/specification and implementation version | To be recorded with assumptions | Pending |
| Robust aggregation behavior | Original method paper and threat model | To be recorded with attack setting | Pending |
| Differential privacy accounting | Mechanism/accounting reference and library version | To be recorded with parameters | Pending |
| Media-forensics dataset facts | Official dataset release and license | To be recorded with immutable revision | Pending |
| Evaluation protocol | Method paper plus dataset/split documentation | To be recorded with limitations | Pending |

## Verification steps

1. Start from a primary publication, official specification, or dataset
   release. Do not cite a search-result snippet as evidence.
2. Record authors, title, venue or release, year, persistent identifier, and
   the exact version accessed.
3. Quote or paraphrase the precise passage that supports the claim, including
   its assumptions and scope. A paper demonstrating a method on one dataset
   does not support a universal performance claim.
4. Check that the implementation, protocol, and dataset version used by
   FedForge match the cited method. Record deviations.
5. Have a second reviewer mark the row verified, limited, or rejected. Keep
   the source and verification date with the experiment record.

## Writing rule

Until the register is populated, use language such as “to be verified” and
describe only what the local code actually demonstrates. Do not add invented
DOIs, benchmark tables, state-of-the-art comparisons, or claims that a
security property follows from a package name.
