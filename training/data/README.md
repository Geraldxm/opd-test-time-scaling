# Training data

| Data | Use | Access |
|---|---|---|
| Math | Training snapshots and framework adapters | See [`math-vault`](https://github.com/Geraldxm/math-vault). This public code repository does not duplicate the math datasets. |
| CodeR1 12K | 12,458 Code training examples | [`CodeR1-12K/README.md`](CodeR1-12K/README.md): download the frozen upstream file and build the two training formats. The upstream dataset does not declare a license; this repository does not redistribute it. |
| Fact NQ + HotpotQA | 169,615 closed-book training examples | [`FactQA/README.md`](FactQA/README.md): download the pinned upstream training split and build the frozen EOPD Parquet adapter. NQ here is training data; no NQ evaluation set is included. |

Downloads and generated datasets are ignored by Git. The manifests record source revisions, SHA-256 digests, row counts, and adapter contracts.
