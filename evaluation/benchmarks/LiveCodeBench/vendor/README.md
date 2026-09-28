# Frozen LiveCodeBench judge subset

This subset was frozen from a G-OPD code-evaluation snapshot at commit
`37371a4c31ad7947746200d234161769191f4748`. This identifies the G-OPD source
snapshot; it is not a pinned LiveCodeBench GitHub commit. The original
[LiveCodeBench project](https://github.com/LiveCodeBench/LiveCodeBench) is
linked for benchmark context. SHA-256 digests for each copied source file are
recorded in [`data/manifest.json`](../data/manifest.json). The MIT license
attribution is preserved in [`LCB-LICENSE.txt`](LCB-LICENSE.txt).

The copy contains only the code-generation data model, execution harness,
per-sample judge, and pass@K estimator needed after generation. Inference
clients and unrelated benchmark scenarios are omitted. Three small
`__init__.py` files provide local package-loading glue and are not part of the
copied source-file digest list.

Local wrapper code runs this frozen judge subset inside the `sandbox.py` Linux
namespace and chroot boundary. Its `reliability_guard` is defense in depth and
is not a replacement for that OS isolation.
