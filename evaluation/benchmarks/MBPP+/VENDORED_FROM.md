# EvalPlus vendoring and scorer provenance

The original project is [EvalPlus](https://github.com/evalplus/evalplus). This
directory contains only the runtime needed to score pre-generated HumanEval+
and MBPP+ samples; inference support is intentionally absent. The bundled
source files and licenses are listed in `vendor/evalplus/SHA256SUMS`. The
source-tree digest is
`e1a5530bae474eade8245f1588901938dd0a06354d42b38c2ed3d6f098480e5d`; it is
calculated by sorting those manifest entries and hashing, for each entry,
`relative_path + NUL + raw_sha256_bytes + newline`.

The source snapshot came from the G-OPD code-evaluation fork at repository
commit `37371a4c31ad7947746200d234161769191f4748`. That identifier is
a G-OPD source snapshot commit, **not** an EvalPlus upstream commit. The frozen
judge module `evalplus/eval/__init__.py` has SHA-256
`2ba7560f74466c500b95fe995b14c509aa75f6c452df10d2583059878d60a07f`.

The G-OPD working tree contained a local timeout safeguard around module-level
execution. The exact diff from the G-OPD commit is included as
[`VENDORED_TIMEOUT_PATCH.diff`](VENDORED_TIMEOUT_PATCH.diff), SHA-256
`4bba455ec37fd2a579a5e7e9347d34341208701ac095908162c1afbb66719e4d`. It wraps
`exec(code, exec_globals)` in `time_limit(max(time_limits))`. The vendored CLI
also removes inference entry points and uses this package's frozen data and
stdlib-only loading paths. The software and dataset licenses are retained with
the vendored files.

Accepted paper-run manifests pin generation and dataset information but do
not attest the EvalPlus working-tree source hash. This frozen source and patch
make current replay reproducible, but they cannot establish the exact source
diff used for every accepted historical score.
