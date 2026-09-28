# Evaluation

`math_eval/` is a frozen source snapshot of
[Geraldxm/math-eval](https://github.com/Geraldxm/math-eval) at commit
`7e4b21750fc4917649c325a092d83ba2da05f36f`. It produces sealed raw
responses and completed run manifests, and scores Math responses by replay.
The benchmark directories provide their paper-specific data and prompts.

```text
benchmark data + prompt -> math_eval inference -> sealed raw + manifest
                                                  |
                        Math -> math_eval replay -+
                        Code -> local code judge -+
                        Fact -> local EM scorer --+
                                                  v
                                         per-response verdicts -> pass@K/avg@K
```

HumanEval+, MBPP+, and LiveCodeBench each contain their own post-inference
converter, frozen judge source, scoring entry point, and README. Their CPU
judges do not use the Math answer parser. FactQA uses its own tagged-answer
parser and alias exact-match scorer. The four Math datasets use the frozen
`math_eval` replay path. See each benchmark README for commands and upstream
attribution.
