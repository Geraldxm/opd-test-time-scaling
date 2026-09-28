# 示例数据

`example_math.jsonl` 是仓库内可直接运行的原创 canonical example，共 10 行。每行至少包含 `id`、`problem`、`answer`。

- 行数 / 唯一 ID：10 / 10
- 用途：quickstart、schema 演示，以及数字、分数、表达式、元组、区间和集合答案的 parser smoke；不代表 benchmark 难度。

完整 MATH-500 canonical 和其他经过来源记录的数据版本在 [Geraldxm/math-vault](https://github.com/Geraldxm/math-vault) 维护；其中 `canonical/` 是本仓库直接适配的输入。本仓库不复制数据转换管线。

math-vault 当前将 MATH-500 上游许可证标记为“未声明”，其仓库 MIT License 不对数据重新授权。使用或再分发完整数据前应核对 [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 的最新上游条款。
