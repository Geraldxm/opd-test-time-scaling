# Training

`rethink_opd/verl/` is the frozen OPD training engine carried forward from the
review artifact. It retains upstream `LICENSE` and `Notice.txt`; the source
project is [Thinking-Space/Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD).
The one-step [`example_train.sh`](example_train.sh) is a runnable training-path
example once local student and teacher checkpoints and a suitable GPU environment
are supplied. It is not a frozen contract for every paper experiment.

From the repository root:

```bash
python data/build_example_train.py
bash training/example_train.sh print
PYTHON_BIN=python bash training/example_train.sh config
STUDENT_MODEL=/path/to/student TEACHER_MODEL=/path/to/teacher bash training/example_train.sh run
```

The example builds one synthetic Parquet row in `output/`. The full training
data acquisition paths are under `data/`:

- Math: source and canonical training data are documented by
  [math-vault](https://github.com/Geraldxm/math-vault).
- Code: CodeR1-12K source plus the exact local conversion to the OPD training
  JSON are documented under `data/CodeR1-12K/`.
- Fact: the main manuscript's full NQ+HotpotQA training pool and its closed-book
  adapter are documented under `data/FactQA/`. NQ is training data
  here; no NQ evaluation benchmark is included.

The main manuscript's training-configuration table remains authoritative for
each reported run. Checkpoint download links will be added to the root README
after the releases are available.
