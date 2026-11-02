# Configuration

Python modules and the Bash drivers share one dotenv loader ([`chainuq/config.py`](../chainuq/config.py)). Variables
exported in the shell take precedence over `.env`, and relative paths resolve against the repository root.
`CHAINUQ_ENV_FILE` selects another dotenv file.

| Variable | Default | Purpose |
|---|---|---|
| `CHAINUQ_ARTIFACTS` | `artifacts` | generations, judge outputs, features, checkpoints, predictions, results |
| `CHAINUQ_POOL_ARTIFACTS` | `${CHAINUQ_ARTIFACTS}/pools` | artifact tree of the 6,000-question unlabeled pools |
| `CHAINUQ_ANALYSIS` | `${CHAINUQ_ARTIFACTS}/analysis` | evidence cache and Stage-2 analysis outputs |
| `CHAINUQ_LOGS` | `${CHAINUQ_ARTIFACTS}/logs` | driver and judge-server logs |
| `CHAINUQ_DATASETS` | `datasets` | prepared benchmark splits and question pools |
| `CHAINUQ_MODELS` | `models` | local model directories, named as in `registry.py` |
| `CHAINUQ_PAPER_DIR` | `latex` | LaTeX project that receives `table/` and `fig/` |
| `CHAINUQ_PYTHON`, `CHAINUQ_VLLM` | `python`, `vllm` | executables of the driver scripts |
| `CHAINUQ_GEN_GPU`, `CHAINUQ_JUDGE_GPU` | `0`, `1` | GPUs for generation/features/training and for the judge server |
| `CHAINUQ_JUDGE_PORT` | `8100` | judge server port |
| `CHAINUQ_WORKERS` | `30` | processes of the CPU analyses in `scripts/run_analysis.sh` |
| `CHAINUQ_ANALYSIS_JUDGE` | empty | judge whose verdict and P(true) replace the main judge's in `analysis.variants` and `analysis.information` |
| `CHAINUQ_ANALYSIS_SPLITS` | `test,test2` | held-out splits scored by `analysis.variants` |

The dotenv syntax supports `KEY=value`, `export KEY=value`, single and double quotes, comment lines, trailing
comments and `${NAME}` expansion of previously defined variables. Values are data: shell substitutions are never
executed, and malformed lines fail with their line number. The scripts bootstrap the loader with the Python on
`PATH` (or an exported `CHAINUQ_PYTHON`); a `CHAINUQ_PYTHON` set inside `.env` runs the stages.

Library settings such as `HF_HOME`, `HF_HUB_OFFLINE`, `TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS` and
`PYTORCH_CUDA_ALLOC_CONF` can also live in `.env`. Leave `HF_HUB_OFFLINE=0` for the first data preparation and set
it to `1` once models and datasets are cached.

Point a single command at another tree with a shell export:

```bash
CHAINUQ_ARTIFACTS=/scratch/chainuq python -m chainuq.pipeline.audit --backbone llama
```

Keep `.env` private; commit portable defaults only to `.env.example`.
