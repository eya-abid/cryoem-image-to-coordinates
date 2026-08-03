# Portable Training Templates

Copy `.env.example` to `.env`, set controlled data and output roots, then export the variables before using these commands. The templates preserve retained hyperparameters but cannot run without the corresponding controlled datasets and split manifests.

```bash
set -a
source .env
set +a
bash examples/run_her2_experimental_direct_512d.sh
```

Training outputs must remain outside the Git checkout. Inference should be run separately against the declared test split, followed by `cryoem-coords evaluate` and `cryoem-coords assignment`.
