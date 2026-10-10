# SpecSlot

This repository contains the released SpecSlot implementation and the vanilla
JiT baselines used in the paper. It supports class-conditional ImageNet
training/evaluation and the text-to-image (T2I) protocol used in the scaling
experiment.

## Environment

Use Python 3.10 on a Linux machine with an NVIDIA GPU and CUDA-compatible
PyTorch wheels. The single environment covers ImageNet, T2I training, FID/IS,
GenEval, and DPG-Bench:

```bash
conda create -n specslot python=3.10 -y
conda activate specslot
python -m pip install "uv==0.12.21"
uv sync --locked --extra eval
source .venv/bin/activate
uv pip check
```

The optional `eval` extra builds the OpenMMLab components required by GenEval.
The complete T2I asset and evaluator setup is documented below. The first
FID/IS run may download Inception weights.

## ImageNet

ImageNet must use the standard `ImageFolder` layout:

```text
/path/to/your/ImageNet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Each launcher uses the paper's 120-epoch protocol, EMA1, 50-step Heun
sampling, and the reported CFG and bottleneck settings. It writes checkpoints,
`logs/train.log`, JSON training records, and the final evaluation under the
chosen output directory. Re-running a launcher resumes from
`checkpoint-last.pth`.

The ImageNet launcher fixes the reported geometry and CFG values; changing CFG
for a sweep requires using the research code or adjusting the corresponding
release-contract check.

### SpecSlot

```bash
bash scripts/run_specslot.sh /path/to/your/ImageNet /path/to/your/output 256 k2
bash scripts/run_specslot.sh /path/to/your/ImageNet /path/to/your/output 256 k4
bash scripts/run_specslot.sh /path/to/your/ImageNet /path/to/your/output 512 k2
bash scripts/run_specslot.sh /path/to/your/ImageNet /path/to/your/output 512 k4
```

| Resolution | Allocation | Window/stride | Slots | Bottleneck | CFG | Batch/GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 256 | K2 | 32/16 | 512 | 512 | 5.00 | 64 |
| 256 | K4 | 32/16 | 1,024 | 1,536 | 4.00 | 64 |
| 512 | K2 | 64/32 | 512 | 3,072 | 10.50 | 128 |
| 512 | K4 | 64/32 | 1,024 | 4,096 | 8.25 | 128 |

### JiT baselines

```bash
bash scripts/run_jit.sh /path/to/your/ImageNet /path/to/your/output 256 b8
bash scripts/run_jit.sh /path/to/your/ImageNet /path/to/your/output 256 b16
bash scripts/run_jit.sh /path/to/your/ImageNet /path/to/your/output 512 b16
bash scripts/run_jit.sh /path/to/your/ImageNet /path/to/your/output 512 b32
```

JiT uses its standard spatial patch tokenizer, CFG 2.90, EMA1, and the same
flow-matching and sampling protocol. Training batches are 64 images/GPU at
256px and 128 images/GPU at 512px with activation checkpointing.

## Text-to-image

The T2I protocol reads the BLIP3o WebDataset mixture (JourneyDB,
Long-Caption, and Short-Caption) from a directory containing tar shards:

```text
/path/to/your/T2I/
  **/*.tar
```

Images use deterministic short-side resize and center crop; captions are
encoded by a frozen OpenCLIP ViT-L/14 (`laion2b_s32b_b82k`, 77 tokens). A
learned 32-query adapter maps the text states to the global context slots.
SpecSlot and JiT use independent adapters with the same shape and optimizer
protocol. Caption dropout selects a precomputed in-memory null-text embedding.

The T2I launcher keeps the paper protocol: 400 epochs (400,000 updates),
1,000 steps/epoch,
batch/GPU 64, constant flow-matching learning rate, EMA, and 50-step Heun
sampling. SpecSlot uses CFG 4.0 and B=1,536. JiT uses CFG 2.9; `auto`
selects the reported B/8 baseline. Only the reported SpecSlot and vanilla JiT
baselines are included.

```bash
# SpecSlot, 256px
bash scripts/run_t2i_8gpu.sh \
  /path/to/your/T2I /path/to/your/t2i_runs/specslot-256 \
  specslot 256 64

# JiT-B/8, 256px
JIT_VARIANT=b8 bash scripts/run_t2i_8gpu.sh \
  /path/to/your/T2I /path/to/your/t2i_runs/jit-b8-256 \
  jit 256 64

# JiT-B/16, 256px
JIT_VARIANT=b16 bash scripts/run_t2i_8gpu.sh \
  /path/to/your/T2I /path/to/your/t2i_runs/jit-b16-256 \
  jit 256 64
```

Set `EVAL_INTERVAL=10` and `EVAL_ASSETS_DIR=/path/to/your/data/Eval` to run
generation and benchmark evaluation inside the training loop. Use
`EVAL_MAX_IMAGES=0` and `EVAL_DPG_MAX_PROMPTS=0` for the complete official
GenEval and DPG-Bench manifests. Each evaluation writes generated images,
`metrics.json`, and TensorBoard scalars below `evaluations/epoch-XXXX/`.

```bash
EVAL_INTERVAL=10 \
EVAL_ASSETS_DIR=/path/to/your/data/Eval \
EVAL_MAX_IMAGES=0 EVAL_DPG_MAX_PROMPTS=0 \
bash scripts/run_t2i_8gpu.sh \
  /path/to/your/T2I /path/to/your/t2i_runs/specslot-256-eval \
  specslot 256 64
```

## T2I evaluation setup

The optional `eval` dependency group installs the evaluator stack in the same
`.venv`. On a Linux NVIDIA machine, run the following from the repository root:

```bash
python -m pip install "uv==0.12.21"
uv sync --locked --extra eval
source .venv/bin/activate
```

The lockfile pins the training and evaluator stack, including Torch 2.5.1,
NumPy 1.24.4, OpenCLIP, Fairseq 0.12.2, MMCV 1.7.2, and MMDetection 2.28.2.
MMCV is built with its CUDA operators, so the machine needs a C++ compiler and
CUDA 12.x (`nvcc --version`). The pinned Torch wheel targets CUDA 12.4. The
launcher selects the repository `.venv` automatically and checks the training
runtime before launching workers. When evaluation is enabled, the training
process validates the evaluator imports and assets before the first run.
`DRY_RUN=1` previews a command without launching workers.

The release does not bundle the BLIP3o shards or benchmark assets. Prepare the
training shards and the evaluator directory separately, then pass their paths
to the launcher as shown above.

The evaluation asset directory should contain:

```text
/path/to/your/data/Eval/
  geneval/evaluation/evaluate_images.py
  geneval/prompts/evaluation_metadata.jsonl
  geneval_object_detector/
  ella/dpg_bench/compute_dpg_bench.py
  ella/dpg_bench/dpg_bench.csv
  modelscope/
```

GenEval uses the official Mask2Former configuration from the selected
MMDetection installation. If the installed package does not include its config
tree, set `GENEVAL_MODEL_CONFIG` to the matching file in a complete MMDetection
2.x checkout. DPG-Bench uses the ELLA evaluator and mPLUG VQA checkpoint.
FID/KID are optional and require an explicit real-image directory; the
repository never substitutes ImageNet statistics for that reference set.

The shared sampler follows the ImageNet endpoint-aware Heun schedule: a nominal
50-step rollout has 99 velocity evaluations, or 198 conditional/unconditional
forwards with CFG. Prompt order, image seeds, and manifests are deterministic.
The model interface is:

```python
x_hat = model(z, t, text_states, text_valid_mask)
```

To evaluate during training, `EVAL_INTERVAL=10` runs generation after every
completed interval. `EVAL_WORKERS=0` uses all training ranks; set it to `1` for
a single-worker smoke test. Rank-local images and raw scores are stored under
`evaluations/epoch-XXXX/shards/`; rank 0 verifies coverage, merges the shards,
recomputes the global benchmark averages, and writes `metrics.json`. Uneven
workloads synchronize through a CPU control group, and evaluator failures are
recorded rather than converted into zero scores.

```bash
EVAL_INTERVAL=10 EVAL_WORKERS=8 \
EVAL_ASSETS_DIR=/path/to/your/data/Eval \
EVAL_MAX_IMAGES=0 EVAL_NUM_SAMPLES=4 \
EVAL_DPG_MAX_PROMPTS=0 EVAL_BATCH_SIZE=8 \
REQUIRE_EVAL_METRICS=1 \
bash scripts/run_t2i_8gpu.sh \
  /path/to/your/data/T2I \
  /path/to/your/t2i_runs/specslot-256-eval \
  specslot 256 64
```

`EVAL_MAX_IMAGES=0` keeps all official GenEval prompts and
`EVAL_DPG_MAX_PROMPTS=0` keeps the complete DPG-Bench CSV. Use positive caps
only for smoke tests. Each evaluation writes generated images, raw evaluator
logs, `metrics.json`, and TensorBoard scalars. The DPG image directory is kept
separate from the GenEval directory so benchmark file layouts cannot interfere.

Training logs are written to `logs/train.log`, `logs/steps.jsonl`,
`train.jsonl`, and `tensorboard/`. Use `RESUME=auto` with an existing output
directory to restore its last checkpoint, optimizer, EMA, and logger state.

The T2I modules remain at the repository root with a consistent `t2i_` prefix;
moving them into a subpackage would require changing launcher imports and add
no reproducibility benefit. The release contains only the 256px T2I protocol
reported in the paper.

## License

MIT. Please cite the accompanying paper when using this implementation.
