# PGDN

**Prototype-Guided Diffusion Noise for Non-Overlapping Cross-Domain Sequential Recommendation**

PGDN transfers preference information between domains with disjoint user sets. Shared preference prototypes connect unpaired interaction histories, and a random-prefix diffusion task provides additional supervision from transitions within target-domain histories.

## Method

1. **Source-domain pretraining** uses Gaussian diffusion for history-conditioned next-item prediction. A shared encoder learns preference prototypes from source and target training histories. The prototype memory tracks support from both domains and stores means and second moments estimated from source histories.
2. **Target-domain training** initializes the denoiser with the source-pretrained parameters. In the transfer-guided branch, the full target history retrieves the most relevant shared prototypes. Their source statistics define a diagonal Gaussian from which a source preference is sampled. The model decodes the difference between the sampled preference and the target query and mixes it with Gaussian noise. In the target-specific branch, **random-prefix target diffusion** samples a cut within an observed history and uses the item immediately after the cut as the auxiliary target. This branch uses Gaussian noise and shares the denoiser and target item embeddings with the main branch.
3. **Inference** uses the full target history to construct the initial mixed noise. The reverse sampler holds the prototype component and its mixing coefficient fixed and adjusts the reverse mean at each step. Items are ranked by the dot product of their embeddings with the final denoised representation. The random-prefix objective is used during training only.

## Environment

The launchers use Bash 4 or later, GNU coreutils, and a CUDA-capable NVIDIA GPU. The paper reports experiments with Python 3.8, PyTorch 2.4.1, and an NVIDIA A40. Install NumPy and pandas in the same Python environment as CUDA-enabled PyTorch. See the [PyTorch 2.4.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v241) for CUDA build options.

## Data

The included Amazon review splits are preprocessed pandas DataFrames loaded from `data/<domain>/{train,val,test}.df`. In the paper, ratings of at least 4 become implicit feedback. Interactions are ordered chronologically, users with fewer than five interactions are removed, and histories contain up to eight items plus the next-item label. Each domain is split approximately 80/10/10.

| Directory | Train | Validation | Test | Item vocabulary |
| --- | ---: | ---: | ---: | ---: |
| `data/movie/` | 83,366 | 10,420 | 10,422 | 98,507 |
| `data/video/` | 16,438 | 2,054 | 2,056 | 23,978 |
| `data/toys/` | 34,833 | 4,354 | 4,355 | 113,989 |

Each row contains `seq` (an eight-position history left-padded with ID `0`), `len_seq` (history length), and `next` (the next-item ID). The launchers read these fixed splits directly.

## Run PGDN

The two launchers run Movies → Video and Movies → Toys, each with seeds 2020–2024. The paper also reports Movies → Clothing, Movies → Books, and Toys → Video.

> Training requires the shared Python result-summary helper referenced by `SUMMARY_SCRIPT` in the launchers. This file is pending in the current package. The launchers call it to verify each seed and summarize the run directory.

From the repository root in a Bash shell, set `PYTHON_BIN` to the Python executable for your environment and select the physical GPU for each task. The commands below select GPU 0.

```bash
PYTHON_BIN="$(command -v python)" VIDEO_GPU_ID=0 RUN_NAME=pgdn_video_5seeds \
  bash run_pgdn_video_5seeds.sh
PYTHON_BIN="$(command -v python)" TOYS_GPU_ID=0 RUN_NAME=pgdn_toys_5seeds \
  bash run_pgdn_toys_5seeds.sh
```

Set `DRY_RUN=1` before a command to preview the seeds, GPU, and settings without training.

| Launcher | Diffusion steps | Prototype loss weight | Random-prefix loss weight |
| --- | ---: | ---: | ---: |
| [Movies → Video](run_pgdn_video_5seeds.sh) | 16 | 0.5 | 0.75 |
| [Movies → Toys](run_pgdn_toys_5seeds.sh) | 128 | 0.01 | 0.75 |

Both launchers use Adam with learning rate 0.001, batch size 512, embedding dimension 128, 64 prototypes, and up to five retrieved prototypes. They sample one random prefix per eligible training history with a five-epoch weight warm-up. After source pretraining, the launchers rebuild the prototype memory by clustering source training histories and estimating target support from a fixed 20% subset of target training histories. The rebuilt memory and the noise decoder are frozen during target training.

The table records the supplied launchers' settings. Section III-A of the paper describes a 32-step setting. Section II describes soft-assignment and moving-average prototype memory updates and a trainable target-stage noise decoder.

## Evaluation and outputs

Both launchers use the `d2tcdr_legacy` evaluation protocol and the sampled-validation checkpoint selection rule. Test metrics are HR@K and NDCG@K at K = 5, 10, and 20. Sampled ranking uses one positive and 100 sampled negatives. Full ranking scores the configured item vocabulary and masks the padding ID. Evaluation uses complete batches and retains rows whose labels are absent from training.

Each launcher writes to `runs/<RUN_NAME>/`. The directory contains the run plan and configuration, code and data hashes, per-seed logs and status, source-pretraining checkpoints, and prototype-bank checkpoints. To continue an interrupted experiment, repeat the same command with `RESUME=1` and the original run name, GPU, Python path, code, and data. The launcher skips completed seeds and reruns incomplete seeds.

## Code

| File | Role |
| --- | --- |
| [src/main.py](src/main.py) | Loads the splits and runs both training stages. |
| [src/model.py](src/model.py) | Encodes histories, learns shared prototypes, and constructs prototype-guided noise. |
| [src/source_guided_prototypes.py](src/source_guided_prototypes.py) | Rebuilds the prototype memory used by the two launchers. |
| [src/diffurec.py](src/diffurec.py) | Implements the conditional denoiser and forward and reverse diffusion. |
| [src/target_transition.py](src/target_transition.py) | Samples target-history prefixes and their next-item labels. |
| [src/trainer.py](src/trainer.py) | Trains the two stages and computes ranking metrics. |

The conditional diffusion implementation follows [DiffuRec](https://github.com/WHUIR/DiffuRec).
