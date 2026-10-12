# PDM

Official implementation of **PDM: Prototype-Guided Dual Magic-Cube Learning
for Class-Imbalanced Semi-Supervised Medical Image Segmentation**.

PDM contains two independently initialized MagicNet students. Each branch owns
an EMA teacher and a branch-specific multi-prototype bank. On unlabeled voxels
where the students disagree, prediction reliability and prototype support are
combined to select the teaching direction. The selected peer distribution
provides soft KL supervision, while reliable peer pseudo-labels provide
feature-space prototype correction. During inference, only the two students
are retained and their logits are averaged before softmax.

## Repository structure

```text
PDM/
├── train_DualMagic_proto_single_gpu.py        # main PDM training entry
├── prototype_guidance.py                      # prototype memory and guidance
├── pdm_runtime.py                            # shared training/runtime utilities
├── GALoss.py                                  # GACE and GADice
├── pdm_schedules.py                          # directional-weight schedule
├── dataloaders/
├── networks/
├── utils/
├── data/                                      # local data; ignored by Git
└── requirements_galoss_no_torch.txt
```

Only the final PDM training entry and its dependencies are included. Shared
runtime functions and the cosine weight schedule live in `pdm_runtime.py` and
`pdm_schedules.py`; legacy training entries and experimental loss classes are
not included. The `utils/test_*.py` files are required by validation and final
testing during training, not separate ablation scripts.

## Environment

The code was tested in the following environment:

| Component | Tested version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| NumPy | 2.4.4 |
| SciPy | 1.18.0 |
| h5py | 3.16.0 |
| nibabel | 5.4.2 |
| MedPy | 0.5.2 |
| scikit-image | 0.26.0 |
| surface-distance | 0.1 |
| tensorboardX | 2.6.5 |
| tqdm | 4.69.1 |
| einops | 0.8.2 |
| timm | 1.0.28 |

Exact package versions are not mandatory. A new Conda environment can be
created as follows; replace `name` with any environment name:

```bash
conda create -n name python=3.12 -y
conda activate name
```

Install a PyTorch/torchvision build compatible with the CUDA driver on the
machine. The generic command below installs the default build available from
the configured package index; the command from the official PyTorch selector
is recommended when a specific CUDA build is required.

```bash
python -m pip install torch torchvision
```

Install the remaining dependencies. `timm` is installed without dependency
resolution so that it cannot replace an already configured PyTorch stack.

```bash
python -m pip install -r requirements_galoss_no_torch.txt
python -m pip install --no-deps timm
```

## Data preparation

Data and preparation instructions can be obtained from
[GALoss](https://github.com/cicailalala/GALoss) and
[DHC](https://github.com/xmed-lab/DHC). Follow their dataset preparation
instructions and obtain the data under their respective licenses.

The entire `data/` directory, including image volumes and split files, is
ignored by Git. Place the prepared data and the corresponding Synapse and
AMOS split text files inside this repository as follows:

```text
PDM/data/
├── Synapse/
│   ├── 0001.h5
│   ├── 0002.h5
│   └── ...
├── Synapse_npy/
│   └── splits/
│       ├── labeled_10p.txt
│       ├── unlabeled_10p.txt
│       ├── labeled_20p.txt
│       ├── unlabeled_20p.txt
│       ├── labeled_40p.txt
│       ├── unlabeled_40p.txt
│       ├── eval.txt
│       └── test.txt
├── AMOS/
│   ├── amos_0001_image.npy
│   ├── amos_0001_label.npy
│   └── ...
└── amos_splits/
    ├── labeled_5p.txt
    ├── unlabeled_5p.txt
    ├── eval.txt
    └── test.txt
```

All commands below must be executed from the `PDM` directory. Consequently,
the default relative paths resolve to `PDM/data`, `PDM/model`, and `PDM/logs`.

```bash
cd /path/to/PDM
mkdir -p model logs
```

The main training entry reads the released DHC/SKCDF Synapse partition directly
from `data/Synapse_npy/splits/`; the former GALoss hard-coded partition is not
included. `--labelnum 2`, `4`, and `8` select the 10%, 20%, and 40%
labeled lists, respectively. The paper setting is `--labelnum 4`. AMOS reads
the prepared text files in `data/amos_splits/`.

## Single-GPU training

One PDM experiment uses one GPU. Both students are trained on the same device
with sequential forward/backward construction to control 3D activation memory.

### Synapse, DHC/SKCDF 20% labeled split

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_DualMagic_proto_single_gpu.py \
  --dataset_name Synapse \
  --root_path ./data/Synapse/ \
  --synapse_split_path ./data/Synapse_npy/splits \
  --labelnum 4 \
  --seed 42 \
  --device 0 \
  --num_workers 2 \
  --max_iteration 17000 \
  --aux_max_weight 0.02 \
  --prototype_count 3 \
  --save_path ./model \
  --exp PDM_K3_17k_s42 \
  > logs/pdm_synapse_k3_17k_s42.log 2>&1 &
```

### AMOS, 5% labeled split

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_DualMagic_proto_single_gpu.py \
  --dataset_name AMOS \
  --root_path ./data/AMOS/ \
  --amos_split_path ./data/amos_splits \
  --labelnum 10 \
  --seed 1337 \
  --device 0 \
  --num_workers 2 \
  --max_iteration 60000 \
  --aux_max_weight 0.02 \
  --prototype_count 3 \
  --save_path ./model \
  --exp PDM_K3_60k_s1337 \
  > logs/pdm_amos_k3_60k_s1337.log 2>&1 &
```

The commands use 17,000 iterations for Synapse and 60,000 for AMOS. Both
students are updated once per iteration. Validation selects the best paired
checkpoint within the run; final testing evaluates Student A, Student B, and
their logit ensemble using that pair.

Use a unique `--exp` value for every run. The training script intentionally
refuses to write into a nonempty experiment directory, preventing accidental
checkpoint replacement.

## Main defaults

| Parameter | Value |
|---|---:|
| Number of prototypes per class | 3 |
| Prototype momentum | 0.99 |
| Prototype temperature | 0.1 |
| Samples per class and update stage | 256 |
| Labeled prototype loss weight | 0.02 |
| Prototype disagreement loss weight | 0.02 |
| Prototype guidance strength | 0.35 |
| Class-wise confidence quantile | 0.7 |
| Minimum prototype-update confidence | 0.25 |
| Minimum prototype teacher score | 0.45 |
| Maximum directional KL weight | 0.02 |

At inference, EMA teachers, prototype banks, cube-location supervision, and
direction selection are discarded. The saved Student-A and Student-B logits
are averaged before softmax.
