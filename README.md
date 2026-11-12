# Recommender Systems on H&M Data

A personal project comparing several recommendation approaches on the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) data.

I wanted to check whether complex neural models really beat classical methods on sparse data with
short purchase histories. In my experiments they didn't: the best result came from EASE, a simple
linear item-to-item model.

## What's inside

- H&M data preparation and filtering
- a temporal leave-last-out split that never uses future purchases
- one evaluation pipeline shared by all models
- ranking over the full catalogue of 35,411 items
- Recall, NDCG and MRR
- tests for the split, metrics, models and data leakage
- a small two-stage retrieve → rerank experiment

Models I tried:

- Popularity and Popularity Decay
- ItemKNN
- ALS and BPR
- EASE
- SASRec and GRU4Rec
- Mult-VAE
- a LightGBM reranker on top of EASE candidates

I also wrote training code and an adapter for LightGCN, but haven't added its results to the
comparison yet.

## Results

The main metric is Recall@20 on the validation split. For each user, the model ranks the whole
catalogue, and items the user has already bought are excluded.

| Model | Recall@20 |
|---|---:|
| Random | 0.0005 |
| GRU4Rec | 0.0118 |
| SASRec | 0.0142 |
| BPR | 0.0219 |
| Popularity | 0.0263 |
| Popularity Decay | 0.0282 |
| Mult-VAE | 0.0442 |
| ItemKNN | 0.0484 |
| ALS | 0.0491 |
| **EASE** | **0.0560** |

On this data the classical models did better than the sequential ones. My working guess is that for
H&M purchases, which items get bought together matters more than the exact order they are bought in.

I also tried a LightGBM reranker that takes the top 100 candidates from EASE and combines scores from
several models with item features. On the test split it raised Recall@20 from `0.0419` to `0.0550`.

These results hold only for this split, these model settings and the H&M data. I don't see them as
a general conclusion about classical versus neural recommenders.

## How evaluation works

For each user:

1. The last purchase of an item they hadn't bought before is the test target.
2. The one before it is the validation target.
3. Everything earlier goes into training.
4. Items the user has already bought are masked out.
5. Metrics are computed over the full catalogue, with no sampled negatives.

All models are evaluated by the same code in `src/eval/`.

## Running it

The project targets Python 3.11.

```bash
make install
```

Downloading the data needs a Kaggle account, accepted competition rules and a configured
`~/.kaggle/kaggle.json`.

```bash
source .venv/bin/activate
make prep
make split
```

To run an evaluation:

```bash
.venv/bin/python -m src.cli eval \
  --model popularity \
  --split val \
  --config configs/base.yaml
```

Available model names:

```text
random, popularity, popularity_decay, itemknn, als, bpr,
ease, sasrec, gru4rec, multivae, lightgcn
```

Some of the heavier models were trained on Kaggle. To evaluate them locally, put their checkpoints in
`artifacts/`. Data, model weights and run outputs are not stored in Git.

To run the tests:

```bash
make test
```

## Project structure

```text
configs/    experiment configs
scripts/    scripts for local and Kaggle runs
src/data/   data loading, preparation and the split
src/eval/   metrics and the shared evaluation pipeline
src/models/ model implementations and adapters
src/rerank/ features for the reranker
tests/      tests
```
