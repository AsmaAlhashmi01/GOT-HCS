# GoT-HCS: Graph-of-Thoughts Enhanced Hierarchical Reasoning for Automated Clinical Hospital Course Summarization

A multi-stage clinical summarization framework that models hospital course documentation as a directed acyclic graph of clinical reasoning units. The pipeline integrates BioClinicalBERT-based entity extraction, Graph Attention Networks (GAT), UMLS knowledge augmentation, and FLAN-T5-Large with iterative Graph-of-Thoughts refinement to generate coherent Brief Hospital Course (BHC) summaries from heterogeneous EHR inputs. GoT-HCS outperforms GPT-4 by 6.5 BLEU and 3.8 MEDCON points on MIMIC-IV-BHC while maintaining a 6.8 s/summary inference latency on a single NVIDIA A100.

***

## Architecture

<img width="1966" height="642" alt="architecture" src="https://github.com/user-attachments/assets/b52a1b39-5a0e-4a8e-9abc-33906b598d72" />


***

## Datasets

All datasets require credentialed PhysioNet access. Complete the relevant Data Use Agreement before downloading.

| Dataset | Access | Task | Train / Val / Test |
|---|---|---|---|
| MIMIC-IV-BHC | https://physionet.org/content/mimic-iv-bhc/ | BHC summarization | 18,000 / 2,000 / 2,000 |
| MIMIC-III | https://physionet.org/content/mimiciii/ | Discharge summarization | 8,000 / 1,000 / 1,000 |
| MTS-Dialog | https://github.com/abachaa/MTS-Dialog | Dialogue summarization | 1,600 / 200 / 200 |

Each split must be stored as a `.jsonl` file with `"input"` (clinical note) and `"output"` (reference summary) fields, placed under `data/{dataset_name}/`.

***

## Environment

```bash
conda create -n got-hcs python=3.10
conda activate got-hcs

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install transformers==4.40.0 torch-geometric rouge-score sacrebleu bert-score sentencepiece accelerate

# MEDCON metric (requires UMLS license: https://www.nlm.nih.gov/research/umls/)
pip install quickumls
python -m quickumls.install /path/to/umls/data /path/to/quickumls_db
```

***

## Training

```bash
python train.py \
  --data_root data/ \
  --base_model_name google/flan-t5-large \
  --datasets mimic_iv_bhc mts_dialog mimic_iii \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --batch_size 16 \
  --max_epochs 30 \
  --warmup_ratio 0.05 \
  --fp16 \
  --output_dir checkpoints/
```

To include UMLS knowledge graph augmentation, add:

```bash
  --umls_embeddings_path /path/to/umls_embeddings.json \
  --umls_relations_path /path/to/umls_relations.tsv
```

***

## Evaluation

```bash
python eval.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --base_model_name google/flan-t5-large \
  --data_root data/ \
  --datasets mimic_iv_bhc mts_dialog mimic_iii \
  --quickumls_path /path/to/quickumls_db \
  --output_dir eval_results/
```

Results are written to `eval_results/summary_metrics.json` and per-dataset JSON files containing per-sample predictions and aggregate BLEU, ROUGE-L, BERTScore, and MEDCON scores.
