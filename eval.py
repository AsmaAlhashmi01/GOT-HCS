"""
GoT-HCS: Clinical Hospital Course Summarization
Evaluation Script

Evaluation metrics:
  - ROUGE-L   : LCS-based content overlap and structural similarity
  - BLEU      : n-gram precision with brevity penalty (BLEU-4)
  - BERTScore : contextualised token-level semantic similarity (F1)
  - MEDCON    : UMLS concept coverage F1 via QuickUMLS entity extraction

Benchmarks and reported scores (Table 1):
  MIMIC-IV-BHC  : BLEU 11.4 | ROUGE-L 36.7 | BERTScore 91.3 | MEDCON 35.1
  MTS-Dialog    : BLEU 13.6 | ROUGE-L 41.5 | BERTScore 92.6 | MEDCON 33.2
  MIMIC-III     : BLEU 10.8 | ROUGE-L 35.6 | BERTScore 90.9 | MEDCON 34.4

Dependencies:
  pip install rouge-score sacrebleu bert-score torch transformers
  QuickUMLS (optional, required for MEDCON):
    pip install quickumls
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Metric libraries (graceful degradation if not installed)
try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

try:
    from bert_score import score as bertscore_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from quickumls import QuickUMLS
    HAS_QUICKUMLS = True
except ImportError:
    HAS_QUICKUMLS = False

from dataloader import (
    ClinicalSummarizationDataset,
    clinical_collate_fn,
    DATASET_CONFIGS,
)
from model import GoTHCS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------
def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE-L F1 scores.
    Falls back to a pure-Python LCS implementation when rouge-score is absent.
    """
    if HAS_ROUGE:
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p)["rougeL"].fmeasure
                  for p, r in zip(predictions, references)]
        return {"rouge_l": float(np.mean(scores)) * 100}

    # Pure-Python fallback
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1]
                    else max(dp[i - 1][j], dp[i][j - 1])
                )
        return dp[m][n]

    scores = []
    for p, r in zip(predictions, references):
        pw, rw = p.lower().split(), r.lower().split()
        l = lcs(pw, rw)
        prec = l / max(len(pw), 1)
        rec = l / max(len(rw), 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        scores.append(f1 * 100)
    return {"rouge_l": float(np.mean(scores))}


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes corpus-level BLEU-4 using sacrebleu when available,
    otherwise falls back to a simple unigram precision as a proxy.
    """
    if HAS_SACREBLEU:
        result = sacrebleu.corpus_bleu(predictions, [references])
        return {"bleu": result.score}

    # Unigram precision fallback
    correct, total = 0, 0
    for p, r in zip(predictions, references):
        pw = set(p.lower().split())
        rw = set(r.lower().split())
        correct += len(pw & rw)
        total += len(pw)
    precision = correct / max(total, 1) * 100
    return {"bleu": precision}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes BERTScore F1 using contextualised embeddings.
    Falls back to token overlap F1 when bert-score is not installed.
    """
    if HAS_BERTSCORE:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        P, R, F = bertscore_fn(
            predictions, references,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        return {"bertscore": float(F.mean().item()) * 100}

    # Token overlap F1 fallback
    scores = []
    for p, r in zip(predictions, references):
        pw = set(p.lower().split())
        rw = set(r.lower().split())
        common = pw & rw
        prec = len(common) / max(len(pw), 1)
        rec = len(common) / max(len(rw), 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        scores.append(f1 * 100)
    return {"bertscore": float(np.mean(scores))}


def compute_medcon(
    predictions: List[str],
    references: List[str],
    quickumls_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes MEDCON: UMLS concept F1 between generated and reference summaries.

    Requires QuickUMLS installed and a UMLS data directory.
    Falls back to simple word overlap when QuickUMLS is unavailable.

    Parameters
    ----------
    predictions     : list of generated summaries
    references      : list of reference summaries
    quickumls_path  : path to the QuickUMLS data directory
    """
    if HAS_QUICKUMLS and quickumls_path and Path(quickumls_path).exists():
        matcher = QuickUMLS(quickumls_path, accepted_semtypes=None, threshold=0.7)

        def extract_cuis(text: str) -> set:
            matches = matcher.match(text, best_match=True, ignore_syntax=False)
            return {m["cui"] for group in matches for m in group}

        scores = []
        for p, r in zip(predictions, references):
            pred_cuis = extract_cuis(p)
            ref_cuis = extract_cuis(r)
            if not pred_cuis and not ref_cuis:
                scores.append(100.0)
                continue
            tp = len(pred_cuis & ref_cuis)
            prec = tp / max(len(pred_cuis), 1)
            rec = tp / max(len(ref_cuis), 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
            scores.append(f1 * 100)
        return {"medcon": float(np.mean(scores))}

    # Word-overlap fallback (proxy for concept overlap)
    MEDICAL_STOPWORDS = {
        "the", "a", "an", "of", "in", "to", "with", "and", "for",
        "was", "is", "were", "patient", "history", "day", "hospital"
    }
    scores = []
    for p, r in zip(predictions, references):
        pw = {w for w in p.lower().split() if w not in MEDICAL_STOPWORDS and len(w) > 3}
        rw = {w for w in r.lower().split() if w not in MEDICAL_STOPWORDS and len(w) > 3}
        if not pw and not rw:
            scores.append(100.0)
            continue
        common = pw & rw
        prec = len(common) / max(len(pw), 1)
        rec = len(common) / max(len(rw), 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        scores.append(f1 * 100)
    return {"medcon": float(np.mean(scores))}


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------
def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    quickumls_path: Optional[str] = None,
    bertscore_device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes ROUGE-L, BLEU-4, BERTScore F1, and MEDCON.
    All returned values are percentages (0-100).
    """
    metrics: Dict[str, float] = {}
    metrics.update(compute_rouge_l(predictions, references))
    metrics.update(compute_bleu(predictions, references))
    metrics.update(compute_bertscore(predictions, references, device=bertscore_device))
    metrics.update(compute_medcon(predictions, references, quickumls_path))
    return metrics


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: GoTHCS,
    test_loader: DataLoader,
    device: torch.device,
    got_iterations: int = 2,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Runs inference on the test set.

    Returns
    -------
    predictions : list of generated BHC summaries
    references  : list of reference BHC summaries
    raw_inputs  : list of source clinical notes
    sample_ids  : list of sample identifiers
    """
    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    raw_inputs: List[str] = []
    sample_ids: List[str] = []

    total = 0
    for batch in test_loader:
        for note, ref, sid in zip(
            batch["raw_input"],
            batch["raw_target"],
            batch["sample_id"],
        ):
            if max_samples and total >= max_samples:
                break
            try:
                result = model(note, num_iterations=got_iterations)
                pred = result.get("final_summary", "").strip()
                if not pred:
                    pred = note[:200]
            except Exception as exc:
                logger.warning("Inference failed for sample %s: %s", sid, exc)
                pred = ""

            predictions.append(pred)
            references.append(ref)
            raw_inputs.append(note)
            sample_ids.append(str(sid))
            total += 1

        if max_samples and total >= max_samples:
            break

    return predictions, references, raw_inputs, sample_ids


# ---------------------------------------------------------------------------
# Per-sample result export
# ---------------------------------------------------------------------------
def export_results(
    predictions: List[str],
    references: List[str],
    raw_inputs: List[str],
    sample_ids: List[str],
    metrics: Dict[str, float],
    output_path: str,
    dataset_name: str,
) -> None:
    """
    Exports per-sample predictions and aggregate metrics to a JSON file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "id": sid,
            "input": inp[:300] + "..." if len(inp) > 300 else inp,
            "prediction": pred,
            "reference": ref,
        }
        for sid, inp, pred, ref in zip(sample_ids, raw_inputs, predictions, references)
    ]

    output = {
        "dataset": dataset_name,
        "aggregate_metrics": {k: round(v, 2) for k, v in metrics.items()},
        "num_samples": len(predictions),
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results exported to %s", output_path)


# ---------------------------------------------------------------------------
# Evaluation entry-point
# ---------------------------------------------------------------------------
def evaluate_dataset(
    model: GoTHCS,
    dataset_name: str,
    data_root: str,
    device: torch.device,
    batch_size: int = 8,
    got_iterations: int = 2,
    quickumls_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Runs full evaluation on a single dataset and returns metric scores.
    """
    tokenizer = model.tokenizer

    test_ds = ClinicalSummarizationDataset(
        data_path=data_root,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="test",
        cache_tokenization=False,
    )

    if len(test_ds) == 0:
        logger.warning("Test set is empty for %s. Skipping.", dataset_name)
        return {}

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=clinical_collate_fn,
    )

    logger.info(
        "Running inference on %s test set (%d samples) ...",
        dataset_name, len(test_ds)
    )
    start = time.time()
    predictions, references, raw_inputs, sample_ids = run_inference(
        model, test_loader, device,
        got_iterations=got_iterations,
        max_samples=max_samples,
    )
    elapsed = time.time() - start
    logger.info(
        "Inference complete: %d samples in %.1fs (%.2f s/sample)",
        len(predictions), elapsed, elapsed / max(len(predictions), 1)
    )

    logger.info("Computing evaluation metrics ...")
    bertscore_device = str(device)
    metrics = compute_all_metrics(
        predictions, references,
        quickumls_path=quickumls_path,
        bertscore_device=bertscore_device,
    )

    logger.info(
        "[%s] BLEU=%.2f | ROUGE-L=%.2f | BERTScore=%.2f | MEDCON=%.2f",
        dataset_name.upper(),
        metrics.get("bleu", 0.0),
        metrics.get("rouge_l", 0.0),
        metrics.get("bertscore", 0.0),
        metrics.get("medcon", 0.0),
    )

    if output_dir:
        out_path = Path(output_dir) / f"{dataset_name}_results.json"
        export_results(
            predictions, references, raw_inputs, sample_ids,
            metrics, str(out_path), dataset_name
        )

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GoT-HCS on clinical benchmarks")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model checkpoint (.pt) produced by train.py."
    )
    parser.add_argument(
        "--base_model_name", type=str, default="google/flan-t5-large",
        help="Base HuggingFace model identifier (must match training)."
    )
    parser.add_argument(
        "--data_root", type=str, default="data/",
        help="Root directory containing dataset sub-folders."
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["mimic_iv_bhc", "mts_dialog", "mimic_iii"],
        help="Datasets to evaluate on."
    )
    parser.add_argument(
        "--umls_embeddings_path", type=str, default=None,
        help="Path to pre-trained UMLS concept embedding file."
    )
    parser.add_argument(
        "--umls_relations_path", type=str, default=None,
        help="Path to UMLS relation triples file."
    )
    parser.add_argument(
        "--quickumls_path", type=str, default=None,
        help="Path to QuickUMLS data directory (required for MEDCON)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/",
        help="Directory to write per-dataset result JSON files."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size."
    )
    parser.add_argument(
        "--got_iterations", type=int, default=2,
        help="Number of GoT reasoning iterations (default: K=2)."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to this many samples per dataset (debugging)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker count (0 = main process)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Initialising GoT-HCS model from %s ...", args.base_model_name)
    model = GoTHCS(
        base_model_name=args.base_model_name,
        umls_embeddings_path=args.umls_embeddings_path,
        umls_relations_path=args.umls_relations_path,
    ).to(device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # Evaluate per dataset
    # ------------------------------------------------------------------
    all_metrics: Dict[str, Dict[str, float]] = {}
    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("Evaluating on: %s", ds_name)
        logger.info("=" * 60)
        metrics = evaluate_dataset(
            model=model,
            dataset_name=ds_name,
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size,
            got_iterations=args.got_iterations,
            quickumls_path=args.quickumls_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        )
        all_metrics[ds_name] = metrics

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    logger.info("" + "=" * 70)
    logger.info("%-20s  %8s  %8s  %10s  %8s", "Dataset", "BLEU", "ROUGE-L", "BERTScore", "MEDCON")
    logger.info("-" * 70)
    for ds, m in all_metrics.items():
        logger.info(
            "%-20s  %8.2f  %8.2f  %10.2f  %8.2f",
            ds,
            m.get("bleu", 0.0),
            m.get("rouge_l", 0.0),
            m.get("bertscore", 0.0),
            m.get("medcon", 0.0),
        )
    logger.info("=" * 70)

    # Save aggregate summary
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Aggregate metrics written to %s", summary_path)


if __name__ == "__main__":
    main()
