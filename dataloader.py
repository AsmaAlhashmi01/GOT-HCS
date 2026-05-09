"""
GoT-HCS: Clinical Hospital Course Summarization
DataLoader Module

Supports three benchmark datasets:
  - MIMIC-IV-BHC  (train 18,000 / val 2,000 / test 2,000)
  - MTS-Dialog    (train  1,600 / val   200 / test   200)
  - MIMIC-III     (train  8,000 / val 1,000 / test 1,000)

Avg input length : MIMIC-IV-BHC ~2150 tok, MTS-Dialog ~1050 tok, MIMIC-III ~1850 tok
Avg output length: MIMIC-IV-BHC  ~185 tok, MTS-Dialog  ~145 tok, MIMIC-III  ~205 tok
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-level configuration
# ---------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, Dict] = {
    "mimic_iv_bhc": {
        "splits": {"train": 18000, "val": 2000, "test": 2000},
        "avg_input_tokens": 2150,
        "avg_output_tokens": 185,
        "avg_clinical_entities": 42,
        "max_input_length": 2048,
        "max_output_length": 300,
        "task": "bhc_summarization",
    },
    "mts_dialog": {
        "splits": {"train": 1600, "val": 200, "test": 200},
        "avg_input_tokens": 1050,
        "avg_output_tokens": 145,
        "avg_clinical_entities": 24,
        "max_input_length": 1024,
        "max_output_length": 256,
        "task": "dialogue_summarization",
    },
    "mimic_iii": {
        "splits": {"train": 8000, "val": 1000, "test": 1000},
        "avg_input_tokens": 1850,
        "avg_output_tokens": 205,
        "avg_clinical_entities": 37,
        "max_input_length": 2048,
        "max_output_length": 350,
        "task": "discharge_summarization",
    },
}


# ---------------------------------------------------------------------------
# Core Dataset class
# ---------------------------------------------------------------------------
class ClinicalSummarizationDataset(Dataset):
    """
    PyTorch Dataset for clinical summarization benchmarks.

    Each sample contains:
        input_ids          : tokenized clinical note (input to the encoder)
        attention_mask     : corresponding attention mask
        labels             : tokenized reference BHC summary (decoder targets)
        decoder_input_ids  : shifted right labels (teacher forcing)
        raw_input          : raw clinical note string (for graph construction)
        raw_target         : raw reference summary string (for metric computation)
        sample_id          : unique identifier per sample
        dataset_name       : originating dataset tag
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        dataset_name: str = "mimic_iv_bhc",
        split: str = "train",
        max_input_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        cache_tokenization: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.cache_tokenization = cache_tokenization

        assert self.dataset_name in DATASET_CONFIGS, (
            f"Unrecognised dataset: {self.dataset_name}. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )
        cfg = DATASET_CONFIGS[self.dataset_name]
        self.max_input_length = max_input_length or cfg["max_input_length"]
        self.max_output_length = max_output_length or cfg["max_output_length"]

        self.samples: List[Dict] = []
        self._load_data()

        if self.cache_tokenization:
            logger.info(
                "Pre-tokenizing %d samples for %s / %s ...",
                len(self.samples), self.dataset_name, split
            )
            self._tokenized_cache: List[Dict] = [
                self._tokenize(s) for s in self.samples
            ]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """
        Attempts to load data from standard file conventions.
        Accepted formats  : JSON lines (.jsonl) or JSON array (.json)
        Expected keys     : "input" (clinical note), "output" (reference BHC)
        Optional keys     : "id", "subject_id", "hadm_id"
        """
        candidates = [
            self.data_path / f"{self.split}.jsonl",
            self.data_path / f"{self.split}.json",
            self.data_path / self.dataset_name / f"{self.split}.jsonl",
            self.data_path / self.dataset_name / f"{self.split}.json",
        ]

        loaded = False
        for fpath in candidates:
            if fpath.exists():
                self.samples = self._read_file(fpath)
                logger.info(
                    "Loaded %d samples from %s", len(self.samples), fpath
                )
                loaded = True
                break

        if not loaded:
            logger.warning(
                "No data file found for dataset=%s split=%s under %s. "
                "Returning an empty dataset. Provide data files at: %s",
                self.dataset_name, self.split, self.data_path,
                [str(c) for c in candidates],
            )

    @staticmethod
    def _read_file(fpath: Path) -> List[Dict]:
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.suffix == ".jsonl":
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed line %d in %s: %s",
                            line_no, fpath, exc
                        )
            else:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = list(data.values())
        return samples

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _tokenize(self, sample: Dict) -> Dict:
        raw_input: str = sample.get("input", sample.get("text", ""))
        raw_target: str = sample.get("output", sample.get("summary", ""))
        sample_id: str = str(sample.get("id", sample.get("hadm_id", id(sample))))

        # Prepend task prefix following T5 convention
        encoder_input = f"Summarize clinical note: {raw_input}"

        enc = self.tokenizer(
            encoder_input,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                raw_target,
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels = dec["input_ids"].squeeze(0).clone()
        # Replace padding token id with -100 so it is ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "raw_input": raw_input,
            "raw_target": raw_target,
            "sample_id": sample_id,
            "dataset_name": self.dataset_name,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.cache_tokenization:
            return self._tokenized_cache[idx]
        return self._tokenize(self.samples[idx])


# ---------------------------------------------------------------------------
# Multi-dataset wrapper
# ---------------------------------------------------------------------------
class MultiDatasetLoader:
    """
    Convenience wrapper that instantiates train / val / test DataLoaders
    for one or more clinical benchmark datasets simultaneously.

    Usage:
        loader = MultiDatasetLoader(
            data_root="data/",
            tokenizer=tokenizer,
            datasets=["mimic_iv_bhc", "mts_dialog"],
            batch_size=16,
        )
        train_loader = loader.get_train_loader()
    """

    def __init__(
        self,
        data_root: str,
        tokenizer: AutoTokenizer,
        datasets: Optional[List[str]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_input_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
        cache_tokenization: bool = True,
        pin_memory: bool = True,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.datasets = datasets or list(DATASET_CONFIGS.keys())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.cache_tokenization = cache_tokenization
        self.pin_memory = pin_memory

        self._datasets: Dict[str, Dict[str, ClinicalSummarizationDataset]] = {}
        self._build_datasets()

    def _build_datasets(self) -> None:
        for ds_name in self.datasets:
            self._datasets[ds_name] = {}
            for split in ("train", "val", "test"):
                self._datasets[ds_name][split] = ClinicalSummarizationDataset(
                    data_path=str(self.data_root),
                    tokenizer=self.tokenizer,
                    dataset_name=ds_name,
                    split=split,
                    max_input_length=self.max_input_length,
                    max_output_length=self.max_output_length,
                    cache_tokenization=self.cache_tokenization,
                )

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        from torch.utils.data import ConcatDataset

        all_splits = [
            self._datasets[ds][split]
            for ds in self.datasets
            if len(self._datasets[ds][split]) > 0
        ]
        if not all_splits:
            raise RuntimeError(
                f"No samples found for split={split} across datasets {self.datasets}"
            )
        combined = ConcatDataset(all_splits) if len(all_splits) > 1 else all_splits[0]
        return DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(split == "train"),
            collate_fn=clinical_collate_fn,
        )

    def get_train_loader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader("val", shuffle=False)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader("test", shuffle=False)

    def get_single_dataset(
        self, dataset_name: str, split: str
    ) -> ClinicalSummarizationDataset:
        return self._datasets[dataset_name][split]


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------
def clinical_collate_fn(batch: List[Dict]) -> Dict:
    """
    Stacks tensor fields; preserves string fields as lists.
    Compatible with the GoTHCS model forward signature.
    """
    tensor_keys = {"input_ids", "attention_mask", "labels"}
    collated: Dict = {}
    for key in batch[0]:
        if key in tensor_keys:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated


# ---------------------------------------------------------------------------
# Standalone convenience factory
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_root: str,
    tokenizer_name_or_path: str = "google/flan-t5-large",
    datasets: Optional[List[str]] = None,
    batch_size: int = 16,
    max_input_length: int = 2048,
    max_output_length: int = 300,
    num_workers: int = 4,
    cache_tokenization: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    One-call factory for (train_loader, val_loader, test_loader).

    Parameters
    ----------
    data_root             : root directory containing dataset sub-folders
    tokenizer_name_or_path: HuggingFace tokenizer identifier
    datasets              : list of dataset names (default: all three)
    batch_size            : global batch size per GPU (paper: 16)
    max_input_length      : max encoder tokens (paper: up to 2048)
    max_output_length     : max decoder tokens (paper: ~300)
    num_workers           : DataLoader worker count
    cache_tokenization    : pre-tokenize entire split at init time

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    loader = MultiDatasetLoader(
        data_root=data_root,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        cache_tokenization=cache_tokenization,
    )
    return loader.get_train_loader(), loader.get_val_loader(), loader.get_test_loader()


# ---------------------------------------------------------------------------
# CLI / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GoT-HCS DataLoader smoke-test")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="mimic_iv_bhc")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--tokenizer", type=str, default="google/flan-t5-large"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = ClinicalSummarizationDataset(
        data_path=args.data_root,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        split=args.split,
        cache_tokenization=False,
    )
    print(f"Dataset size : {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys  :", list(sample.keys()))
        print("input_ids    :", sample["input_ids"].shape)
        print("labels       :", sample["labels"].shape)
        print("raw_input[:80]:", sample["raw_input"][:80])
        print("raw_target[:80]:", sample["raw_target"][:80])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=clinical_collate_fn,
        shuffle=False,
        num_workers=0,
    )
    if len(dataset) > 0:
        batch = next(iter(loader))
        print("Batch input_ids shape  :", batch["input_ids"].shape)
        print("Batch labels shape     :", batch["labels"].shape)
        print("Batch dataset_name     :", batch["dataset_name"])
