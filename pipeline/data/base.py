"""Base classes for the data loading layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """
    This is a class that represents a dataset that is returned by the data loaders.
    To make it more general, we can use a dataclass to represent the dataset,
    which can contain both the actual data and the metadata.
    It contains the following attributes:
    id: identifier of the datasetsets, e.g. "OpenML-1578", "TCGA-LGG-GEX", "GEO-GSE12345".
    source: source of each dataset (openml, tcga, geo_array...).
    name: name of the actual dataset
    x: feature matrix (samples x features)
    y: target variable (series of length n_samples)
    task_type: classification or regression
    metadata: actual metadata coming from the fetching phase, which can be used for the soft rules.
    It can contain information such as the organism, the platform, the licence, etc.
    domain: high-level category tag — "biomedical" (clinical / cancer / patient),
        "biological" (broader life sciences, e.g. gene expression studies), or
        "general" (everything else, e.g. tabular ML benchmarks).
    """

    id: str
    source: str
    name: str
    X: pd.DataFrame
    y: pd.Series
    task_type: str
    metadata: dict
    domain: str = "general"


@dataclass(frozen=True)
class CandidateInfo:
    """Lightweight metadata record from the scraping phase (no actual data)."""

    id: str
    source: str
    name: str
    n_samples: int | None
    n_features: int | None
    task_type: str
    licence: str
    url: str
    metadata: dict
    domain: str = "general"


class DataSource(Protocol):
    """Protocol that every loader must implement."""

    def list_candidates(self) -> list[CandidateInfo]:
        """Scrape metadata and return candidate records that pass metadata-level filters."""
        ...

    def fetch(self, candidate: CandidateInfo) -> Dataset | None:
        """Download actual data, run hard rules. Returns None if hard rules fail."""
        ...
