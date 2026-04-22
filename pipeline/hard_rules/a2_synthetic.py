# A2: reject synthetic / artificial datasets
# TCGA and GEO are always real data so they auto-pass
# for OpenML we scan the name and tags for suspicious keywords
from __future__ import annotations

import re

from pipeline.hard_rules.base import RuleResult

# keywords that suggest a dataset is not real for regex
SYNTHETIC_KEYWORDS = re.compile(
    r"synthetic|artificial|generated|simulated|random",
    re.IGNORECASE,
)

# these sources are real data by definition
REAL_SOURCES = {"tcga", "geo"}


def check_metadata(source="", name="", metadata=None, **_kwargs): #works with regex of the metadata. -> only picks out of the kwargs source and name and whats in the metadata chekc
    # tcga and geo are always real
    if source.lower() in REAL_SOURCES:
        return RuleResult(rule="A2", passed=True)

    # check the dataset name
    if SYNTHETIC_KEYWORDS.search(name):
        return RuleResult(rule="A2", passed=False, reason=f"name looks synthetic: '{name}'")

   
    return RuleResult(rule="A2", passed=True)
