# A2: reject synthetic / artificial datasets
# TCGA and GEO are always real data so they auto-pass
# for OpenML we scan the name and tags for suspicious keywords
from __future__ import annotations

import re

from pipeline.hard_rules.base import RuleResult

# keywords that suggest a dataset is not real for regex
SYNTHETIC_KEYWORDS = re.compile(
    r"synthetic|artificial|generated|simulated|random|"
    r"toy|demo|dummy|mock|fake|bogus|"
    r"monte.?carlo|"
    r"make_(classification|regression|blobs|moons|circles|swiss_roll)",
    re.IGNORECASE,
)

# these sources are real data by definition
# match loader source prefixes: "tcga", "geo_array", "geo_rnaseq"
REAL_SOURCE_PREFIXES = ("tcga", "geo")

def check_data(**_kwargs):
    return RuleResult(rule="A2", passed=True)


def check_metadata(source="", name="", metadata=None, **_kwargs):
    # tcga and geo are always real data
    if source.lower().startswith(REAL_SOURCE_PREFIXES):
        return RuleResult(rule="A2", passed=True)

    # check the dataset name
    if SYNTHETIC_KEYWORDS.search(name):
        return RuleResult(rule="A2", passed=False, reason=f"name looks synthetic: '{name}'")

    # check tags espicially for openml
    tags = (metadata or {}).get("tags", [])
    for tag in tags:
        if SYNTHETIC_KEYWORDS.search(str(tag)):
            return RuleResult(rule="A2", passed=False, reason=f"tag looks synthetic: '{tag}'")

    return RuleResult(rule="A2", passed=True)
