# ChEMBL loader: fetch drug-target interaction datasets
# Queries ChEMBL REST API for bioactivity data (IC50 values)


import logging
import requests
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult
from pipeline import stats

logger = logging.getLogger(__name__)

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
MIN_COMPOUNDS = 100  

def list_candidates(max_candidates=50):
	candidates = []
	seen_targets = set()

	# get human protein targets
	try:
		url = f"{CHEMBL_BASE_URL}/targets.json?organism=Homo_sapiens&limit=200"
		response = requests.get(url, timeout=15)
		targets_data = response.json()
		targets = targets_data.get('target', [])

		logger.info(f"Found {len(targets)} human targets")

	except Exception as e:
		logger.error(f"Failed to fetch targets: {e}")
		return candidates

	# filter targets with enough bioactivity data
	for target in targets:
		if len(candidates) >= max_candidates:
			break

		target_id = target.get('target_chembl_id')
		target_name = target.get('target_name', 'Unknown')

		if not target_id or target_id in seen_targets:
			continue

		seen_targets.add(target_id)

		# get bioactivity count for this target
		url = f"{CHEMBL_BASE_URL}/activity.json?target_id={target_id}&limit=1"
		response = requests.get(url, timeout=10)
		act_data = response.json()

		# get total count from page metadata
		total_count = act_data.get('page_meta', {}).get('total_count', 0)

		# skip if not enough compounds
		if total_count < MIN_COMPOUNDS:
			continue

		logger.info(f"{target_name}: {total_count} bioactivities")

			# create candidate
		candidate = CandidateInfo(
				id=target_id,
				source="chembl",
				name=target_name,
				n_samples=total_count,
				n_features=2000,  # approximate molecular descriptors
				task_type="regression",  # IC50 prediction
				licence="CC0",
				url=f"https://www.ebi.ac.uk/chembl/target/{target_id}",
				metadata={"target_type": target.get('target_type', 'protein')},
				domain="chemical",
			)

		candidates.append(candidate)

	return candidates
