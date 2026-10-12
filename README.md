# A Global View of How LLMs Value Human Time

In this repository, we explore how LLMs evaluate the value of people's waiting time from a global perspective.


## Main Contribution
(i) We offer a systematic, global-scale evaluation on the trade-off between monetary compensation and waiting time across different LLMs.

(ii) We analyse our results against economic indicators and a human-derived patience score to compare the LLM-provided answers.

(iii) We uncover systematic geographic patterns across five LLMs and discuss the potential impact of these patterns. 

## Main Repository Structure

| Folder | Description | Subfolders |
|---|---|---|
| `notebooks/` | Analysis and experiment notebooks | |
| `results/` | Generated findings and outputs | |
| `tradeoff_data/` | Scripts that run the experiments and save the raw LLM responses (`data.pkl`) | |
| `tradeoff_table/` | Script that loads the saved raw responses and computes the transition values (compensations) (`LR_results.pkl`) | |
| `config/` | Configuration files | |
| `utils/` | Core library: model initialization, experiment execution, and the logic for computing transition values | |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

