# A Global View of How LLMs Value Human Time

In this repository, we explore how LLMs evaluate the value of people's waiting time from a global perspective.

A framework for this work:
![schematic](schematic.png)

## Main Contribution
(i) We offer a systematic, global-scale evaluation on the trade-off between monetary compensation and waiting time across different LLMs.

(ii) We analyse our results against economic indicators and a human-derived patience score to compare the LLM-provided answers.

(iii) We uncover systematic geographic patterns across five LLMs and discuss the potential impact of these patterns. 

## Main Repository Structure

| Folder | Description | Subfolders |
|---|---|---|
| `notebooks/` | Analysis and experiment notebooks | |
| `results/` | Generated findings and outputs | |
| `figures/` | Visualizations of results | 1. `prior/`: example results from different experiment setups<br>2. `prompt_variation/`: example results on different prompt variations<br>3. `scatterplot/`: all linear regression plots across the three indicators under three prompts<br>4. `world_map/`: all LLMs' responses visualized on world maps under three prompts |
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

