# CM3604 Deep Learning Group Coursework — Fact-Checking ChatGPT Outputs

Group coursework for **CM3604 Deep Learning** (Informatics Institute of Technology, in collaboration with Robert Gordon University Aberdeen). This repository implements two fact-checkers that verify ChatGPT-generated biographical claims against retrieved Wikipedia passages, based on the FActScore methodology (Min et al., EMNLP 2023).

## Final Results

| Part | Mode             | Accuracy          | Brief target |
|------|------------------|-------------------|--------------|
| 1    | `word_overlap`   | **81.00%** (179/221) | 75% |
| 2    | `entailment`     | **84.62%** (187/221) | 83% |

Both parts exceed the targets set in the assessment brief.

## Team

| Member | Role |
|--------|------|
| Sandali Balasooriya  | Part 1 lead (Word Overlap Fact Checker), report integration |
| Dilshan Indigahawela  | Part 2 lead (Textual Entailment Fact Checker) |
| Sachinthana Pathiranage  | Experiments and Part 3 lead (Error Analysis) |
| Manura Hettiarachchi  | Testing framework and additional experiments |

## Setup

1. **Install Python 3.10.11** from https://www.python.org/downloads/release/python-31011/ (tick "Add to PATH" during install, custom location `C:\Python310` recommended). Keep any newer Python you already have — the `py` launcher picks which one to use.

2. **Clone this repo** via GitHub Desktop, or:
   ```
   git clone https://github.com/SandaliBalasooriya/CM3604-factchecker.git
   cd CM3604-factchecker
   ```

3. **Open PowerShell** in the cloned repo folder (Shift + right-click → "Open in Terminal").

4. **Create and activate the virtual environment**:
   ```
   py -3.10 -m venv venv
   venv\Scripts\activate
   ```

5. **Install pinned requirements**:
   ```
   pip install -r requirements.txt
   ```

6. **Download NLTK data** (also downloaded automatically on first run):
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_sm
   ```

7. **(Optional, for GPU)** If you want to run Part 2 on a CUDA-capable GPU, swap torch for the CUDA build:
   ```
   pip uninstall torch torchvision -y
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```
   Note: this is a 0.0.1 patch deviation from the pinned `torch==2.5.0`. The submitted results were produced on CPU with the pinned version.

## Running the two modes

```
python factchecking_main.py --mode word_overlap   # Part 1 — 179/221 = 81.00%, ~1 second
python factchecking_main.py --mode entailment     # Part 2 — 187/221 = 84.62%, ~7 minutes on CPU
```

Add `--cuda` to the entailment command if PyTorch is installed with CUDA support.

## Running the tests

```
python -m unittest discover tests -v
```

Runs all **19 unit tests** (16 covering Part 1 preprocessing, scoring and prediction behaviour, plus 3 covering the driver's label handling). Completes in under 0.1 seconds. No model inference required.

## Repository structure

### Core framework
- `factcheck.py` — all three fact checker classes: `WordRecallThresholdFactChecker` (Part 1), `EntailmentModel` and `EntailmentFactChecker` (Part 2)
- `factchecking_main.py` — main driver (unmodified from the supplied framework)
- `requirements.txt` — pinned package versions for Python 3.10
- `data/` — the labelled dev set and retrieved Wikipedia passages

### Experiments and analysis
- `threshold_sweep.py` — Part 1 preprocessing ablation and scorer comparison (Tables 1.1, 1.2, 1.3)
- `evaluate_entailment.py` — Part 2 threshold sweep with SHA-256 cache (Table 2.3)
- `experiments.py` — Part 2 decision-rule comparison and pruning sweep (Tables 2.1, 2.2)
- `export_errors.py` — exports the final classifier's errors for Part 3 analysis
- `inspect_entailment.py` — development helper for inspecting individual predictions
- `results/` — cached scores and generated CSVs
- `report/` — experiment outputs behind the written report's tables

### Testing
- `tests/test_word_overlap.py` — 16 unit tests covering Part 1 preprocessing, scoring and prediction
- `tests/test_evaluation_labels.py` — 3 tests covering the driver's IR-as-NS handling

### Optional additional experiments
- `report/window_test.py` — compares single-sentence evidence with adjacent sentence pairs for Part 2