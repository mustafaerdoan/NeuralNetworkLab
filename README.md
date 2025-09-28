# BabyLM Training and Evaluation

This project trains two small language models (BabyLMs) from scratch and evaluates them on 1,000 minimal pairs (BLiMP). It includes data prep, training, evaluation, and plots.

## Project Structure
```
.
├── configs/
│   ├── model1.yaml                  # Small model config (2 MB)
│   └── model2.yaml                  # Larger model config (4 MB)
├── data/
│   ├── raw/
│   │   ├── train_small.txt
│   │   ├── eval_small.txt
│   │   ├── train_large.txt
│   │   └── eval_large.txt
│   └── processed/
│       └── minimal_pairs.json       # 1,000 BLiMP pairs
├── models/
│   ├── babylm_model_small/
│   │   └── final/                   # Final HF format model + tokenizer/
│   └── babylm_model_large/
│       └── final/
├── results/
│   ├── model1_evaluation.json
│   ├── model2_evaluation.json
│   └── model_comparison.png
├── src/
│   ├── prepare_data.py              # Build 2 MB / 4 MB text files
│   ├── prepare_minimal_pairs.py     # Build BLiMP 1,000 pairs
│   ├── train.py                     # Train BabyLMs (LLaMA config)
│   ├── evaluate.py                  # Evaluate minimal pairs (perplexity)
│   ├── plot_training.py             # Plot training/eval loss
│   └── plot_comparison.py           # Bar chart by phenomenon
├── report.md                        # One-page report
└── requirements.txt
```

## Setup
1) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

## Quick start
The repository already contains outputs. To fully reproduce:

1) (Optional) Rebuild training data
```bash
python src/prepare_data.py --output_file data/raw/train_small.txt --size_mb 2 --split train
python src/prepare_data.py --output_file data/raw/eval_small.txt  --size_mb 0.5 --split validation
python src/prepare_data.py --output_file data/raw/train_large.txt --size_mb 4 --split train
python src/prepare_data.py --output_file data/raw/eval_large.txt  --size_mb 0.5 --split validation
```

2) (Optional) Rebuild the 1,000 minimal pairs
```bash
python src/prepare_minimal_pairs.py --output_file data/processed/minimal_pairs.json --num_pairs 1000
```

3) Train both models
```bash
python src/train.py --config configs/model1.yaml
python src/train.py --config configs/model2.yaml
```

4) Plot training curves
```bash
python src/plot_training.py --model_path models/babylm_model_small
python src/plot_training.py --model_path models/babylm_model_large
```

5) Evaluate on minimal pairs
```bash
python src/evaluate.py --model_path models/babylm_model_small/final --eval_data data/processed/minimal_pairs.json --output results/model1_evaluation.json
python src/evaluate.py --model_path models/babylm_model_large/final --eval_data data/processed/minimal_pairs.json --output results/model2_evaluation.json
```

6) Compare models
```bash
python src/plot_comparison.py --model1_results results/model1_evaluation.json --model2_results results/model2_evaluation.json --output_file results/model_comparison.png
```


Recommended upload targets:

- Code, configs, report, and small result files: GitHub (public repo). Include `README.md`, `report.pdf`, `results/*.json`, and plots. Avoid committing large weight binaries unless using Git LFS.
- Trained models: Hugging Face Hub (two separate repos). Upload the contents of:
  - `models/babylm_model_small/final/`
  - `models/babylm_model_large/final/`

These folders already contain the necessary HF files (e.g., `config.json`, `pytorch_model.bin`, `tokenizer/`, etc.). After uploading, place the model links in this README.

Example usage :
```python
from transformers import LlamaForCausalLM, AutoTokenizer

model = LlamaForCausalLM.from_pretrained("<your-hf-username>/babylm-model-small")
tokenizer = AutoTokenizer.from_pretrained("<your-hf-username>/babylm-model-small")
```

## Repro checklist 
1. Clone the GitHub repo and `cd` into it.
2. Create venv and `pip install -r requirements.txt`.
3. (Skip if using prebuilt data) Run the optional data prep commands above.
4. Run the two training commands (or reuse the provided `models/*/final`).
5. Run the two evaluation commands.
6. Open `results/model_comparison.png` and the JSONs; read `report.pdf`.

If anything fails, please check Python 3.10+, `torch>=2.0`, `transformers>=4.30`, and that the dataset download is allowed by your network.
