# PII Anonymization in Educational Data

## Overview
This repository contains the implementation for my Bachelor's thesis **Balancing Privacy and Readability: A Comparative Evaluation of Text Anonymization Methods for Learning Analytics**. 

The pipeline compares three different PII anonymization approaches on educational data:

1. **Rule-based**: Baseline method using pattern matching with regex and function word filtering
2. **Presidio**: Baseline method using Microsoft's hybrid framework with NLP + pattern matching framework
3. **Transformer-based**: Custom developed DeBERTa model trained on student essays (microsoft/deberta-v3-large english)

All approaches are evaluated on the Kaggle PII Detection Competition dataset including **6807 student essays** using **5-fold cross-validation**. The evaluation measures both the PII detection accuracy of the implemented methods (Precision, Recall, F1-Score), and their readability preservation metrics (Flesch, FK Grade, Gunning Fog).

---

## Key Features
- **Three anonymization methods** compared and evaluated with a consistent pipeline
- **5-fold cross-validation** with statistical significance testing (McNemar, paired t-test)
- **Readability analysis** to measure the text quality before and after anonymization
- **Categorical evaluation** across 7 PII types (NAME_STUDENT, EMAIL, URL_PERSONAL, etc.)
- **Error analysis** with false negative/positive examples showing their surrounding context

---

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for DeBERTa training and inference)
- ~2GB disk space (excluding models and full datasets)


### Setup
```bash
# Clone the repository
git clone https://github.com/zozbay/pii-anonymization-in-educational-data.git
cd pii-anonymization-in-educational-data

# Create virtual environment
python -m venv venv
source venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

---

### Dataset

**Dataset Source**: [Kaggle PII Detection Competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

**Models Source**: [HuggingFace](https://huggingface.co/zozbay/pii-detection-deberta-v3-large)

**Note**: This repository does not include the complete data files (`data/train.json` and `data/fold_X_train.json`) and the models ('`models/deberta_fold_{0-4}`') due to size constraints, the validation folds are included for reference.

To use this pipeline:
1. Download models from HuggingFace
```bash

   # Run from project root directory

   python download_models.py

```
Or download manually from [HuggingFace](https://huggingface.co/zozbay/pii-detection-deberta-v3-large)

Create a models folder in root directory and place the downloaded models in models/ directory. After download, this directory should contain:
```
   models/
   ├── fold_0/
   ├── fold_1/
   ├── fold_2/
   ├── fold_3/
   └── fold_4/
```
2. Download training data:
   - Download train.json from [Kaggle's Competition Website](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)
   - Place in data/ directory

3. Generate fold splits:
```bash

   python save_datasets.py

```

**Overview of the Dataset:**

| Metric | Count |
|--------|-------|
| **Total Documents** | 6,807 |
| **Total Tokens** | 4,992,533 |
| **PII Tokens** | 2,739 |
| **Non-PII Tokens** | 4,989,794 |
| **Total PII Entities** | 1,606 |

**PII Category Distribution:**

| Category | Count |
|----------|-------|
| NAME_STUDENT | 1,365 |
| URL_PERSONAL | 110 |
| ID_NUM | 78 |
| EMAIL | 39 |
| USERNAME | 6 |
| PHONE_NUM | 6 |
| STREET_ADDRESS | 2 |

---

## Usage

### Quick Evaluation for a Single Method
```bash
# Evaluate Rule-based method (on first validation fold)
python main.py rule data/fold_0_val.json

# Evaluate Presidio
python main.py presidio data/fold_0_val.json

# Evaluate DeBERTa (requires trained model)
python main.py customdeberta data/fold_0_val.json
```

### Training DeBERTa (Per Fold)
```bash
# Train on fold 0
python deberta_train_fold.py 0

# Train all folds
for i in {0..4}; do python deberta_train_fold.py $i; done
```

### 5-Fold Cross-Validation
```bash
# Evaluate single fold for all 3 methods (You do not need to run these individually unless specifically needed)
python evaluation/single_fold_eval.py 2

# Run complete 5-fold CV with statistical tests (Complete evaluation script)
python evaluation/all_folds_eval.py
```

### Generate Results

**Entity-level categorical analysis for distinct PII categories (Standard evaluation):**
```bash
python evaluation/categorical_entity_level.py
```

**Token-level categorical analysis for distinct PII categories (For complementary error evaluation):**
```bash
python evaluation/categorical_token_level.py
```

**Readability preservation analysis (For all methods):**
```bash
python evaluation/readability_eval.py
```

**Error analysis showing false positives/negatives in their surrounding context (For the developed transformer-model):**
```bash
python evaluation/context_error_examples.py
```


---

## Directory Structure
```
pii-anonymization-in-educational-data/
│
├── data/
│   ├── train.json                    # Original dataset containing 6807 essays
│   ├── fold_{0-4}_train.json         # Training folds (5,445 essays each)
│   ├── fold_{0-4}_val.json           # Validation folds (1,362 essays each)
│   └── output/                       # Anonymized text output directory
│
├── evaluation/
│   ├── all_folds_eval.py             # Full 5-fold cross-validation for all methods
│   ├── categorical_entity_level.py   # Analysis per PII category (entity-level)
│   ├── categorical_token_level.py    # Analysis per PII category (token-level)
│   ├── context_error_examples.py     # Error examples with context
│   ├── readability_eval.py           # Readability analysis
│   └── single_fold_eval.py           # Single fold evaluation for all methods
│
├── methods/
│   ├── rule_based.py                 # Rule-based method
│   ├── presidio.py                   # The hybrid method Microsoft Presidio
│   └── custom_deberta.py             # The transformer-based model DeBERTa-v3
│
├── models/
│   ├── deberta_fold_0/               # Trained DeBERTa models
│   ├── deberta_fold_1/               
│   ├── deberta_fold_2/
│   ├── deberta_fold_3/
│   └── deberta_fold_4/
│
├── pipeline/
│   ├── anonymizer.py                 # Main anonymization pipeline
│   ├── evaluator.py                  # NER evaluation
│   ├── obfuscator.py                 # PII obfuscation
│   ├── readability.py                # Readability metrics
│   └── reconstruct.py                # Text reconstruction from tokens
│
├── results/
│   ├── category_entity_level_results.json # Entity-level categorical metrics
│   ├── category_token_level_results.json  # Token-level categorical metrics
│   ├── cross_validation_results.json      # Aggregated 5-fold CV results
│   ├── error_examples_all_folds.json      # Error analysis with context
│   ├── readability_results.json           # Readability analysis
│   └── fold_{0-4}_comparison.json         # Individual fold results (0-4)
│
├── deberta_train_fold.py             # DeBERTa training script
├── main.py                            # Main entry point
├── save_datasets.py                   # Creates 5-fold splits
├── requirements.txt                   # Required Python dependencies
├── download_models.py                 # Downloads fine-tuned models
├── structure.txt                      # Detailed file structure
└── README.md                          # This file
```

---

## Results Summary

### Detection Performance (5-Fold CV)

| Method       | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Rule-based   | 2.3%      | 40.1%  | 4.3%     |
| Presidio     | 16.4%     | 64.0%  | 26.0%    |
| **DeBERTa** | **95.7%** | **83.3%** | **89.0%** |

### Readability Preservation (Mean Δ)

| Method       | Δ Flesch | Δ FK Grade | Δ Gunning Fog |
|--------------|----------|------------|---------------|
| Rule-based   | +0.15 ± 0.84 | -0.02 ± 0.12 | -0.02 ± 0.18 |
| Presidio     | -0.02 ± 0.85 | +0.02 ± 0.30 | +0.02 ± 0.32 |
| **DeBERTa** | **+0.01 ± 0.20** | **-0.00 ± 0.05** | **-0.00 ± 0.06** |

**Note**: Δ < 0.20 means negligible impact on readability (the best value is 0).

**Key Findings:**
- The transformer-based model DeBERTa achieves **the highest F1-score with 89.0%** and significantly outperforms Presidio (26.0%) and rule-based (4.3%) approaches.
- DeBERTa also achieves **the best readability preservation** with Δ < 0.20 with low variance for all readability metrics.
- Rule-based and Presidio show low mean changes but high variability (σ up to 0.85), indicating inconsistent alterations across documents.
- All differences are statistically significant (McNemar's test, p < 0.01) for all pairwise comparisons.

---

## Acknowledgments

- **Dataset**: [Kaggle PII Detection Competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data) by The Learning Agency Lab

- **Models**: 
  - [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) by Microsoft
  - [Presidio](https://github.com/microsoft/presidio) by Microsoft
  - [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg) by spaCy

- **Libraries**: 
  - [Hugging Face Transformers](https://github.com/huggingface/transformers) for model training
  - [seqeval](https://github.com/chakki-works/seqeval) for entity level NER evaluation metrics
  - [scikit-learn](https://scikit-learn.org/) for token level metrics during categorical analysis
  - [textstat](https://github.com/textstat/textstat) for readability analysis
  - [Faker](https://github.com/joke2k/faker) for PII surrogate generation
  - [SciPy](https://scipy.org/) for statistical significance tests
  - [PyTorch](https://pytorch.org/) for the deep learning framework

---

## License

All code files provided here as a part of my Bachelor's Thesis are for reference, reproducibility, and academic purposes. 

**Note**: Please check the [competition rules](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/rules) if you decide to use the dataset.

---

## Citation

If you use this code or methodology in your research, please cite:
```bibtex
@bachelorsthesis{zozbay2026pii,
  author = {Zeynep Deniz Özbay},
  title = {Balancing Privacy and Readability:
A Comparative Evaluation of
Text Anonymization Methods
for Learning Analytics},
  school = {Humboldt Universität zu Berlin},
  year = {2026},
  type = {Bachelor's Thesis}
}
```

---

## Contact Me

**Author**: Zeynep Deniz Özbay  
**Institution**: Humboldt Universität zu Berlin  
**Email**: [zeynep.deniz.oezbay@student.hu-berlin.de]  
**GitHub**: [@zozbay](https://github.com/zozbay)

For questions or issues, please feel free to contact me. :)