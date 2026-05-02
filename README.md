# SwP Benchmark Pipeline

This repository contains the evaluation pipeline used to compare **Sampling with People (SwP)**, **human cloze surprisal**, and **language-model surprisal** on the **BK21 self-paced reading benchmark**.

---

### Stage A. Generate unidirectional surprisal values
For each target word in BK21, compute unidirectional surprisal from:
- GPT-2
- Gemma-270M
- Gemma-12B

The outputs are merged into a master CSV.

### Stage B. Add SwP surprisal values
SwP surprisal, entropy, and target-rank information are merged into the same master table.

### Stage C. Run the main benchmark
Fit linear mixed-effects models to predict **log reading time** from:
- lexical controls only
- cloze only
- predictor only
- cloze + predictor

### Stage D. Run the coverage-controlled benchmark
Run the same benchmark again, but only on rows where the **SwP target word is present**. This makes it possible to separate:
- vocabulary coverage effects
from
- model quality effects

---

## Prerequisites

### Python
Packages:
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `transformers`
- `torch`
- `wordfreq`
- `nltk`

### R
Required packages:
- `lme4`
- `lmerTest`

Install in R with:

```r
install.packages(c("lme4", "lmerTest"))
```

---

## Surprisal pipeline

This project only uses the **unidirectional** surprisal values in the final benchmark.

### Input
The pipeline assumes a BK21 target-word table with sentence context and target words.

### Output
The final master file, created from `data_merge.ipynb` should contain these columns:

```text
ITEM, condition, critical_word, sentence,
cloze_surprisal,
gpt2_uni_surprisal,
gemma270m_uni_surprisal,
gemma12b_uni_surprisal,
word_length, word_frequency, log_rt
```

Final master file used in the report

```text
data/master_modeling_data.csv
```

### SwP-augmented file used later

```text
data/master_modeling_data_with_swp.csv
```

This second file should additionally contain:

```text
swp_surprisal_bits,
swp_entropy_bits,
swp_target_rank,
swp_top1,
swp_target_in_top5,
swp_target_in_top10
```

---

## Main LME benchmark

### Main script command

From R:

```r
source("LME/run_all_data.R")
```

---

## SwP benchmark on the same framework

To benchmark SwP on the same dataset and model structure, run:

```r
source("LME/benchmark_utils.R")
```
This produces the **full-benchmark SwP result** reported in the paper.

---

## Coverage-controlled benchmark

The coverage-controlled benchmark evaluates **all predictors on the same subset**, namely rows where the SwP target word is available.

This is important because SwP's vocabulary is restricted, so full-dataset performance reflects both:
- predictor quality
- vocabulary coverage

### Run command

```r
source("LME/run_coverage_controlled.R")
```

---

## Notebook

The final notebook should focus solely on report-facing analyses.

```text
final_analysis.ipynb
```

