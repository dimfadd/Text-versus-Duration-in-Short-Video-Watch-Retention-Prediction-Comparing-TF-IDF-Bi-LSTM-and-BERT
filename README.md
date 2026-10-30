# Text versus Duration in Short-Video Watch Retention Prediction: Comparing TF-IDF, Bi-LSTM, and BERT

Code and experiment outputs for the paper of the same title by Dimas Fadilah Permana R and Ema Utami (Universitas AMIKOM Yogyakarta), submitted to *MALCOM: Indonesian Journal of Machine Learning and Computer Science*.

The study asks how much a video's spoken script (its ASR transcript) predicts watch retention on a short-video platform, and how that compares with video duration alone. Retention is the average share of a video that viewers watch, capped at 1. Because duration is both a predictor and the denominator of the target, the analysis checks for mathematical coupling and repeats the experiments on a duration-adjusted target.

## Repository contents

```
notebooks/
  01-preprocessing.ipynb           data cleaning, target construction, splits
  02-modeling.ipynb                11 models x 4 experiments, tuning and test predictions
  03-evaluasi-dan-analisis.ipynb   metrics, bootstrap tests, coupling check, figures
results/
  01_preprocessing/                data trail, descriptive statistics, column dictionary, plots
  02_modeling/
    ringkasan_pelatihan.csv        selected settings and test scores per model and experiment
    utama/ adj/ author/ noclip/    one folder per experiment:
      pengaturan.json              experiment configuration
      log_tuning.csv               every hyperparameter combination tried, with validation RMSE
      prediksi_uji.csv             test-set predictions of every model (video id, actual, predicted)
  03_evaluation/                   tables and figures used in the paper
```

The notebooks are saved with their cell outputs, so the results can be read without running anything. Narrative text and variable names in the notebooks are in Indonesian.

## Data

The source is the Kuaishou large-scale short-video dataset released by Shang et al. (Tsinghua University):

> Y. Shang, C. Gao, N. Li, and Y. Li, "A large-scale dataset with behavior, attributes, and content of mobile short-video platform," arXiv:2502.05922, 2025. https://doi.org/10.48550/arXiv.2502.05922

The raw data and the processed files that contain transcripts or creator IDs (`dataset_full.csv`, `train.csv`, `val.csv`, `test.csv`) are not included here. Please obtain the dataset from its authors. Notebook 01 rebuilds every processed file from the raw interaction, category, and ASR files.

Processing steps (see `results/01_preprocessing/jejak_data.csv`):

| Step | Videos |
|---|---|
| Videos in the interaction file (794,053 rows, 129,483 unique interactions) | 31,496 |
| With an ASR transcript | 31,492 |
| Duration 5 to 120 seconds | 22,620 |
| At least 5 unique viewers | 5,075 |
| Transcript of at least 5 words | 4,648 |

The 4,648 videos are split 70/15/15 into train, validation, and test sets (3,253 / 697 / 698). A second, creator-disjoint split keeps all videos of a creator in one partition.

## Experiments

| Code | Target | Split | Models |
|---|---|---|---|
| `utama` | retention | random | all 11 |
| `adj` | duration-adjusted retention (residual of an isotonic fit on duration) | random | 9 (no ablations) |
| `author` | retention | creator-disjoint | 9 (no ablations) |
| `noclip` | retention without the cap at 1 | random | 5 (tree models only) |

Models, grouped by the input they receive:

- Duration only: XGBoost.
- Text only: TF-IDF + Random Forest, TF-IDF + XGBoost, Bi-LSTM, BERT (`bert-base-uncased`).
- Text and duration: the same four architectures with duration added.
- Ablations (BERT): duration passed through a learned projection, and duration written as text at the start of the transcript.

Hyperparameters are chosen by the lowest validation RMSE. Test performance is measured with Spearman rank correlation (SRC), partial SRC controlling for duration, MAE, RMSE, top-decile lift, and Precision@10%. Model differences use a paired bootstrap (2,000 resamples, the same resample indices for all models) with Holm correction.

## Main results (test set, random split)

| Input | Model | SRC | 95% CI |
|---|---|---|---|
| Duration | XGBoost | 0.476 | 0.415 to 0.534 |
| Text | TF-IDF + RF | 0.299 | 0.230 to 0.363 |
| Text | TF-IDF + XGBoost | 0.277 | 0.207 to 0.343 |
| Text | BERT | 0.272 | 0.199 to 0.343 |
| Text | Bi-LSTM | 0.100 | 0.027 to 0.174 |
| Text + duration | TF-IDF + XGBoost | 0.483 | 0.418 to 0.538 |
| Text + duration | TF-IDF + RF | 0.461 | 0.398 to 0.516 |
| Text + duration | BERT | 0.319 | 0.245 to 0.390 |
| Text + duration | Bi-LSTM | 0.255 | 0.182 to 0.326 |
| Ablation | BERT, duration as text | 0.470 | 0.406 to 0.528 |
| Ablation | BERT, projected duration | 0.302 | 0.230 to 0.372 |

In short:

- Duration alone is the strongest single input. Adding text to it gives no significant gain (TF-IDF + XGBoost, +0.007).
- The text-only models have a partial SRC near zero once duration is held constant, so on this target most of what they capture is duration carried by transcript length (SRC between word count and duration is 0.737).
- The coupling between duration and the target is real but weaker than chance would produce: SRC(duration, retention) is -0.484 against a permutation null of -0.719.
- On the duration-adjusted target, duration alone falls to 0.014, while BERT on text reaches 0.182 (95% CI 0.103 to 0.254). That is the only significant text-over-duration gain after Holm correction (p = 0.020).
- Results hold on the creator-disjoint split and without the cap at 1.

All tables behind these numbers are in `results/03_evaluation/`.

## Running the notebooks

The notebooks were run on Kaggle (notebook 02 on a Tesla T4 GPU) and also detect Google Colab or a local machine. Run them in order:

1. `01-preprocessing.ipynb` reads the raw dataset and writes the processed files.
2. `02-modeling.ipynb` reads the output of notebook 01 and writes `hasil_model_revisi/`.
3. `03-evaluasi-dan-analisis.ipynb` reads the outputs of notebooks 01 and 02 and writes the tables and figures.

Main libraries: pandas, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers, SciPy, matplotlib, and `emoji`. All random seeds are set to 42. Tree models and the statistics in notebook 03 are deterministic. Bi-LSTM and BERT training on a GPU can vary slightly between runs.

## Citation

If you use this code, please cite the paper (details will be added after publication):

```
D. F. Permana R and E. Utami, "Text versus Duration in Short-Video Watch Retention Prediction:
Comparing TF-IDF, Bi-LSTM, and BERT," MALCOM: Indonesian Journal of Machine Learning and
Computer Science.
```

## Contact

Dimas Fadilah Permana R, Master of Informatics, Universitas AMIKOM Yogyakarta.
