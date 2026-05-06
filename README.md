# ImpAcX_OnHW (Fork): Improving Accuracy and Explainability of Handwriting Recognition Models

This is a **fork** of the original project [ImpAcX_OnHW](https://github.com/.../impacx_onhw) that contains code and information to help reproduce, verify, and extend the results reported in the paper **["Improving Accuracy and Explainability of Handwriting Recognition Models"](https://arxiv.org/abs/2209.09102)**.  
This fork may include additional features, refactored code, or modified experiments while maintaining the core functionality of the original repository.

## How to cite this project

If you use *this fork* in your research or development, please still credit the original authors and, if applicable, mention the forked repository. For example:

```bibtex
@misc{impacx_onhw,
  author = {H. Azimi and S. Chang and J. Gold and K. Karabina},
  title = {{Improving Accuracy and Explainability of Handwriting Recognition Models}},
  year = {2022},
  publisher = {arXiv},
  pages = {1-20},
  url = {https://arxiv.org/abs/2209.09102},
  doi = {10.48550/ARXIV.2209.09102},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

If you explicitly depend on modifications made in this fork, you may add a separate `@misc` entry for the fork (using your GitHub handle and repo URL).

---

## Description

The project directory structure is:

```bash
├── dataset
├── DTW_KNN.py
├── ensemble_evaluation.py
├── explain.py
├── failure_space_analysis.py
├── failure_space_array.py
├── OnHW_DL_optimize.py
├── OnHW_ML_DL_baseline_and_optimized_train_and_eval.py
├── OnHW_ML_extract_features_various_nsig.py
├── OnHW_ML_train_KNN_various_ncomp.py
├── OnHW_ML_train_KNN_various_nsig.py
├── plot_kNN_results.py
├── README.md
├── requirements.txt
├── results_to_visual.py
└── utils
```

- `requirements.txt` lists all Python dependencies for this project.

### Dataset setup

Before running the code, download the **OnHW‑chars dataset** from [Stabilo Digital](https://stabilodigital.com/onhw-dataset/) and unzip it into the `dataset` folder so the structure looks like:

```bash
dataset/
└── IMWUT_OnHW-chars_dataset_2021-06-30
    └── OnHW-chars_2021-06-30
        ├── onhw2_both_dep_0
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        ├── onhw2_both_dep_1
        ├── ...
        ├── onhw2_upper_indep_4
        │   ├── X_test.npy
        │   ├── X_train.npy
        │   ├── y_test.npy
        │   └── y_train.npy
        └── readme.txt
```

### Main scripts and their roles

- **`OnHW_ML_DL_baseline_and_optimized_train_and_eval.py`**  
  Calls `make_some_folders()` to create the `output/` directory tree, then runs:
  - Feature filtering and extraction with tsfresh: `OnHW_ML_filter_and_extract()`.  
  - Training and evaluation of 7 machine‑learning baseline models: `OnHW_ML_train_all()` and `OnHW_ML_evaluate_all()`.  
  - Training and evaluation of the same 7 models under a metric‑learning transformation:  
    `OnHW_MetricLearn_train_all()` and `OnHW_MetricLearn_evaluate_all()`.  
  - Training and evaluation of 12 deep‑learning baseline models:  
    `OnHW_DL_train_all()` and `OnHW_DL_evaluate_all()`.  
  - Training and evaluation of 4 *optimized* deep‑learning models (from `OnHW_DL_optimize.py`):  
    `OnHW_DL_optimized_train_all()` and `OnHW_DL_optimized_evaluate_all()`.

- **`OnHW_DL_optimize.py`**  
  Uses [Optuna](https://optuna.org/) to optimize hyperparameters for 4 deep‑learning models. The resulting models and parameter sets can be passed to the `DL_OPT_MODEL_LIST` variable in `OnHW_ML_DL_baseline_and_optimized_train_and_eval.py` to be trained and evaluated.

- **`OnHW_ML_extract_features_various_nsig.py`**  
  Extracts and selects features for various levels of `n_significant` to reproduce **Figure 3** (by default using all 5 folds of lowercase writer‑independent data). The `n_significant` levels and datasets are configurable in `./utils/config.py`.

- **`OnHW_ML_train_KNN_various_nsig.py`**  
  After running `OnHW_ML_extract_features_various_nsig.py`, trains and evaluates KNN models for `k` from 1 to 49 and various `n_significant` levels. Results are saved under `./output/ml_results` for **Figure 3**.

- **`OnHW_ML_train_KNN_various_ncomp.py`**  
  After feature extraction, trains and evaluates KNN models for `k ∈ [1, 49]` and various `n_components` to produce **Figure 4**. By default `n_significant = 17` and all 5 folds of lowercase writer‑independent data are used; settings are in `./utils/config.py`. Results go to `./output/ml_results`.

- **`results_to_visual.py`**  
  After model trainings and evaluations, computes per‑fold and mean accuracies, standard deviations, and generates the interactive HTML visualization `ml_dl_all_results.html` under `./output/visuals`.

- **`plot_kNN_results.py`**  
  Uses `pyplot` and the `./output/ml_results` files (from the KNN scripts above) to produce **Figures 3 and 4**, saved under `./output/visuals`.

- **`ensemble_evaluation.py`**  
  After model 