# Machine Learning Labs

Jupyter notebooks from a university machine learning course: signal denoising, image clustering,
regression on financial time series, and classical classifiers (k-NN, Naive Bayes, SVM) on image data,
implemented in Python with scikit-learn, scikit-image and OpenCV.

## Labs

| Lab | Topic | Techniques / models | Key result (from saved outputs) | Notebook |
|-----|-------|---------------------|---------------------------------|----------|
| 1 | ECG signal denoising | Discrete wavelet denoising (BayesShrink, soft thresholding, `sym8`, 3 levels) on a noisy ECG signal | Plots of noisy vs. denoised and original vs. denoised signal | [ecg_wavelet_denoising.ipynb](01-ecg-wavelet-denoising/ecg_wavelet_denoising.ipynb) |
| 2.1 | Image dataset exploration | Per-class image count, class distribution bar chart | 2 balanced classes: 78 `red` / 77 `blue` images | [dataset_class_distribution.ipynb](02-image-dataset-clustering/dataset_class_distribution.ipynb) |
| 2.2 | Unsupervised image clustering | Grayscale flattening, `StandardScaler`, K-Means (k = 2) | Images plotted per cluster with their true labels | [kmeans_image_clustering.ipynb](02-image-dataset-clustering/kmeans_image_clustering.ipynb) |
| 3.1 | Bitcoin price regression | `LinearRegression` on one year of daily BTC-USD closing prices (`yfinance`) | Test MSE: 63,944,179.89 | [linear_regression.ipynb](03-bitcoin-price-regression/linear_regression.ipynb) |
| 3.2 | Bitcoin price time-series regression | `GradientBoostingRegressor` (100 estimators) on the same data | Test MSE: 2,646,648.92 (about 24x lower than linear regression) | [time_series_gradient_boosting.ipynb](03-bitcoin-price-regression/time_series_gradient_boosting.ipynb) |
| 5 | Image classification | k-Nearest Neighbors (k = 3) vs. Gaussian Naive Bayes on 64x64 flattened images; predictions exported to CSV and per-image plots | Results written to Google Drive (not stored in the notebook) | [knn_vs_naive_bayes.ipynb](05-knn-vs-naive-bayes/knn_vs_naive_bayes.ipynb) |
| 6 | Image similarity | Linear SVM (`SVC`) on absolute-difference features of image pairs (similar / not similar), model saved with `joblib` | Accuracy 0.50 on a tiny test split (14 pairs in total); sample test pair predicted similar with score 0.76 | [svm_image_similarity.ipynb](06-svm-image-similarity/svm_image_similarity.ipynb) |

Lab 4 is not included in this repository.

## Tech stack

- Python 3, Jupyter / Google Colab
- NumPy, pandas, Matplotlib
- scikit-learn (K-Means, linear regression, gradient boosting, k-NN, Naive Bayes, SVM)
- scikit-image and PyWavelets (wavelet denoising, image I/O and resizing)
- OpenCV (image loading, resizing, color conversion)
- yfinance (Bitcoin price data), joblib (model persistence)

## Getting started

The notebooks were written and run in Google Colab, and their saved outputs are kept, so they can be
read directly on GitHub.

To run them locally:

```bash
git clone https://github.com/armandursei/MachineLearning.git
cd MachineLearning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Data:

- **Lab 1** uses the ECG sample bundled with PyWavelets and **Lab 3** downloads prices from Yahoo Finance,
  so both run without extra files.
- **Labs 2, 5 and 6** read image datasets from Google Drive (`/content/drive/MyDrive/Colab Notebooks/...`).
  The datasets are not part of this repository. In Colab, mount Google Drive and place the data at those
  paths; locally, change the path variables at the top of each notebook.
  - Labs 2 and 5 expect a folder with one sub-folder per class (Lab 2.2 loads `.jpeg` files).
  - Lab 6 expects a `dataset.csv` whose three columns name the `Reference`, `Identical` and `Unmatching`
    image sub-folders, plus a `Test` folder with the images to compare.

## Project structure

```
MachineLearning/
├── 01-ecg-wavelet-denoising/
│   └── ecg_wavelet_denoising.ipynb
├── 02-image-dataset-clustering/
│   ├── dataset_class_distribution.ipynb
│   └── kmeans_image_clustering.ipynb
├── 03-bitcoin-price-regression/
│   ├── linear_regression.ipynb
│   └── time_series_gradient_boosting.ipynb
├── 05-knn-vs-naive-bayes/
│   └── knn_vs_naive_bayes.ipynb
├── 06-svm-image-similarity/
│   └── svm_image_similarity.ipynb
├── requirements.txt
└── README.md
```

Note: the plot images saved in the notebook outputs were rendered before translation, so some axis
labels and titles in them are still in Romanian. Re-running the notebooks produces the English labels.

## Author

George-Armand Ursei - [GitHub](https://github.com/armandursei)
