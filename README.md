
# Alzheimer's Disease Classification Using KNN

An educational machine learning project that explores patient data and uses **K-Nearest Neighbors (KNN)** to predict the dataset's Alzheimer's diagnosis label. The notebook covers data exploration, preprocessing, evaluation, and an explanation of an individual prediction using its nearest neighbors.

## Dataset

The code reads `alzheimers_disease_data(in).csv` and uses:

- **Target:** `Diagnosis` (`0` = No Alzheimer, `1` = Alzheimer).
- **Excluded columns:** `PatientID` and `DoctorInCharge`.
- **Predictors:** all remaining columns, including clinical, cognitive, demographic, and lifestyle measurements.

The notebook comments describe an expected dataset of **2,149 records**, producing **1,719 training records** and **430 test records**. Actual dimensions are printed when the notebook runs.

The dataset is not included with this README. Obtain it from its original provider and check its documentation and usage terms. Add the dataset source and citation to the repository before sharing the project.

## Workflow

1. Inspect sample records, dimensions, data types, missing values, duplicate counts, and unique values.
2. Examine diagnosis counts and percentages.
3. Visualize BMI, systolic blood pressure, total cholesterol, and MMSE using boxplots.
4. Count potential outliers using the interquartile range (IQR) rule; outliers are not automatically removed.
5. Create an **80/20 stratified train/test split** with `random_state=42`.
6. Prepare the predictors:
   - Standardize columns with more than 10 unique values.
   - One-hot encode `Ethnicity`, ignoring unseen categories during transformation.
   - Pass the remaining binary and ordinal columns through unchanged.
7. Fit preprocessing on the training set and transform both subsets.
8. Train a baseline KNN classifier with **k = 5**.
9. Evaluate predictions and compare alternative values of k.
10. Explain one test prediction using neighbor distances, labels, votes, and a PCA visualization.

## Evaluation

The baseline evaluation reports:

- Accuracy
- Balanced accuracy
- Precision for the Alzheimer class
- Recall for the Alzheimer class
- Macro F1-score

The notebook also compares **k = 3, 5, 7, 9, 11, and 15** using balanced accuracy and Alzheimer recall. Scores are calculated during execution; no measured results are supplied here.

## Visualizations

- Diagnosis distribution with counts and percentages
- Boxplots for selected measurements
- MMSE versus total cholesterol before and after standardization
- Confusion matrix
- Baseline performance bar chart
- Balanced accuracy and Alzheimer recall across k values
- A two-dimensional PCA projection highlighting a query patient and its five nearest neighbors

PCA is used only for visualization. Neighbor search uses all processed features, so distances in the PCA plot do not represent the full model distances exactly.

## Requirements

- Python 3.10 or newer
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn 1.2 or newer (`OneHotEncoder` uses `sparse_output=False`)
- Jupyter Notebook, JupyterLab, or Google Colab

Install the dependencies locally:

```bash
python -m pip install pandas numpy matplotlib seaborn "scikit-learn>=1.2" jupyterlab
```

## Run the Notebook

### Google Colab

1. Open the project notebook in Colab.
2. Upload the CSV file.
3. Update the loading line to match its location, for example:

   ```python
   df = pd.read_csv("/content/alzheimers_disease_data(in).csv")
   ```

4. Run the cells in order.

### Local Jupyter

1. Install the dependencies above.
2. Place the dataset beside the notebook and update the loading line:

   ```python
   df = pd.read_csv("alzheimers_disease_data(in).csv")
   ```

3. Start JupyterLab:

   ```bash
   jupyter lab
   ```

4. Open the notebook and run all cells from the beginning.

The code uses notebook `display()` output and is intended for an interactive notebook environment.

## Prediction Explanation

For the first test patient, the notebook displays the actual and predicted diagnosis, the five nearest training neighbors, their distances, and their diagnosis votes. This explanation uses **k = 5**, even if another k performs better in the comparison.

## Current Limitations

- Alternative k values are compared on the test set. Selecting k from those results makes that test set part of model selection; use training-only cross-validation and a separate final test evaluation for a stronger estimate.
- Feature types are selected using unique-value counts from the full predictor dataset. For stricter evaluation, determine these from training data or define them from the dataset schema.
- Missing values and duplicate rows are inspected but not automatically handled. Review them before training; the current workflow does not include imputation.
- The greater-than-10-unique-values rule is a heuristic and may not reflect every feature's meaning.
- Results from this dataset do not establish performance on other patient populations.

This project is for education and research, not clinical diagnosis or medical decision-making.
