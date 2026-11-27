# Task 3: Telco Customer Churn (Random Forest and XGBoost)

By TSAWO Beautrel Carel

## Objective
Predict which telecom customers will leave (churn). The classes are
imbalanced, so I compared several ways to handle this and chose metrics
that fit the problem (recall, PR-AUC, ROC-AUC) instead of accuracy.

## Data
- 7,043 customers, 21 columns: the target `Churn`, the id `customerID`
  (removed) and 19 features (4 numeric, 15 categorical).
- 26.5 % of customers left, so a model that always predicts "stay" would
  reach 73.5 % accuracy while catching no churner.
- `TotalCharges` was read as text. After conversion, 11 values were missing,
  all with `tenure = 0`: new customers who had not paid anything yet. I
  imputed them with 0 (inside the pipeline), not with the median, because
  the real value is logically 0.

## Method
1. Exploration: types, missing values, target balance, churn by contract.
2. Stratified split 80/20 (`random_state=42`), done before any preprocessing.
3. Preprocessing in a `ColumnTransformer`: constant imputation and scaling
   for numeric columns, `OneHotEncoder` for categorical columns (45 columns
   after encoding).
4. Baseline: logistic regression.
5. Imbalance: class weights and SMOTE. SMOTE is used inside an `imblearn`
   pipeline, so it only touches the training part of each fold.
6. Random Forest and XGBoost, first with default parameters, then with
   limited depth to reduce overfitting.
   - Random Forest: `n_estimators=300`, `max_depth=8`, `min_samples_leaf=10`,
     `class_weight="balanced"`.
   - XGBoost: `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`,
     `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight` = ratio of
     stayed to left customers in the train set.
7. Comparison with 5-fold stratified cross-validation on the train set.
8. I chose Random Forest before looking at the test set, and used the test
   set only once.

## Results

Cross-validation on the train set (5 folds):
- Logistic regression: ROC-AUC 0.846, recall 0.544, precision 0.654, PR-AUC 0.662
- Logistic regression + class weights: ROC-AUC 0.846, recall 0.801, precision 0.517, PR-AUC 0.660
- Logistic regression + SMOTE: ROC-AUC 0.845, recall 0.789, precision 0.526, PR-AUC 0.658
- Random Forest (default): ROC-AUC 0.823, recall 0.641, precision 0.566, PR-AUC 0.605
- XGBoost (default): ROC-AUC 0.811, recall 0.581, precision 0.560, PR-AUC 0.599
- Random Forest (tuned): ROC-AUC 0.848, recall 0.797, precision 0.528, PR-AUC 0.663
- XGBoost (tuned): ROC-AUC 0.846, recall 0.791, precision 0.525, PR-AUC 0.666

Test set (Random Forest, used once):
- ROC-AUC: 0.843
- PR-AUC: 0.652
- Recall: 0.794
- Precision: 0.519
- Confusion matrix: 760 true negatives, 275 false alarms, 77 missed
  churners, 297 caught churners

## What I found
- Class weights and SMOTE raise recall from 0.54 to about 0.80 and lower
  precision from 0.65 to about 0.52. The ROC-AUC stays the same: the model
  ranks customers equally well, only the decision changes.
- Default trees were worse than logistic regression (ROC-AUC 0.82 and 0.81
  against 0.85). Limiting their depth fixed most of the overfitting.
- After tuning, all models are equivalent: the differences are smaller than
  the variation between folds. I do not claim a clearly best model. I kept
  Random Forest because it has the best ROC-AUC, a stable recall, and it is
  one of the two models asked in the task.
- The test results are close to the cross-validation results, so I do not
  see overfitting. The ROC-AUC (0.843) is above the 0.80 target.
- The most important features are the month-to-month contract, `tenure`,
  `TotalCharges`, the two-year contract, no online security, no tech support
  and fiber optic. This matches the exploration: 42.7 % of month-to-month
  customers left, against 2.8 % for two-year contracts.

## Limits
- Feature importance shows what the model uses, not what causes churn.
- `tenure` and `TotalCharges` are correlated, so their importance is shared.
- With recall 0.79, about 1 churner in 5 is still missed, and about half of
  the alerts are false alarms.
- I used the default threshold (0.5) and did not optimize it.
- Hyperparameters were chosen by hand, not with a systematic search.
- Results come from one split of a small dataset (5,634 training rows).

## Files
- `01_exploration.ipynb`: data inspection and conclusions.
- `02_modeling.ipynb`: split, preprocessing, models, cross-validation, test
  evaluation, confusion matrix, ROC curve, feature importance.
- `data/Telco-Customer-Churn.csv`: dataset (IBM sample data).

## Run it
```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks in order (Kernel > Restart & Run All).