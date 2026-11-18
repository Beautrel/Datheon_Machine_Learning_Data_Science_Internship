# Task 2: California Housing Prices (OLS, Ridge, Lasso)

By TSAWO Beautrel Carel

## What I did
I used the California Housing dataset to predict house prices. I compared
three linear models: OLS, Ridge and Lasso.

## Data
- 20,640 rows (one row = one neighborhood), 8 features, 1 target (`MedHouseVal`).
- No missing values. All columns are numbers.
- The price is capped at 5.0 ($500,000). About 4.81 % of the rows have this value.

## Steps
1. I looked at the data (`info`, `describe`, histogram, correlation heatmap).
2. I split the data: 80 % train, 20 % test (`random_state=42`), before any scaling.
3. I built a pipeline: `StandardScaler`, then the model. This way the scaler
   learns only from the training data.
4. I tried OLS, RidgeCV and LassoCV. The two last ones choose alpha by
   cross-validation (50 values between 0.00001 and 100).
5. I compared the models with 5-fold cross-validation on the train set.
6. I chose Ridge before looking at the test set, because it is more stable
   when features are correlated. I used the test set only once.
7. I plotted the residuals.

## Results

Cross-validation RMSE on the train set (5 folds):
- OLS: 0.721 (+/- 0.010)
- Ridge: 0.720 (+/- 0.010)
- Lasso: 0.721 (+/- 0.011)

Alphas chosen by cross-validation:
- Ridge: [ALPHA RIDGE]
- Lasso: 0.0007

Test set (Ridge, used only once):
- RMSE: 0.7456
- MAE: 0.5332
- R2: 0.5758

## What I found
- The three models give almost the same score. The difference (0.001) is
  smaller than the variation between folds (0.010).
- I think regularization does not help here because I have many rows (16,512)
  and only 8 features.
- `AveRooms`/`AveBedrms` (0.85) and `Latitude`/`Longitude` (-0.92) are strongly
  correlated. Their coefficients have opposite signs, so I do not interpret
  them one by one.
- `MedInc` is the most useful feature (correlation 0.69 with the price).
- Lasso did not set any coefficient to zero.
- The test RMSE (0.746) is close to the cross-validation RMSE (0.720), so the
  model does not seem to overfit.

## Residuals
- The residuals are centered on 0, with a long tail on the right: the model
  under-estimates the expensive neighborhoods.
- The diagonal line in the residual plot comes from the capped price (5.0).
- Some predictions are negative, and one prediction is near 11.5, which is
  impossible because the maximum price is 5.0.

## Limits
- The price is capped, so the model under-estimates the expensive areas.
- A linear model can predict negative prices or prices above the cap.
- R2 is 0.58, so a big part of the variation is not explained.
- I did not remove outliers (for example `AveOccup` goes up to 1243).
- I checked correlation with the heatmap only, not with VIF.

## Files
- `01_exploration.ipynb`: data inspection and plots.
- `02_modeling.ipynb`: split, pipelines, cross-validation, test, residuals.

## Run it
```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
jupyter notebook
```
Then run the notebooks in order.