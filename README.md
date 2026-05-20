# cex-pricing-model
Can You Predict Used Game Prices? — Building a Random Forest Pricing Pipeline on CeX Data

I scraped CeX’s used game catalogue and tried to build a machine learning pricing model for predicting resale prices.

The interesting part is that the model performed much worse than expected achieving a low R² metric — and that turned out to be the most valuable insight.

The dataset contained:
- game titles
- ratings
- retail prices
- trade-in prices

I cleaned the scraped data using pandas and engineered features such as:
- title length 
- franchise extraction
- DLC indicators

I built a full preprocessing + modelling pipeline using scikit-learn:

- missing value imputation (numerical and categorical)
- scaling numerical features
- one-hot encoding categorical variables
- Random Forest regression
- train/test splitting
- model evaluation using MAE and R²

One important finding:

The model achieved weak predictive power (R² ≈ 0.18).

Initially this looked disappointing, but the real issue was the dataset itself.

The dataframe lacked genuinely predictive market information such as:
- release year
- demand
- stock levels
- historical pricing - this one is arguably the most important as I would have engineered different time frames for the model to use for predictive analysis
- review scores
- competitor pricing - would have introduced an element of dynamic pricing to maximise KPIs by considering smaller time intervals with higher volatility

Without those variables, the model mostly learned broad franchise-level pricing patterns rather than true market dynamics.

This project taught me:
- how easily target leakage can inflate ML performance - only the training data should be imputed and scaled, so that the machine learning model is unable to analyse the entire dataframe's distribution and/or be affected by non-existent values.
- why feature engineering matters more than model complexity - the efficiency of the model was largely dependent on the data imputation and feature engineering portions
- the importance of evaluating datasets critically before modelling
