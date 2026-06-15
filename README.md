# Crop Yield Prediction (Tomato, Cameroon-focused)

This project demonstrates end-to-end data science for predicting tomato crop yield in an equatorial environment (Cameroon-like conditions). It includes dataset exploration, feature engineering, baseline models, hyperparameter tuning and model saving — all orchestrated in the notebook `tomato-yield-predictor.ipynb`.
Key highlights
- Focus: tomato crop type filtered for temperature (16–44°C), humidity (20–100%) and wind speed (5–25 km/h) to simulate Cameroonian conditions.
- Best performing family: tree-based models. After hyperparameter tuning, XGBoost and RandomForest give the strongest validation performance.
- Most important features: `Soil_Quality` and the `N` fertilizer component (based on XGBoost feature importances).
Quick results (selected)
- Polynomial Linear pipeline (baseline): val RMSE ≈ 10.65 t/ha
- RandomForest (baseline): val RMSE ≈ 3.82 t/ha
- XGBoost (after tuning): cross-val RMSE ≈ 3.66 (mean_test_score ≈ -3.66), val/test around ~4.0 t/ha
Why this matters
- For smallholder farmers, actionable levers such as improving soil quality or adjusting nitrogen application can meaningfully move yield predictions. The model highlights which levers are most influential.
Repository structure
- `tomato-yield-predictor.ipynb` — analysis, training, tuning, visualization and model saving.
- `datasets/crop_yield_dataset.csv` — synthetic dataset used for development.
- `datasets/saved/tomato_crop_yield_prediction_min_max.csv` — saved predictions (post-processed dataset).
- `datasets/saved/feature_importance.csv` — feature importance exported from the best model.
- `models/` — generated performance CSVs and saved model artifacts (see `models/saved/`).

Reproduce locally
1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn xgboost plotly seaborn matplotlib joblib
```

2. Open and run the notebook: launch Jupyter or run the notebook cells in VS Code.

3. To re-run training (from the notebook) the main steps are: data filtering → train/val/test split → scale/encode → train baseline models → grid search the top models → save best model.

Quick commands (from repo root)

```powershell
# Run notebook interactively (VS Code or Jupyter)
jupyter notebook tomato-yield-predictor.ipynb

# Inspect saved artifacts
# Predictions: datasets/saved/tomato_crop_yield_prediction_min_max.csv
# Feature importance: datasets/saved/feature_importance.csv
# Trained model (joblib): models/saved/xgbr_tomato_v1.pkl
```

Notes on metrics
- The notebook evaluates models using root-mean-squared error (RMSE). The hyperparameter search used `neg_root_mean_squared_error` scoring; reported `mean_test_score` is negative RMSE (so -3.66 → RMSE ≈ 3.66 t/ha).
- Depending on your deployment needs, you can convert RMSE to relative errors (RMSPE) if required.

Next steps
- Validate with real field data from the target region to check transferability.
- Add time-aware models (LSTM/Temporal models) if temporal dependencies become critical.
- Build a lightweight API to serve predictions and expose simple recommendations (e.g., increase soil quality, tweak N application).

Credits & license
This repository is a research/demo project. License and attribution can be added as needed.

If you want, I can:
- run the notebook end-to-end and collect the exact metrics reported to a short results file,
- add a `requirements.txt`, or
- create a small inference script that loads the saved model and predicts on new samples.
# crop-yield-prediction
Crop yield prediction model

Dataset Overview:
This dataset provides a detailed and synthetic representation of crop yield and related environmental factors across a decade (2014-2023). The dataset is structured to include daily observations, capturing a wide range of variables that influence crop production. The data is particularly valuable for agricultural researchers, data scientists, and machine learning enthusiasts interested in predicting crop yields based on environmental and soil conditions.

Features Description:
Date:
Type: DateTime
Description: Represents the specific date for each observation, ranging from January 1, 2014, to December 31, 2023. This feature allows for time-series analysis, enabling models to understand trends and seasonal patterns.
Crop_Type:
Type: Categorical
Description: Specifies the type of crop observed on that date. The dataset includes ten major crops:
Wheat
Corn
Rice
Barley
Soybean
Cotton
Sugarcane
Tomato
Potato
Sunflower
Soil_Type:
Type: Categorical
Description: Denotes the type of soil in which the crop is grown. The dataset categorizes soil into five types:
Sandy
Clay
Loamy
Peaty
Saline
Significance: Soil type significantly affects the soil's chemical properties and, consequently, crop yield.
Soil_pH:
Type: Numeric (Float)
Description: Represents the pH level of the soil on the given date. The pH level is crucial for understanding the soil's acidity or alkalinity, which impacts nutrient availability and crop health.
Ranges:
Sandy: 6.0 - 7.5
Clay: 5.5 - 7.0
Loamy: 6.0 - 7.0
Peaty: 4.5 - 6.5
Saline: 7.5 - 8.5
Temperature:
Type: Numeric (Float)
Description: The average daily temperature recorded in Celsius. Temperature plays a critical role in crop development and can vary seasonally:
Winter: Lower temperatures (around 15°C)
Spring: Moderate temperatures (around 25°C)
Summer: Higher temperatures (around 35°C)
Autumn: Cooler temperatures (around 20°C)
Humidity:
Type: Numeric (Float)
Description: The daily average humidity percentage. Humidity affects plant transpiration and moisture levels in the soil, influencing crop yield.
Ranges: Typically between 30% and 80%, adjusted based on temperature.
Wind_Speed:
Type: Numeric (Float)
Description: Represents the average wind speed in km/h. Wind speed can influence evapotranspiration rates and pollination processes.
Generated: Using a normal distribution with a mean of 10 km/h and a standard deviation of 3 km/h.
N:
Type: Numeric (Float)
Description: The nitrogen content in the soil (in ppm). Nitrogen is a vital nutrient that promotes vegetative growth in crops.
Variation: Adjusted based on soil type and crop type, with a base value modified by crop-specific requirements.
P:
Type: Numeric (Float)
Description: The phosphorus content in the soil (in ppm). Phosphorus is essential for root development and energy transfer within plants.
Variation: Similar to nitrogen, it is influenced by both soil type and crop type.
K:
Type: Numeric (Float)
Description: The potassium content in the soil (in ppm). Potassium is crucial for overall plant health and disease resistance.
Variation: Modified based on soil and crop type.
Crop_Yield:
Type: Numeric (Float)
Description: The estimated yield of the crop (in metric tons per hectare) for that particular day. Yield is calculated using a simplified model that considers nutrient levels (NPK), soil pH, temperature, and humidity.
Model: Combines factors like yield potential based on soil nutrients, temperature suitability, and humidity effects to estimate the yield.
Soil_Quality:
Type: Numeric (Float)
Description: An index representing the overall quality of the soil for crop production. It is derived from the balance of NPK levels and soil pH, providing an indicator of the soil's health and fertility.
Data Generation Process:
This dataset was generated using a synthetic model designed to mimic real-world agricultural conditions. The model incorporates domain-specific knowledge about how different environmental factors and soil characteristics influence crop growth. Here’s a breakdown of the data generation process:

Soil pH Calculation:
pH levels were assigned based on typical ranges for each soil type. These ranges reflect the natural variability found in different soil types and their suitability for various crops.

NPK Levels:
The nitrogen, phosphorus, and potassium (NPK) content was determined using base values associated with each soil type. These values were then adjusted based on the specific needs of each crop, simulating how different crops deplete or enrich soil nutrients.

Temperature and Humidity:
Daily temperature and humidity values were generated considering seasonal trends. Winter, spring, summer, and autumn have distinct temperature profiles, which were reflected in the dataset.

Crop Yield Calculation:
Crop yield was estimated using a combination of soil quality (NPK levels and pH), temperature, and humidity. This approach allows for understanding how various factors interact to influence crop productivity.

Daily Data:
The dataset includes daily records over a 10-year period, providing high-resolution data that can be used for detailed analysis or machine learning models, especially in time-series forecasting.

Potential Use Cases:
Predictive Modeling: Use the dataset to train machine learning models for predicting crop yields based on environmental and soil conditions.
Time-Series Analysis: Analyze trends in crop production over time, identifying seasonal patterns and the impact of climate variability.
Soil and Crop Management: Develop insights into how different soil types and nutrient levels affect crop growth, which can inform sustainable agriculture practices.
Climate Impact Studies: Explore how changes in temperature, humidity, and other climatic factors influence agricultural output.
Data Visualization: Create visualizations to better understand the relationships between different variables, such as the impact of soil pH on crop yield.
Dataset Limitations:
Synthetic Nature: The dataset is synthetic, meaning it was generated using modeled assumptions rather than collected from real-world observations. While this allows for controlled variability and completeness, it may not capture all the complexities of real agricultural environments.
Simplified Yield Model: The crop yield calculation is based on a simplified model that may not account for all the factors influencing crop production, such as pests, diseases, or extreme weather events.
Generalized Crop and Soil Data: The crop and soil data are generalized and may not reflect specific regional differences in agricultural practices or soil characteristics.
Conclusion:
This dataset offers a comprehensive and flexible foundation for exploring the complex interactions between environmental factors and crop yield. Its detailed structure and time-series nature make it suitable for a wide range of agricultural studies and predictive modeling tasks. While synthetic, the dataset has been carefully crafted to simulate realistic agricultural scenarios, making it a valuable resource for both educational purposes and advanced research.