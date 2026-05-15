<img src='./Images/idealized_flood_guard.jpeg' style='width: 100%; height: 400px; object-fit: cover;' />

---

<h1 align='center'>
NAIROBI FLOOD GUARD
</h1>

> **Authors**: Group 4

---

<h2 align='center'>
1. OVERVIEW
</h2>

Nairobi Flood Guard is a data science project that addresses the growing threat of flooding across Kenya, motivated by the devastating April 2024 floods abd the recent 2026 floods

It has two components:

- **A flood susceptibility model** and

- **A matatu route optimization system**

It is built using open data and reproducible tools

---

<h2 align='center'>
2. BUSINESS UNDERSTANDING
</h2>

### _Problem Statement_

Flooding in Nairobi is extremely disruptive and leads to loss of life, displacement, and infrastructure damage. Current flood response is largely reactive than predictive

### _Objectives_

- **Flood Susceptibility Prediction**

- **Matatu Route Optimization**

## _Stakeholders_

- Kenya Red Cross / National Disaster Management Unit

- Nairobi City County

- Matatu operators and SACCOs

- General public in flood-prone wards

## _Success Metrics_

- High **recall**

- Route recommendations that successfully avoid confirmed flood zones

- Ward-level risk scores that align with known historically flooded areas

## _Scope and Limitations_

- Labels are based on a single flood event

- GTFS data is from 2019

- Ward-level predictions are coarse

- The model predicts **susceptibility** not exact flood timing or depth

---

 <h2 align='center'>
 3. DATA UNDERSTANDING
 </h2>

This project utilises five datasets, each contributing a different dimension to the flood prediction and route optimization pipeline

### a) SRTM Digital Elevation Model (DEM)

The Shuttle Radar Topography Mission (SRTM) DEM provides elevation data at 90 metre resolution. It was used to derive four terrain features per ward: mean elevation, minimum elevation, maximum elevation, and slope

**Source**: OpenTopography (SRTM GL3 product)

### b) CHIRPS Rainfall Data

The Climate Hazards Group InfraRed Precipitation with Station Data (CHIRPS) provides daily rainfall estimates at approximately 5km resolution. Ninety daily rasters covering February-April 2024 were used to derive three rainfall features per ward: cumulative rainfall, maximum single-day rainfall, and total rainfall in the seven days preceding the April 26 flood event

**Source**: UCSB Climate Hazards Group

### c) UNOSAT Flood Extent - FL20240426KEN

A satellite-derived flood extent geodatabase produced by UNOSAT following the April 2024 Kenya floods. The Kenya-wide maximum flood water extent polygon was used to generate binary flood labels for each ward — flooded (1) or not flooded (0).

**Source:** UNOSAT / UNITAR

### d) Kenya Wards Shapefile

A polygon shapefile of Kenya's 1450 administrative wards including ward name, sub-county, county, and 2009 census population. This served as the spatial backbone of the project - all raster datasets were aggregated to ward level through spatial joins and zonal statistics.

**Source:** Regional Centre for Mapping of Resources for Development (RGMRD)

### e) GTFS Feed 2019 - Nairobi Matatu Network

A General Transit Feed Specification (GTFS) dataset describing Nairobi's matatu public transport network as of 2019, including 136 routes, 4,284 stops, and 36,483 route shape points. This dataset underpins the route optimization component of the project.

**Source:** Digital Matatus Project

### f) Compiled Feature Matrix - floods.gpkg

All datasets were processed and merged into a single GeoPackage file (`floods.gpkg`) containing one row per ward with all features and the flood label. More information about about the compiled feature matrix can be found [here](./Data/floods_description.md).

### _EDA_

After loading and examining the dataset (checking for null values and duplicates), the following visualizations were developed:

#### i) Class Imbalance visualization

<img src='./Images/class_distribution.png' />

The not flooded class accounts for ~79% of the data in the dataset. This confirms that the dataset suffers from class imbalance which was addressed.

#### ii) Feature distributions

<img src='./Images/feature_distributions_by_flood_label.png' />

The feature distribution plots reveal that flooded wards receive less rainfall than non-flooded ones suggesting that, at ward scale, rainfall intensity is a weak standalone predictor of flooding. The elevation features show the clearest separation and are better predictors.

#### iii) Correlation heatmap

<img src='./Images/feature_correlation_matrix.png' />

The correlation heatmap confirms the previous observations. Elevation features dominate - `elevation_min_m` and `elevation_mean_m` carry the strongest negative correlations with flooding (-0.50 and -0.50 respectively), followed by `elevation_max_m` (-0.39) and `slope_mean_deg` (-0.26).

All rainfall features correlate weakly with flooding, with `rain_max_daily_mm` showing virtually no linear relationship (-0.001). The heatmap also reveals high inter-correlation between the three elevation features.

#### iv) Top 10 most flooded counties

<img src='./Images/top_10_flooded_counties.png' />

Again, some of the top 10 counties are ones that do not receive a lot of rainfall e.g. Turkana and Garissa. They experience flooding due to their terrain which, during those rare seasons when it does rain, does not allow the water to drain effectively.

#### _Key Takeaway_

The dataset reveals that in Kenya, flooding is primarily a terrain-driven phenomenon at the ward scale. Low-lying wards flood not necessarily because they receive more rain, but because water from surrounding higher ground drains into them. This has implications for the model - terrain features will dominate predictions, and rainfall adds marginal value at this spatial scale.

---

<h2 align='center'>
4. MODEL BUILDING AND EVALUATION
</h2>

Four classification model families were independently developed and tuned by the project team, each in its own dedicated notebook located in the `Model/Notebooks/` directory:

a) [Logistic Regression](./Models/Notebooks/logistic_notebook.ipynb) (baseline) - saved [here](./Models/best_log_reg_model.pkl)

b) [Random Forest Classifier](./Models/Notebooks/random_forest_notebook.ipynb) - saved [here](./Models/best_random_forest_model.joblib)

c) [XGBoost Classifier](./Models/Notebooks/XGBoost_notebook.ipynb) - saved [here](./Models/best_xg_boost_model.pkl)

d) [Neural Network](./Models/Notebooks/neural_notebook.ipynb) - saved [here](./Models/best_neural_model.keras)

Each model was iteratively improved through hyperparameter tuning, regularisation, and class imbalance handling before the best version was saved.

The following results were obtained:

### a) Logistic Regression (Baseline)

<img src='./Images/logistic_confusion_matrix.png' />

The baseline logistic regression model did not show strong recall on the flooded class, with more than half of its predicted positives being false positives. It shows strong overall recall by keeping false negatives low. This sets a solid performance floor for the more complex models to beat.

### b) Random Forest Model

<img src='./Images/random_forest_confusion_matrix.png' />

The Random Forest improves on the baseline across all metrics. Its ensemble nature - aggregating predictions from many decision trees - allows it to capture non-linear relationships between terrain features and flood risk that the logistic regression cannot.

### c) XGBoost Classifier Model

<img src='./Images/xgboost_confusion_matrix.png' />

The XGBoost model performs better relative to the Random Forest on this dataset in terms of recall. It also has a high accuracy, precision and f1-score. This is likely attributable to its ensemble nature which, like the Random Forest, allows it to capture non-linear relationships between terrain features and flood risk

### d) Neural Network

<img src='./Images/neural_network_confusion_matrix.png' />

The Neural Network significantly underperforms relative to the Random Forest and XGBoost model. Neural networks typically require large amounts of training data to generalise well - with only 1,450 ward-level samples, the model has limited capacity to learn complex spatial patterns compared to tree-based ensembles.

### _Final Evaluation_

Comparing the metrics of all the models:

| Model          |      AUC | accuracy | precision |   recall | f1-score | support |
| :------------- | -------: | -------: | --------: | -------: | -------: | ------: |
| Logistic       |  0.69898 | 0.689655 |   0.60063 | 0.632194 | 0.604335 |     435 |
| Neural_Network | 0.777919 | 0.737931 |   0.65103 | 0.694622 | 0.661215 |     435 |
| Random_Forest  | 0.881322 | 0.822989 |  0.742775 | 0.792306 | 0.760649 |     435 |
| XGBoost        | 0.896913 | 0.813793 |  0.742293 | 0.818291 | 0.762601 |     435 |

The **XGBoost model** achieves some of the highest metrics among all four models

Given that we are looking for the model with the best recall, and, combined with the fact that it has the best AUC and F1-Score, the **XGBoost model** is selected as the final model for flood susceptibility prediction.

The models' ROC curves reinforce this decision with XGBoost achieving the highest AUC (0.9):

<img src='./Images/roc_curves.png' />
