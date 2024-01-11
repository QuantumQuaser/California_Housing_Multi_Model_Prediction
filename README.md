# California Housing Price Prediction

## Introduction

This repository is dedicated to predicting housing prices in California using machine learning models. The project involves applying and evaluating the Decision Tree, Random Forest, and Gradient Boosting models to determine the most accurate predictor. We delve into feature relationships through exploratory data analysis, refine features with engineering strategies, optimize models, especially Gradient Boosting, and assess their performance through various metrics.

## Dataset Analysis

The California Housing dataset encapsulates the diversity of Californian districts, with features like median income, housing age, average rooms per household, and geographical coordinates. Our analysis aims to untangle the complex interplay between these attributes and housing prices.

### Exploratory Data Analysis (EDA)

My EDA began with visualizing each feature's distribution, understanding geographical influences on price, and inspecting feature interdependencies.

 # <p style="text-align: center;">Histogram:</p>
  
<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Histograms.png" width="600" height="500"></p>


 We observed skewed distributions in features such as median income, suggesting a concentration of block groups with lower economic status. The average number of rooms and bedrooms per household also showed skewness, indicating variability in house sizes across districts.

- **Histograms Analysis:**
- MedInc (Median Income): The income distribution appears to be right-skewed, suggesting that there are more neighborhoods with lower median incomes than high. Feature scaling may be necessary to normalize the distribution.

- HouseAge: Many houses are quite old, with a significant number around the 50-year mark. This feature may be important as older houses could affect the housing prices.
  
- AveRooms (Average Rooms): This is also right-skewed, with outliers present indicating some districts have a high number of rooms.
  
- AveBedrms (Average Bedrooms): Similar to AveRooms, this is also right-skewed with some extreme values.
  
- Population: Highly right-skewed, indicating that most districts have a low population, but a few have very high populations.
  
- AveOccup (Average Occupancy): There are districts with unusually high average occupancy, which could be outliers.

# Scatterplots:

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/scatter%20plots.png" width="500" height="500"></p>


- **Scatter Plots**: The geographical scatter plot illuminated price hotspots in coastal regions and confirmed the premium on sea-facing properties. It also revealed the clustering of high-price districts in areas with higher median incomes.

# correlation Matrix"
  
<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/correlation%20matrices.png" width="500" height="500"></p>

- **Correlation Matrix**: correlation analysis highlighted strong links between house value and median income, while also revealing potential multicollinearity between the number of rooms and bedrooms, prompting us to consider feature engineering to mitigate this.

# Scatter Plot Matrix Analysis

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/3_scatter%20plot%20matrices%20to%20visualize%20pairwise%20relationships%20between%20features..png" width="800" height="700"></p>

Shows relationships between all pairs of features, providing insights into potential multicollinearity and redundant features.

# Box Plot Analysis

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/4_box%20plot%20to%20check%20outliers.png" width="500" height="600"></p>

Indicates several outliers, especially in the population, which might need to be handled either by capping or by removing these outliers to prevent them from skewing the model's performance.
  

## Feature Engineering Strategies

by our EDA, we embarked on feature engineering to enhance model performance:

**Outlier Treatment**: Cap the features with outliers such as AveRooms, AveBedrms, and Population to reduce their effect.

**Feature Scaling**: Since the scales of the features vary widely, apply Min-Max Scaling or Standardization to ensure that all features contribute equally to the distance computations in the models.

**Feature Transformation**: Apply transformations such as logarithmic or square root to highly skewed features to normalize their distributions.

**Location Features**: Create new features that capture the proximity to specific landmarks or city centers.

**Interaction Terms**: Given the correlation between features, create interaction terms that might help in capturing the combined effect on the housing prices (e.g., income to average occupancy ratio).

These strategies were chosen to address specific data characteristics, aiming to distill clearer signals for our predictive models.

## Model Preparation and Training

We prepared our data, splitting it into training and test sets, and scaled the features to provide a uniform playing field for model training.

### Decision Tree
A straightforward model that sets the baseline for performance but may overfit our diverse dataset.

### Random Forest
An ensemble approach that builds resilience against overfitting and benefits from the collective decision-making of multiple trees.

### Gradient Boosting (Before Optimization)
An advanced ensemble method that showed promise but was initially hampered by suboptimal hyperparameters.

## Optimizing Gradient Boosting

Through RandomizedSearchCV, we tuned Gradient Boosting's hyperparameters, such as the number of trees (n_estimators), learning rate, and tree depth, to balance learning granularity with generalization. This meticulous tuning significantly honed the model's accuracy.

## Performance and Residual Analysis

The residual plots provide a window into each model's prediction errors:

# Decision Tree Performance:

MSE: 0.5214
MAE: 0.4600
RMSE: 0.7221
Cross-validated RMSE: 0.7223 ± 0.0190

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Residual%20Plot%20for%20decision%20tree.png" width="500" height="600"></p>

- **Decision Tree conclusion **: The Decision Tree shows the highest errors among the models. Decision Trees are prone to overfitting, which is likely the cause of the higher error rates. The residual plot for the Decision Tree exhibits a relatively wider spread of residuals, indicating more significant prediction errors, and there's a noticeable trend in the residuals as the observed values increase, suggesting the model isn't capturing all the underlying patterns.

  # Random Forest Performance:

 MSE: 0.2537
MAE: 0.3274
RMSE: 0.5037
Cross-validated RMSE: 0.5081 ± 0.0164

  <p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Residual%20Plot%20for%20Random%20Forest.png" width="500" height="600"></p>
  
- **Random Forest conclusion**: Random Forest has much lower error rates across all metrics compared to the Decision Tree, which is expected given that Random Forest is an ensemble model that mitigates overfitting by averaging multiple decision trees. The residual plot for the Random Forest shows a tighter cluster of residuals around zero, indicating better performance, although there are still some outliers, particularly for higher-valued homes.

  
- # Gradient Boosting Performance (Before Optimization)**: While decent, there was noticeable spread indicating potential for optimization.

- MSE: 0.2868
MAE: 0.3662
RMSE: 0.5355
Cross-validated RMSE: 0.5264 ± 0.0172

  <p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Residual%20plot%20for%20gradient%20boosting.png" width="500" height="600"></p>

  **Gradient boosting conclusion:** Gradient Boosting initially performed better than the Decision Tree but not as well as the Random Forest. This could be due to the model not being fully tuned to the dataset. The residual plot before optimization shows a slightly wider spread than Random Forest, indicating room for improvement.

- **Optimized Gradient Boosting performance**: 

  Optimized Gradient Boosting Performance:
  
MSE: 0.2063
MAE: 0.2986
RMSE: 0.4543
Cross-validated RMSE: 0.4609 ± 0.0148

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Residual%20plot%20for%20optimized%20gradient%20boosting.png" width="500" height="600"></p>

After optimization, the Gradient Boosting model shows the best performance with the lowest error rates across all metrics. The optimization of hyperparameters has clearly paid off, as evidenced by the reduced errors. The residual plot for the optimized Gradient Boosting model has the residuals more tightly clustered around zero and fewer extreme values than before optimization, indicating that the model's predictions are more accurate and consistent.

## Performance Results

The models' predictive prowess is encapsulated in the following table:

| Model | MSE | MAE | RMSE | CV RMSE |
| --- | --- | --- | --- | --- |
| Decision Tree | `0.5214` | `0.4600` | `0.7221` | `0.7223` |
| Random Forest | `0.2537` | `0.3274` | `0.5037` | `0.5081` |
| Gradient Boosting (Before Opt.) | `0.2868` | `0.3662` | `0.5355` | `0.5264` |
| Gradient Boosting (Optimized) | `0.2063` | `0.2986` | `0.4543` | `0.4609` |

# Conclusion

## Best and Worst Performing Models:
Based on the data provided, the Optimized Gradient Boosting model is the best performer. The optimization process allowed it to effectively learn the nonlinear relationships and interactions within the data without overfitting, which is evident from the tight clustering of residuals and the lowest error metrics.

The Decision Tree model is the worst performer due to its high error metrics and the clear pattern in the residuals, indicating a lack of model complexity to capture the relationships in the data fully.

The Random Forest model sits in between, with performance significantly better than the Decision Tree but not quite as good as the optimized Gradient Boosting. Its ability to reduce overfitting by averaging multiple trees makes it a strong model, though it may still be improved with further hyperparameter tuning.



