# California Housing Price Prediction

## Introduction

This repository is dedicated to predicting housing prices in California using machine learning models. The project involves applying and evaluating the Decision Tree, Random Forest, and Gradient Boosting models to determine the most accurate predictor. We delve into feature relationships through exploratory data analysis, refine features with engineering strategies, optimize models, especially Gradient Boosting, and assess their performance through various metrics.

## Dataset Analysis

The California Housing dataset encapsulates the diversity of Californian districts, with features like median income, housing age, average rooms per household, and geographical coordinates. Our analysis aims to untangle the complex interplay between these attributes and housing prices.

### Exploratory Data Analysis (EDA)

My EDA began with visualizing each feature's distribution, understanding geographical influences on price, and inspecting feature interdependencies.

 ***<p style="text-align: center;">Histogram:</p>***
  
<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/Histograms.png" width="600" height="500"></p>


 We observed skewed distributions in features such as median income, suggesting a concentration of block groups with lower economic status. The average number of rooms and bedrooms per household also showed skewness, indicating variability in house sizes across districts.

- **Histograms Analysis:**
- MedInc (Median Income): The income distribution appears to be right-skewed, suggesting that there are more neighborhoods with lower median incomes than high. Feature scaling may be necessary to normalize the distribution.

- HouseAge: Many houses are quite old, with a significant number around the 50-year mark. This feature may be important as older houses could affect the housing prices.
  
- AveRooms (Average Rooms): This is also right-skewed, with outliers present indicating some districts have a high number of rooms.
  
- AveBedrms (Average Bedrooms): Similar to AveRooms, this is also right-skewed with some extreme values.
  
- Population: Highly right-skewed, indicating that most districts have a low population, but a few have very high populations.
  
- AveOccup (Average Occupancy): There are districts with unusually high average occupancy, which could be outliers.

<p align="center"><img src="https://github.com/QuantumQuaser/California_Housing_Multi_Model_Prediction/blob/main/Visuals/3_scatter%20plot%20matrices%20to%20visualize%20pairwise%20relationships%20between%20features..png" width="800" height="700"></p>


- **Scatter Plots**: The geographical scatter plot illuminated price hotspots in coastal regions and confirmed the premium on sea-facing properties. It also revealed the clustering of high-price districts in areas with higher median incomes.

  

- **Correlation Matrix**: Our correlation analysis highlighted strong links between house value and median income, while also revealing potential multicollinearity between the number of rooms and bedrooms, prompting us to consider feature engineering to mitigate this.

  ![Correlation Matrix](<CORRELATION_MATRIX_PLACEHOLDER>)

## Feature Engineering Strategies

Informed by our EDA, we embarked on feature engineering to enhance model performance:

1. **Normalization**: Skewed features were normalized to curtail the influence of extreme values.
2. **Outlier Handling**: We capped outliers to temper their impact, ensuring our models remained focused on prevalent trends.
3. **Feature Creation**: New features, such as rooms per occupant and bedrooms per room, were derived to capture underlying patterns that raw features might obscure.

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

## Residual Analysis

The residual plots provide a window into each model's prediction errors:

- **Decision Tree Residuals**: Showed a scattered spread indicating variance in prediction errors, reflecting overfitting tendencies.

  ![Residual Plot for Decision Tree](<RESIDUAL_PLOT_DT_PLACEHOLDER>)

- **Random Forest Residuals**: Demonstrated a tighter cluster around the center, suggesting better performance but with room for improvement.

  ![Residual Plot for Random Forest](<RESIDUAL_PLOT_RF_PLACEHOLDER>)

- **Gradient Boosting Residuals (Before Optimization)**: While decent, there was noticeable spread indicating potential for optimization.

  ![Residual Plot for Gradient Boosting](<RESIDUAL_PLOT_GB_PLACEHOLDER>)

- **Optimized Gradient Boosting Residuals**: Post-tuning, the residuals clustered tightly around zero, illustrating a marked leap in predictive accuracy.

  ![Residual Plot for Optimized Gradient Boosting](<RESIDUAL_PLOT_OGB_PLACEHOLDER>)

## Performance Results

The models' predictive prowess is encapsulated in the following table:

| Model | MSE | MAE | RMSE | CV RMSE |
| --- | --- | --- | --- | --- |
| Decision Tree | `0.5214` | `0.4600` | `0.7221` | `0.7223` |
| Random Forest | `0.2537` | `0.3274` | `0.5037` | `0.5081` |
| Gradient Boosting (Before Opt.) | `0.2868` | `0.3662` | `0.5355` | `0.5264` |
| Gradient Boosting (Optimized) | `0.2063` | `0.2986` | `0.4543` | `0.4609` |

## Conclusion

The journey from raw data to refined predictions has been illuminating. The optimized Gradient Boosting model, with its superior error metrics, stood out as the clear winner. It exemplified how targeted optimization can unleash a model's true potential. The Decision Tree, while insightful, was outshone due to its simplicity. The Random Forest struck a middle ground, robust yet outperformed by its Gradient Boosting counterpart.



