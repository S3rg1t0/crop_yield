# CROP FIELD

### Description:

This dataset is designed for the prediction of crop yield in agricultural farms. It includes 3000 data points generated through a set of simulated environmental and farming features. These features—such as rainfall, soil quality, farm size, sunlight hours, and fertilizer usage—have been carefully selected to represent key factors that affect crop productivity.

The crop yield values (in tons per hectare) are generated using a predefined equation that combines the effects of these factors. This dataset is ideal for exploring regression models, testing machine learning algorithms, or building predictive models for agricultural analysis.

### Context:
The dataset was created to help simulate the relationship between environmental factors and crop productivity. The values used in this dataset reflect realistic ranges for agricultural farms across different climates and farming practices. It’s useful for researchers, data scientists, and agricultural professionals who want to experiment with crop prediction models or gain insights into farming efficiency.

### Features:
The dataset includes the following features:

* rainfall_mm: The average amount of rainfall (in millimeters) during the growing season. (Range: 500 mm to 2000 mm)
* soil_quality_index: A numeric rating of soil quality, on a scale from 1 (poor) to 10 (excellent). (Range: 1 to 10)
* farm_size_hectares: The size of the farm in hectares. (Range: 10 to 1000 hectares)
* sunlight_hours: The average daily hours of sunlight during the growing season. (Range: 4 to 12 hours per day)
* fertilizer_kg: The amount of fertilizer used per hectare (in kilograms). (Range: 100 to 3000 kg/hectare)
* The target variable is crop_yield, which represents the predicted yield of the crop in tons per hectare, calculated using a linear equation based on the above features.

### Usability:
This dataset is useful for various applications, including:

### Uses:
Machine Learning: Train regression models to predict crop yield based on the input features.
Predictive Analytics: Build models to estimate the potential yield based on weather forecasts, farm characteristics, and farming practices.
Data Science Projects: Practice data preprocessing, feature engineering, and model evaluation in the agricultural domain.