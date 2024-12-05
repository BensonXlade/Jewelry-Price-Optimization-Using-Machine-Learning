# Jewelry-Price-Optimization-Using-Machine-Learning

![](https://media.licdn.com/dms/image/v2/D4E12AQFUHo9u0O0mtA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1690996098366?e=2147483647&v=beta&t=4a-gapXSDbupU3ZU5zUa8tpIlLkM_n6M3afKPpikXEQ)

This project focused on predicting jewelry prices to enable stakeholders to set optimal pricing strategies based on market dynamics and customer preferences. The objective was to build a robust predictive model capable of delivering reliable price estimates by analyzing historical sales data, product attributes, and customer demographics.

Problem and Dataset:
The dataset consisted of jewelry sales records, including features such as material type, main metal/Gem and customer purchase details. To ensure data quality, I addressed missing data, inconsistent formatting, and duplicated rows. I cleaned the dataset by removing duplicates, filling missing values where applicable, and standardizing data formats. Feature engineering played a critical role; I applied label encoding to transform categorical variables such as material and style into numerical representations, which improved the dataset’s predictive power.

Model Development and Evaluation:
I evaluated multiple regression models, including Linear Regression, Adaboost Regressor, CatBoost Regressor, and Extra Trees Regressor, using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R². While all models showed varying levels of performance, CatBoost emerged as the best performer, achieving a training MAE of 183.36 and a test MAE of 183.72, with corresponding R² scores of 0.339 and 0.384. This demonstrated its strong ability to capture non-linear relationships and handle categorical data effectively.

Experiment Tracking with MLflow:
I implemented MLflow to track experiments, log model parameters, metrics, and artifacts for each regression model. This enabled me to compare model performance systematically, streamline hyperparameter tuning, and maintain reproducibility throughout the project. The MLflow UI provided detailed insights into each model’s performance, which aided in decision-making.

Conclusion:
This project successfully demonstrated the application of machine learning in solving a real-world pricing optimization problem. By cleaning the data, applying feature engineering, systematically evaluating models, and leveraging tools like MLflow, I developed a reliable predictive solution. This project offers valuable insights for stakeholders seeking to optimize jewelry pricing strategies.
