# Jewelry-Price-Optimization-Using-Machine-Learning
This project aimed to tackle pricing challenges in the jewelry industry by leveraging machine learning to create a robust pricing strategy. The goal was to predict jewelry prices accurately based on historical sales data, competitor pricing, and customer demographics.

![image](https://media.licdn.com/dms/image/v2/D4E12AQFUHo9u0O0mtA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1690996098366?e=2147483647&v=beta&t=4a-gapXSDbupU3ZU5zUa8tpIlLkM_n6M3afKPpikXEQ)

The workflow began with data collection, including historical sales data, competitor pricing, and customer demographics. The data was cleaned and preprocessed to ensure consistency, with feature engineering applied to create additional variables like price-to-weight ratio and customer segmentation scores. Exploratory Data Analysis (EDA) identified trends, seasonal effects, and purchasing behavior, revealing key factors influencing pricing, such as material type and brand.

Multiple machine learning models were developed and tested, including Linear Regression, Gradient Boosting (AdaBoost and Extra Trees), and CatBoost Regressor. Models were evaluated using Root Mean Squared Error (RMSE) and R² score. The CatBoost Regressor performed best, achieving an RMSE of 15.2 and an R² of 0.89, demonstrating strong predictive accuracy. Hyperparameter tuning further improved model performance.

MLflow was used to track experiments, log metrics, and ensure reproducibility. Parameters, metrics, and artifacts were logged across iterations. The final model was deployed using Flask, allowing real-time pricing predictions. Validation confirmed its reliability in production.

Tools and Technologies

Key tools included:
	•	Python Libraries: Pandas, NumPy, Scikit-learn, CatBoost.
	•	MLflow: For tracking and deployment.
	•	Flask: For real-time predictions.

Results and Impact

The optimized pricing model enabled:
	•	Improved pricing accuracy with an RMSE of 15.2 and R² of 0.89.
	•	Competitive and demand-driven pricing strategies.
	•	Enhanced profitability and scalability for future use.

This project provides a strong foundation for ongoing data-driven pricing optimization in the jewelry industry.
