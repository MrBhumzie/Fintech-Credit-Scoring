`Overview`

This project develops a credit scoring model using Logistic Regression to predict loan defaults, leveraging a Kaggle dataset ("Credit Risk Dataset"). Achieving a 0.76 accuracy on both cross-validation and test sets, it demonstrates robust generalization for fintech applications, such as in Nigeria's informal economy.

`Methodology`

`Data Preprocessing:` Handled missing values with mean imputation and encoded categorical variables (e.g., 'person_home_ownership', 'loan_intent') using one-hot encoding.

`Class Imbalance:` Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the minority 'default' class, ensuring the model learns from both repaid and defaulted loans effectively.

`Modeling:` Used Logistic Regression with class_weight='balanced' for fair prediction, achieving stable 0.76 accuracy.

`Feature Importance:` Visualized coefficients to identify key predictors like 'loan_int_rate' and 'person_income'.

`Results`

Accuracy: 0.76 (CV: 0.76 ± 0.00, Test: 0.76), indicating strong performance.
Visualization: Plots show feature importance and the trend of default probability vs. income, aiding interpretable decision-making.

`Design Thinking Approach (Architecture Background)`

Drawing from my B.Sc. in Architecture (Enugu State University of Science and Technology, 2020), I applied design thinking by iteratively refining the model architecture. Just as architectural designs balance aesthetics and functionality, this model balances accuracy and interpretability, using feature importance to guide feature selection—mirroring structural optimization in building design.

`Project Management (Google PM Skills)`

Utilizing skills from my Google Project Management certification, I structured the project with clear milestones: data loading, preprocessing, modeling, and deployment. This ensured efficient resource use (e.g., Kaggle data) and timely delivery, aligning with PM principles of scope, time, and quality management.

`Next Steps`

Deployed via Flask for real-time prediction.
Shared on GitHub for portfolio visibility.

`Usage`

Clone the repository.

Install dependencies: pip install -r requirements.txt.

Run the Flask app: python app.py.

Access via http://localhost:5000/predict with JSON input.

`Acknowledgments`

Kaggle community for the dataset.

xAI for guidance in model development.
