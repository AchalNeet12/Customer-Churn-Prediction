# Customer Churn Prediction
---
---
## 📝 Overview
This project aims to analyze customer churn patterns within a telecom company to understand the underlying factors contributing to customer attrition. The study leverages historical customer data, including demographic information, service usage metrics, and customer interaction logs. Through exploratory data analysis and predictive modeling techniques, key drivers of churn will be identified, such as service dissatisfaction, pricing strategies, contract terms, and customer support effectiveness.
  - **Objective:**
    -  Build a predictive model to classify customers into "churn" or "non-churn" categories.
    -  Provide actionable insights to enhance customer retention strategies.
   
  - **Business Impact:**
    -  Reduce customer acquisition costs by retaining existing customers.
    -  Identify factors leading to customer dissatisfaction and churn.
---
## ⚙ Algorithms Used
   1. **LogisticRegression**

  - Description:
      - Logistic Regression is a linear model for binary classification. It predicts the probability of the target variable belonging to a particular class using a logistic function.
 - Advantages:
     - Simple and interpretable.
        - Works well when there is a linear relationship between features and the target variable
  - Accuracy: 80.15%
   3. **DecisionTreeClassifier**
      
  - Description:
      - A Decision Tree is a non-linear model that splits the dataset into subsets based on feature conditions. Each split is chosen to maximize information gain or minimize Gini impurity.
  - Advantages:
      - Easy to understand and interpret.
      - Captures non-linear relationships effectively.
  - Accuracy: 79.91%
   5. **RandomForestClassifer**
      
  - Description:
      - Random Forest is an ensemble learning method that constructs multiple decision trees during training and merges their outputs to improve accuracy and reduce overfitting.
  - Advantages:
      - Robust to overfitting compared to a single decision tree.
      - Handles missing values and outliers effectively.
  - Accuracy: 81.75%         
   7. **AddaboostClassifier**
      
  - Description:
     - Adaptive Boosting (AdaBoost) combines multiple "weak learners" (typically shallow decision trees) into a single "strong learner." It assigns weights to misclassified samples, allowing the 
       model to focus on hard-to-classify instances in subsequent iterations.
  - Advantages:
     - Improves accuracy by iteratively correcting errors.
     - Effective for imbalanced datasets.
  - Accuracy:82.09%
   5. **GradientBoostClassifer**
  
  - Description:
     - Gradient Boosting is another ensemble method that builds models sequentially. It optimizes a loss function by training each new model to correct the errors of the previous model. Unlike 
                AdaBoost, Gradient Boosting minimizes the error directly using gradient descent.
  - Advantages:
     - Provides high accuracy for both classification and regression tasks.
     - Handles complex, non-linear relationships between features and the target.
 - Accuracy: 82.81%
   6. **XGBClassifier**
  
  - Description:
     - XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting that builds trees sequentially, with each tree correcting errors from the previous one. It uses regularization to 
       reduce overfitting and improve model generalization.
  - Advantages:
     - High accuracy and fast computation.
     - Handles large datasets effectively.
     - Built-in regularization to prevent overfitting.
  - Accuracy: 84.02% (Highest accuracy)

   ## Best Model: XGBClassifier
              - Achieved the highest accuracy among all models.
              - Captures complex patterns and non-linear relationships effectively.
              - Tuned for optimal performance with high reliability.
---
## 📦 About Data
It Divided into 3 Type
1.Demographic information
2.Customer Acconting Information
3.Service information
4.Traget variable

1. **Demographic information**
   -• gender: Whether the customer is a male or a female.
   -• SeniorCitizen: Whether the customer is a senior citizen or not (1, 0).
   -• Partner: Whether the customer has a partner or not (Yes, No)
   -• Dependents : Whether the customer has dependents or not (Yes, No)
2. **Customer Acconting Information**
   -• Contract: The contract term of the customer (Month-to-month, One year, Two year)
   -• PaperlessBilling : Whether the customer has paperless billing or not (Yes, No)
   -• MonthlyCharges: The amount charged to the customer monthly
   -• TotalCharges: The total amount charged to the customer
   -• tenure: Number of months the customer has stayed with the company
   -• PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (au card (automatic))
   -• customeriD: Customer ID
3. **Service information**
   -• PhoneService: Whether the customer has a phone service or not (yes, No)
   -• MultipleLines: Whether the customer has multiple lines or not (yes, No, No phone service)
   -• InternetService: Customer's internet service provider (DSL, Fiber optic, No)
   -• OnlineSecurity: Whether the customer has online security or not (yes, No, No internet service)
   -• OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
   -• DeviceProtection: Whether the customer has device protection or not (yes, No, No internet service)
   -• TechSupport: Whether the customer has tech support or not (yes, No, No internet service)
   -• Streaming TV: Whether the customer has streaming TV or not (Yes, No, No internet service)
   -• StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
4. **Traget variable**
   -• Churn: Whether the customer churn or not (yes or No)
---
## 🤖 Technologies Used
  - **Python:** Core programming language for the project.
  - **Pandas, NumPy:** Data preprocessing and analysis.
  - **Scikit-learn:** Implementing and evaluating machine learning algorithms.
  - **Matplotlib, Seaborn:** Visualizing data and insights.
  - **SMOTE:** Handling imbalanced datasets.
  - **Streamlit:** Creating an interactive and user-friendly web interface.
  - **Base64:** Encoding images for custom background styling in Streamlit.
  - **Joblib:** Saving and loading the trained machine learning model.
---
## 🔍 Data Preprocessing Steps
1. **Data Cleaning**
 - Handled missing values by imputing or dropping rows/columns as necessary.
 - Converted TotalCharges to numeric and addressed non-numeric anomalies.
2. **Feature Encoding**
 - Converted categorical variables into numerical format using label encoding and one-hot encoding.
 - Example:
   Gender: Male → 1, Female → 0
   Contract: Month-to-month → 0, One year → 1, Two year → 2
3. **Feature Scaling**
 - Applied standardization or normalization to continuous variables like MonthlyCharges and TotalCharges for algorithm compatibility.
4. **Feature Engineering**
 - Created derived features like tenure_group to group customers based on their tenure duration.
 - Combined MultipleLines and PhoneService for better insights.
5. **Handling Class Imbalance**
 - Used techniques like oversampling (SMOTE) or class weighting to address the imbalance in churn labels.
6 **Train-Test Split**
 - Split the data into training (80%) and testing (20%) datasets for evaluation.
---
## 📌 Model Deployment
   - The XGBClassifier model has been deployed in a Streamlit application. The app allows users to input customer details and receive a prediction on whether the customer is likely to churn or stay. The 
     model's high accuracy ensures reliable predictions, providing actionable insights for customer retention strategies.
---
## 📈 Result
- Multiple machine learning models were tested for customer churn prediction.
- XGBClassifier outperformed others with the highest accuracy and was selected for deployment.
- The final model effectively identifies customers who are likely to churn with improved precision and recall.
---
## 📊 Model Output
- The user-friendly Streamlit interface accepts customer details as input.
- Outputs whether the customer is "likely to leave" or "likely to stay."
- The predictions are based on the XGBClassifier's trained model, ensuring high accuracy and reliability.
---
## 🎯 Conclusion
- Customer churn prediction helps businesses take proactive measures to retain customers.
- Using XGBClassifier, the system achieved robust results, emphasizing the importance of boosting techniques for such tasks.
- The deployed solution provides actionable insights into customer behavior, aiding in retention strategies.



















