Approach:
Used Pandas Library for working on the data. 
The data was highly imbalanced, with around 99.5% of the data belonging to one class and only 0.5% to the other class.
Analysed the dataset thoroughly for missing values, changed few data types.

Engineered new columns from the original columns. Some of them, which worked are:
is_deliquent: I have engineered this feature from m1 to m12 columns. I have calculated the sum through m1-m12 and made the new column as '1', if the sum is greater than '0', and made the column to '0', if sum of columns is '0'. This feature proved to be pretty important for my predictions.

borrower_credit_performance: I have took some reference from some financial websites and categorized credit score to 3 categories as 'exceptional', 'good' and 'bad'. Obviously Bad credit score, had many defaulters. This feature was also helpful for the final model.

Tried other features like income and total_assets from columns like loan_to_value and debt_income_ratio, but they did not work.
Dropped co-borrower_credit_performance, as most of the rows did not have a co-borrower and respective credit score is 0.

Converted unpaid_principal_bal, debt_to_income_ratio, loan_to_value columns to categorical and dropped them.

Scaled all Numeric features with Robust scaler from sklearn and converted all categorical columns to their One-Hot encoded equivalents.

Used SMOTE technique for generating sythetic samples for minority class. Used 'svm' SMOTE and generated minority samples with 15% ratio of majority samples.

As, the data is huge and imbalanced, used Boosting techniques like XGBoost. It performed well compared to basic models like Linear Regression and Descision Trees.
Used XGBoost classifier for final model building. Got F1-score of around 0.55 on Validation data.

Computed Permutation Importance of features with eli5 library.
