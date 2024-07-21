import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('dc_marvel_movie_performance.csv')

# Convert financial figures from string to numeric
data['Budget'] = data['Budget'].replace('[\$,]', '', regex=True).astype(float)
data['Rotten Tomatoes Critic Score'] = pd.to_numeric(data['Rotten Tomatoes Critic Score'], errors='coerce')

# Create a binary target variable for movie success
data['Is_Successful'] = data['Break Even'].apply(lambda x: 1 if x == 'Success' else 0)

# Selecting features
features = ['Budget', 'Rotten Tomatoes Critic Score', 'Gross to Budget', 'Male/Female-led', 'MCU']
X = data[features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = data['Is_Successful']

# Encoding categorical data
label_encoder = LabelEncoder()
X.loc[:, 'Male/Female-led'] = label_encoder.fit_transform(X['Male/Female-led'])
X.loc[:, 'MCU'] = X['MCU'].apply(lambda x: 1 if x == 'True' else 0)

# Handling missing values
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean of the column
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the model:", accuracy)

from sklearn.model_selection import cross_val_score

# Model Building using Cross-Validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation

print("Cross-validation average score: %.2f" % cv_scores.mean())

from sklearn.model_selection import StratifiedKFold
import numpy as np

# Training the model on the full training data and evaluating on test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test set accuracy: %.2f" % accuracy)

# Model Building using Cross-Validation
model = RandomForestClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5)
cv_scores = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    model.fit(X_train_fold, y_train_fold)
    predictions = model.predict(X_test_fold)
    cv_scores.append(accuracy_score(y_test_fold, predictions))

print("Stratified Cross-validation average score: %.2f" % np.mean(cv_scores))

# Training the model on the full training data and evaluating on test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Test set accuracy: %.2f" % accuracy)

import matplotlib.pyplot as plt

# Fit the model on the full training data
model.fit(X_train, y_train)

# Feature importances
importances = model.feature_importances_
feature_names = ['Budget', 'Rotten Tomatoes Critic Score', 'Gross to Budget', 'Male/Female-led', 'MCU']
indices = np.argsort(importances)[::-1]

# Plot Feature Importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Example new movie data
new_movie_data = {
    'Budget': [150000000],  # example budget
    'Rotten Tomatoes Critic Score': [80],  # example critic score
    # other pre-release features as used in the model
    'Gross to Budget': [2.5],  # estimated ratio (to be adjusted for pre-release data)
    'Male/Female-led': ['Male'],  # leading role gender
    'MCU': [True]  # whether it is a Marvel Cinematic Universe movie
}

# Convert to DataFrame
new_movie_df = pd.DataFrame(new_movie_data)

# Preprocess the new movie data
new_movie_df['Male/Female-led'] = label_encoder.transform(new_movie_df['Male/Female-led'])  # Use same encoder as before
new_movie_df['MCU'] = new_movie_df['MCU'].apply(lambda x: 1 if x else 0)

# Handle any missing values in new data if they are expected
# new_movie_df = pd.DataFrame(imputer.transform(new_movie_df), columns=new_movie_df.columns)

# Scale the new movie data using the same scaler as before
new_movie_scaled = scaler.transform(new_movie_df)

# Predict the success of the new movie
new_movie_success_prediction = model.predict(new_movie_scaled)

# Interpret the prediction
predicted_status = "Success" if new_movie_success_prediction[0] == 1 else "Flop"
print(f"The model predicts the new movie to be a: {predicted_status}")
