import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = './train.csv/train.csv'
data = pd.read_csv(file_path)
data.info(), data.head()


# Create target variable: 1 if Model A wins, 0 if Model B wins
data['target'] = np.where(data['winner_model_a'] == 1, 1, 0)

# Feature engineering: length of prompt and responses
data['prompt_length'] = data['prompt'].apply(len)
data['response_a_length'] = data['response_a'].apply(len)
data['response_b_length'] = data['response_b'].apply(len)

# Select features and target for the model
features = data[['prompt_length', 'response_a_length', 'response_b_length']]
target = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse, r2, model.coef_, model.intercept_)
