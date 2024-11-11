import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = './train.csv/train.csv'
data = pd.read_csv(file_path)

# Target: 1 if Model A wins, 0 if Model B wins
data['target'] = np.where(data['winner_model_a'] == 1, 1, 0)

# Feature engineering: length and word count, similarity approximations
data['prompt_length'] = data['prompt'].apply(len)
data['response_a_length'] = data['response_a'].apply(len)
data['response_b_length'] = data['response_b'].apply(len)
data['prompt_word_count'] = data['prompt'].apply(lambda x: len(x.split()))
data['response_a_word_count'] = data['response_a'].apply(lambda x: len(x.split()))
data['response_b_word_count'] = data['response_b'].apply(lambda x: len(x.split()))
data['prompt_response_a_similarity'] = data['prompt_length'] / (data['response_a_length'] + 1e-6)
data['prompt_response_b_similarity'] = data['prompt_length'] / (data['response_b_length'] + 1e-6)

features = data[['prompt_length', 'response_a_length', 'response_b_length', 'prompt_word_count', 'response_a_word_count', 'response_b_word_count', 'prompt_response_a_similarity', 'prompt_response_b_similarity']]
target = data['target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_tree)
classification_rep = classification_report(y_test, y_pred_tree)

print(accuracy, "\n", classification_rep)