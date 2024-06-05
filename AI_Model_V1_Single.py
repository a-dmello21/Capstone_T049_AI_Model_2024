import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1: Read Data
data = pd.read_excel('training_data.xlsx').fillna('NaN')
print(data.columns)
X = data['Input']
y = data['Tag1']

# Step 2: Split Data into Train, Validation, and Test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Preprocess Data (not implemented in this basic example)

print("Vectorize the words")
# Step 4: Feature Extraction
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)

print("Training the model")
# Step 5: Train a Model
model = SVC()
model.fit(X_train_vectorized, y_train)

# Step 6: Evaluate Model Performance on Validation Set
y_val_pred = model.predict(X_val_vectorized)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred, zero_division=0))



# Step 8: Predict Tags
def predict_tag(paragraph):
    paragraph_vectorized = vectorizer.transform([paragraph])
    predicted_tag = model.predict(paragraph_vectorized)[0]
    return predicted_tag
