import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack
import pickle

# Load the dataset
df = pd.read_csv("dataset_1.csv")

# Separating the data
y = df['label']
x = df['text']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
X_tfidf = vectorizer.fit_transform(x)

# Define classifiers
clf1 = RandomForestClassifier(n_estimators=50)
clf2 = XGBClassifier(n_estimators=50)
clf3 = LogisticRegression()
clf4 = SVC(kernel='sigmoid', gamma=1.0)
clf5 = AdaBoostClassifier(n_estimators=50)
clf6 = BaggingClassifier(n_estimators=50)
clf7 = ExtraTreesClassifier(n_estimators=50)

# Combine classifiers into a list
classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]

# Combine predictions as additional features
X_combined = X_tfidf
for clf in classifiers:
    # Perform cross-validation and get predictions
    pred_cv = cross_val_score(clf, X_tfidf, y, cv=5, scoring='accuracy')

    # Print cross-validation results for each classifier
    print(f'Average Accuracy for {clf.__class__.__name__}: {pred_cv.mean()}')

    # Train each classifier on the entire dataset
    clf.fit(X_tfidf, y)

    # Get predictions on the test set
    pred_test = clf.predict(X_tfidf)

    # Combine predictions as additional features
    X_combined = hstack([X_combined, pred_test.reshape(-1, 1)])

# Train a new model on the combined features
new_model = make_pipeline(RandomForestClassifier())
new_model.fit(X_combined, y)

# Perform cross-validation on the new model
cv_results_new = cross_val_score(new_model, X_combined, y, cv=10, scoring='accuracy')
print(f'Average Accuracy with new model: {cv_results_new.mean()}')

# Save the new model
pickle.dump(new_model, open("new_model_with_cv.pkl", "wb"))
pickle.dump(vectorizer, open("new_vectorizer_with_cv.pkl", "wb"))
