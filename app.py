
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
 
data = {
    'review': [
        "Absolutely love this product! Works perfectly.",
        "Worst purchase I've ever made. Do not recommend.",
        "Pretty decent quality for the price.",
        "It broke after one use. Terrible quality.",
        "Great value for money. Very satisfied!",
        "The item never arrived. Bad experience.",
        "Highly recommend! Exceeded my expectations.",
        "It was okay, nothing special.",
        "Cheap material, not worth it.",
        "Fantastic product! Will buy again."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 1 = P, 0 = N
}
 
df = pd.DataFrame(data)
 
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
model = LogisticRegression()
model.fit(X_train_vec, y_train)
 
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
 
disp = ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, display_labels=["Neg", "Pos"], cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
