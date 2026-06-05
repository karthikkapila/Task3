import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

num_samples = 500

X = np.random.randint(
    0,
    256,
    size=(num_samples,64,64)
)

X = X.reshape(num_samples,-1)

X = X / 255.0

y = np.random.randint(0,2,size=num_samples)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM...")

svm = SVC(kernel='linear')

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Cat","Dog"]
))