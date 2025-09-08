import bentoml

from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train a model
model = svm.SVC(gamma='scale')
model.fit(X, y)

# Save the model with BentoML
saved_model = bentoml.sklearn.save_model("iris_svm_model", model)
print(f"Model saved successfully with tag: {saved_model}")

# Model(tag="iris_svm_model:q6lojremzwih3hrm")
