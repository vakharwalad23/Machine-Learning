import bentoml

# Load the model
model = bentoml.sklearn.get("iris_svm_model:latest")

sklearn_model = model.load_model()
# result = sklearn_model.predict([[4.7, 3.2, 1.3, 0.2]]) -> [0]
result = sklearn_model.predict([[5.2, 2.7, 3.9, 1.4]])  # -> [1]
print(result)
