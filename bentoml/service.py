import bentoml
import numpy as np


@bentoml.service()
class IrisClassifier:
    # Declared as a class var
    bento_model = bentoml.models.BentoModel("iris_svm_model:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
