from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class TFIDFClassifier(Classifier):
    def __init__(
        self,
        max_features: int = 5000,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, samples: List[EvaluationSample]) -> None:
        texts = [sample.text for sample in samples]
        labels = [sample.label for sample in samples]

        x_train = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x_train, labels)
        self.is_fitted = True

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")

        texts = [sample.text for sample in samples]
        x_test = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(x_test)

        return [int(pred) for pred in predictions]
