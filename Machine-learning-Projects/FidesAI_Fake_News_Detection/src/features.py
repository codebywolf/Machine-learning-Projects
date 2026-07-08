import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, max_features: int = 50000):
        """
        - We use a cap of 50,000 features to capture all meaningful vocabulary
        - ngram_range=(1, 2) extracts both single words and adjacent word pairs (bigrams).
        - This allows the model to capture contextual phrases like "white house" or "click bait"
          instead of treating the words in isolation.
        - sublinear_tf=True scales word frequencies logarithmically.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True
        )

    def fit_transform(self, train_text: list):
        """
        Fits the vocabulary on training text data and returns the numeric TF-IDF feature matrix.
        """
        logger.info("Fitting vocabulary and transforming training text data...")
        return self.vectorizer.fit_transform(train_text)

    def transform(self, texts: list):
        """
        Transforms validation, test, or inference text using the already fitted vocabulary matrix.
        """
        logger.info("Transforming evaluation text data...")
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path: str):
        """
        Saves the fitted vectorizer object to disk using joblib.
        """
        logger.info(f'Saving vectorizer state to {path}')
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path: str):
        """
        Loads a saved vectorizer state configuration from your local disk.
        """
        logger.info(f'Loading vectorizer state from {path}')
        self.vectorizer = joblib.load(path)