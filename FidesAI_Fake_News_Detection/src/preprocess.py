import re
import spacy
import unicodedata
import pandas as pd
from typing import List, Union

class TextCleaner:
    def __init__(self, use_lemmatization: bool = True):
        self.use_lemmatization = use_lemmatization
        if self.use_lemmatization:
            # Loads a spaCy English language model (en_core_web_sm)
            # Disables the NER (Named Entity Recognition) and parser (Dependency Parsing) components to make it faster and lighter
            # spacy.prefer_gpu()
            self.nlp = spacy.load(
                'en_core_web_sm',
                disable=['ner', 'parser', 'senter']
            )

    def clean_basic_structures(self, text: str):
        if not isinstance(text, str):
            return ""
        # Remove URLs using regex
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tages
        text = re.sub(r'<[^>]*>', '', text)
        # Normalize Unicode characters (e.g., "café" → "cafe") by decomposing
        # handles accents, curly quotes, odd character formatting
        # accented characters into base + diacritic, then remove all non-ASCII
        # characters (accents, special symbols, etc.) to get plain ASCII text
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        # Collapse multiple white spaces or tabs into a single uniform space
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _pre_clean(self, text: str):
        cleaned = self.clean_basic_structures(text).lower()
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
        return cleaned

    def pipeline(self, texts: Union[str, List[str], pd.Series], batch_size: int = 1024 ):
        if isinstance(texts, str):
            if self.use_lemmatization:
                doc = self.nlp(self._pre_clean(texts))
                tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
                return ' '.join(tokens)
            else:
                return self.clean_basic_structures(texts)

        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        if self.use_lemmatization:
            pre_cleaned = [self._pre_clean(t) for t in texts]
            results = []

            for doc in self.nlp.pipe(pre_cleaned, batch_size=batch_size, n_process=1):
                tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
                results.append(' '.join(tokens))
            return results
        else:
            return [self.clean_basic_structures(t) for t in texts]