"""Módulo para la extración de opiniones."""
import os
import logging
from math import floor, ceil
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm



try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except:
    MODULE = ""

logger = logging.getLogger('CommenlyzerEngine.classifier')
classifier_path = os.environ.get('CLASSIFIER_MODEL_PATH', os.path.join(
    MODULE, 'serialized', 'serialized_opinion_classifier'))
vectorized_path = os.environ.get('VECTORIZER_MODEL_PATH', os.path.join(
    MODULE, 'serialized', 'serialized_opinion_vectorizer'))

logger.info('Classifier path: '+classifier_path)
logger.info('Vectorizer path: '+vectorized_path)
logger.debug(os.environ)

try:
    Classifier = joblib.load(classifier_path)
except FileNotFoundError as e:
    Classifier = svm.SVC()
    logger.warning(str(e)+"\tUsing empty classifier.")
try:
    Vectorizer = joblib.load(vectorized_path)
except FileExistsError as e:
    Vectorizer = TfidfVectorizer()
    logger.warning(str(e)+"\tUsing empty vectorizer.")


def roundx(number):
    """Método que redondea o aproxima un número."""
    #first = int(int((number - floor(number)) >= 0.5) * (floor(number) + 1))
    #second = int(int((number - floor(number)) < 0.5) * (floor(number)))
    # return first + second
    fn = floor(number)
    v = int(number-fn >= 0.5)
    return ceil(number)*v + (1-v)*fn


def extract_opinion(text):
    """
    Explicación:
        Método que recibe un str o una lista de str y devuelve sus clasificaciones.
    Argumentos:
        text: Str o lista de str.
    Retorno:
        answer: str o lista de str con las clasificaciones.
    """
    text = text if isinstance(text, list) else [text]
    data = Vectorizer.transform(text)
    prediction = Classifier.predict(data)
    answers = []
    for item in prediction:
        item = roundx(item)
        if item == 1:
            answers.append('Positive')
        elif item == -1:
            answers.append('Negative')
        else:
            answers.append('Neutral')
    answers = answers[0] if len(answers) == 1 else answers
    return answers


# if __name__ == '__main__':
#     res1, res2 = extract_opinion(['el perro es malo', 'la comida esta buena'])
#     assert res1 == 'Negative'
#     assert res2 == 'Positive'
