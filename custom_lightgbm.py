from lightgbm import LGBMClassifier, Booster as OriginalBooster
import joblib

class CustomBooster(OriginalBooster):
    def __init__(self, booster):
        self.__dict__.update(booster.__dict__)

    def predict(self, data, *args, **kwargs):
        return super().predict(data, *args, **kwargs)

class CustomLGBMClassifier(LGBMClassifier):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._Booster = CustomBooster(self.booster_)
        return self

def wrap_booster(classifier):
    if hasattr(classifier, 'booster_'):
        classifier._Booster = CustomBooster(classifier.booster_)
    return classifier

def load_custom_pipeline(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    if 'classifier' in pipeline.named_steps:
        classifier = pipeline.named_steps['classifier']
        if isinstance(classifier, LGBMClassifier):
            classifier = wrap_booster(classifier)
            pipeline.named_steps['classifier'] = classifier
    return pipeline

