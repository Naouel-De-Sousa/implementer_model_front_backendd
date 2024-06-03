from lightgbm import Booster as OriginalBooster, LGBMClassifier
import joblib

class CustomBooster(OriginalBooster):
    def predict(self, data, *args, **kwargs):
        if hasattr(self, '_handle'):
            return super().predict(data, *args, **kwargs)
        raise AttributeError("'Booster' object has no attribute '_handle'")

class CustomLGBMClassifier(LGBMClassifier):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._Booster = CustomBooster(model_str=self.booster_.model_to_string())
        return self

def wrap_booster(classifier):
    if hasattr(classifier, 'booster_'):
        classifier._Booster = CustomBooster(model_str=classifier.booster_.model_to_string())
    return classifier

def load_custom_pipeline(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    if 'classifier' in pipeline.named_steps:
        classifier = pipeline.named_steps['classifier']
        if isinstance(classifier, LGBMClassifier):
            classifier = wrap_booster(classifier)
            pipeline.named_steps['classifier'] = classifier
    return pipeline


