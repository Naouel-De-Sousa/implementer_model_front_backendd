from lightgbm import LGBMClassifier, Booster as OriginalBooster
import joblib

class CustomBooster(OriginalBooster):
    def __init__(self, model_file=None, model_str=None):
        if model_file:
            super().__init__(model_file=model_file)
        elif model_str:
            super().__init__(model_str=model_str)
        else:
            raise ValueError("Either model_file or model_str must be provided")

    def predict(self, data, *args, **kwargs):
        return super().predict(data, *args, **kwargs)

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



