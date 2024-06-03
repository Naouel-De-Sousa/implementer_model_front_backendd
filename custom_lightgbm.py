from lightgbm import Booster as OriginalBooster
import joblib
from lightgbm import LGBMClassifier
import pandas as pd


class CustomBooster(OriginalBooster):
    def _to_predictor(self, pred_parameter):
        if hasattr(self, '_handle'):
            return self._Booster__inner_predictor(pred_parameter)
        else:
            raise AttributeError("'Booster' object has no attribute 'handle' or '_handle'")
       
    def predict(self, data, *args, **kwargs):
        # Ensure _handle is used if handle is not available
        if hasattr(self, '_handle'):
            self.handle = self._handle
        return super().predict(data, *args, **kwargs)

class CustomLGBMClassifier(LGBMClassifier):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._Booster = CustomBooster(model_file=self.booster_.save_model_to_string())
        return self

def wrap_booster(classifier):
    if hasattr(classifier, 'booster_'):
        classifier._Booster = classifier.booster_
    return classifier

def load_custom_pipeline(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    if 'classifier' in pipeline.named_steps:
        classifier = pipeline.named_steps['classifier']
        if isinstance(classifier, LGBMClassifier):
            classifier = wrap_booster(classifier)
            pipeline.named_steps['classifier'] = classifier
    return pipeline



