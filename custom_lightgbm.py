from lightgbm import Booster as OriginalBooster
import joblib

class CustomBooster(OriginalBooster):
    def _to_predictor(self, pred_parameter):
        if hasattr(self, '_handle'):
            return self._Booster__inner_predictor(pred_parameter)
        else:
            raise AttributeError("'Booster' object has no attribute 'handle' or '_handle'")
       
    def predict(self, data, *args, **kwargs):
        # Use _handle instead of handle
        if hasattr(self, '_handle'):
            self.handle = self._handle
        return super().predict(data, *args, **kwargs)

def load_custom_booster(pipeline_path):
    # Load the pipeline
    pipeline = joblib.load(pipeline_path)
    
    # Extract the classifier from the pipeline and wrap it
    classifier = pipeline.named_steps['classifier']
    classifier._Booster = CustomBooster(model_file=classifier.booster_.model_file_)
    return pipeline

