import threading

_model_persister = None
_model_persister_lock = threading.Lock()

class ModelPersister:
    def __init__(self):
        pass

    @classmethod
    def get_model_persister(cls):
        global _model_persister
        if _model_persister is not None:
            return _model_persister
        with _model_persister_lock:
            if _model_persister is None:
                from .safetensor_model_persister import SafetensorModelPersister
                _model_persister = SafetensorModelPersister()
        return _model_persister

    def model_persist_exist(self, layer_name, saving_path):
        pass

    def persist_model(self, state_dict, layer_name, path):
        pass

    def load_model(self, layer_name, path):
        pass
