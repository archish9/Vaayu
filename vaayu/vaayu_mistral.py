
from transformers import GenerationConfig

from .vaayu_base import VaayuBaseModel



class VaayuMistral(VaayuBaseModel):


    def __init__(self, *args, **kwargs):


        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


