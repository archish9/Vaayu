
from transformers import GenerationConfig


from .vaayu_base import VaayuBaseModel



class VaayuQWen2(VaayuBaseModel):


    def __init__(self, *args, **kwargs):


        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


