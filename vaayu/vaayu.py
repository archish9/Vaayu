

from .vaayu_base import VaayuBaseModel



class VaayuLlama2(VaayuBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

