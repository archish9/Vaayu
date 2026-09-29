"""S12 T4: the runner is created by every load and consulted by the layer loop -- a
feature the server path never reaches is not shipped (CLAUDE.md §8)."""
import inspect

from vaayu.vaayu_base import VaayuBaseModel


def test_init_installs_decode_graphs_last():
    src = inspect.getsource(VaayuBaseModel.__init__)
    assert "self._decode_graphs = install_decode_graphs(self)" in src
    assert src.index("install_decode_graphs(self)") > src.index("self._resolve_decode_top_k_auto()")


def test_the_layer_loop_asks_the_runner_once_per_forward_and_per_layer():
    src = inspect.getsource(VaayuBaseModel.forward)
    assert "_dg.begin_forward(" in src
    assert "_dg.run(layer_name, layer, seq, kwargs) if _dg_on" in src
    assert src.index("_dg.begin_forward(") < src.index("for i, (layer_name, layer) in tqdm(")
