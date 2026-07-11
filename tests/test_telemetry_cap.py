"""VaayuTelemetry must not grow unbounded across long generations."""
from vaayu.telemetry import VaayuTelemetry, LayerTiming


def test_telemetry_token_list_is_capped():
    t = VaayuTelemetry()
    for i in range(t.MAX_TOKENS + 100):
        t.begin_token(i)
        t.record_layer(LayerTiming(layer_name="l", file_bytes=0))
        t.end_token()
    assert len(t.tokens) == t.MAX_TOKENS
    # newest survive, oldest dropped
    assert t.tokens[-1].token_index == t.MAX_TOKENS + 99
