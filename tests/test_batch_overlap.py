from vaayu.telemetry import VaayuTelemetry


def test_record_batch_overlap_accumulates():
    t = VaayuTelemetry()
    t.record_batch_overlap(layer_idx=0, union_size=12, max_possible=16)
    t.record_batch_overlap(layer_idx=1, union_size=8, max_possible=16)
    stats = t.batch_overlap_summary()
    assert stats["layers"] == 2
    assert abs(stats["mean_ratio"] - ((12/16 + 8/16) / 2)) < 1e-9
