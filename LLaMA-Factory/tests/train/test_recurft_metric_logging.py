from llamafactory.train.sft.trainer import _accumulate_recurft_metrics, _consume_recurft_metrics


def test_recurft_metrics_are_averaged_over_logging_window():
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    _accumulate_recurft_metrics(sums, counts, {"hidden": 0.1, "kl": 0.01})
    _accumulate_recurft_metrics(sums, counts, {"hidden": 0.3, "kl": 0.03})

    assert _consume_recurft_metrics(sums, counts) == {"hidden": 0.2, "kl": 0.02}
    assert sums == {}
    assert counts == {}
