from contextlib import contextmanager
from unittest.mock import patch

import torch

from llamafactory.train.sft import recurft


class DummyModel:
    pass


def test_frozen_reference_model_is_used_directly() -> None:
    model = DummyModel()
    ref_model = DummyModel()
    inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    calls = []

    def fake_forward(selected_model, selected_inputs, include_loss=False):
        calls.append((selected_model, selected_inputs, include_loss))
        return "reference-output"

    with patch.object(recurft, "_forward_with_hidden_states", fake_forward), patch.object(
        recurft, "disable_adapter", side_effect=AssertionError("trainable adapter must not be toggled")
    ):
        output = recurft._forward_recurft_reference(model, inputs, ref_model=ref_model)

    assert output == "reference-output"
    assert calls == [(ref_model, inputs, False)]


def test_missing_frozen_reference_keeps_legacy_disabled_adapter_teacher() -> None:
    model = DummyModel()
    inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    events = []

    @contextmanager
    def fake_disable_adapter(selected_model):
        events.append(("enter", selected_model))
        yield
        events.append(("exit", selected_model))

    with patch.object(recurft, "_unwrap_model", return_value=model), patch.object(
        recurft, "disable_adapter", fake_disable_adapter
    ), patch.object(recurft, "_forward_with_hidden_states", return_value="base-output") as forward:
        output = recurft._forward_recurft_reference(model, inputs)

    assert output == "base-output"
    assert events == [("enter", model), ("exit", model)]
    forward.assert_called_once_with(model, inputs, include_loss=False)
