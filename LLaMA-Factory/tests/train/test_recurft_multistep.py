from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from llamafactory.train.sft import recurft


class DummyRecurrent(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states + self.bias

    def has_boundary_head(self) -> bool:
        return False


class DummyBoundaryRecurrent(DummyRecurrent):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__(hidden_size)
        self.boundary = nn.Linear(hidden_size, vocab_size, bias=False)

    def has_boundary_head(self) -> bool:
        return True

    def boundary_logits(self, hidden_states: torch.Tensor, model: nn.Module) -> torch.Tensor:
        return self.boundary(hidden_states)


class DummyProjection(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)


def make_args(step1_weight: float) -> SimpleNamespace:
    return SimpleNamespace(
        recurft_multistep_steps=3,
        recurft_multistep_stride=8,
        recurft_multistep_max_starts=1,
        recurft_multistep_start_selection="linspace",
        recurft_multistep_context_tokens=8,
        recurft_multistep_detach_rollout=True,
        recurft_multistep_step_decay=0.8,
        recurft_multistep_huber_beta=0.1,
        recurft_multistep_mse_loss_clip=0.0,
        recurft_multistep_trim_ratio=0.0,
        recurft_multistep_relative_mse_loss_weight=0.0,
        recurft_multistep_cosine_loss_weight=0.0,
        recurft_multistep_delta_mse_loss_weight=0.0,
        recurft_multistep_delta_cosine_loss_weight=0.0,
        recurft_multistep_contrastive_loss_weight=0.0,
        recurft_multistep_contrastive_temperature=0.1,
        recurft_multistep_contrastive_window=0,
        recurft_multistep_token_ce_loss_weight=1.0,
        recurft_multistep_logit_kl_loss_weight=0.25,
        recurft_multistep_logit_tv_loss_weight=0.5,
        recurft_multistep_logit_temperature=1.0,
        recurft_multistep_logit_context_tokens=8,
        recurft_multistep_step1_logit_loss_weight=step1_weight,
        recurft_multistep_boundary_token_ce_loss_weight=0.0,
        recurft_multistep_boundary_logit_kl_loss_weight=0.0,
        recurft_scheduled_sampling_max_ratio=0.0,
        recurft_scheduled_sampling_warmup_steps=0,
        recurft_scheduled_sampling_start_step=0,
        recurft_loss_on_labels_only=False,
    )


def test_optional_step1_logit_loss_reports_metrics_and_backpropagates() -> None:
    torch.manual_seed(7)
    recurrent = DummyRecurrent(hidden_size=4)
    model = DummyProjection(hidden_size=4, vocab_size=11)
    anchor_hidden = torch.randn(1, 8, 4)
    teacher_logits = torch.randn(1, 8, 11)
    inputs = {
        "input_ids": torch.randint(0, 11, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    with patch.object(
        recurft,
        "_project_anchor_logits",
        lambda model, metadata, anchor_hidden, attention_mask, position_ids: model.proj(anchor_hidden),
    ):
        loss, metrics = recurft._compute_multistep_recurrent_loss(
            recurrent_module=recurrent,
            anchor_hidden=anchor_hidden,
            teacher_logits=teacher_logits,
            inputs=inputs,
            finetuning_args=make_args(step1_weight=2.0),
            model=model,
            metadata={},
            global_step=0,
        )

    assert loss is not None
    loss.backward()
    assert recurrent.bias.grad is not None
    assert metrics["recurft_multistep_step1_logit_loss"] > 0.0
    assert metrics["recurft_multistep_step1_logit_loss_weight"] == 2.0
    assert "recurft_multistep_token_ce_k1" in metrics
    assert "recurft_multistep_logit_kl_k1" in metrics
    assert "recurft_multistep_logit_tv_k1" in metrics


def test_step1_logit_loss_is_backward_compatible_when_disabled() -> None:
    torch.manual_seed(11)
    recurrent = DummyRecurrent(hidden_size=4)
    model = DummyProjection(hidden_size=4, vocab_size=11)
    inputs = {
        "input_ids": torch.randint(0, 11, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    with patch.object(
        recurft,
        "_project_anchor_logits",
        lambda model, metadata, anchor_hidden, attention_mask, position_ids: model.proj(anchor_hidden),
    ):
        loss, metrics = recurft._compute_multistep_recurrent_loss(
            recurrent_module=recurrent,
            anchor_hidden=torch.randn(1, 8, 4),
            teacher_logits=torch.randn(1, 8, 11),
            inputs=inputs,
            finetuning_args=make_args(step1_weight=0.0),
            model=model,
            metadata={},
            global_step=0,
        )

    assert loss is not None
    assert metrics["recurft_multistep_step1_logit_loss"] == 0.0
    assert "recurft_multistep_token_ce_k1" not in metrics


def test_cyclic_start_selection_rotates_across_middle_anchors() -> None:
    legacy = recurft._select_multistep_starts(
        seq_len=1024,
        steps=4,
        stride=64,
        max_starts=2,
        device=torch.device("cpu"),
    )
    assert legacy == [0, 960]

    selections = [
        recurft._select_multistep_starts(
            seq_len=1024,
            steps=4,
            stride=64,
            max_starts=2,
            device=torch.device("cpu"),
            selection="cyclic",
            offset=step,
        )
        for step in range(16)
    ]
    covered = {anchor for starts in selections for anchor in starts}
    assert {64, 128, 512, 576}.issubset(covered)
    assert all(len(starts) == 2 for starts in selections)


def test_multistep_boundary_loss_trains_head_on_rollout_states() -> None:
    torch.manual_seed(17)
    recurrent = DummyBoundaryRecurrent(hidden_size=4, vocab_size=11)
    model = DummyProjection(hidden_size=4, vocab_size=11)
    args = make_args(step1_weight=0.0)
    args.recurft_multistep_token_ce_loss_weight = 0.0
    args.recurft_multistep_logit_kl_loss_weight = 0.0
    args.recurft_multistep_logit_tv_loss_weight = 0.0
    args.recurft_multistep_boundary_token_ce_loss_weight = 1.0
    args.recurft_multistep_boundary_logit_kl_loss_weight = 0.25
    inputs = {
        "input_ids": torch.randint(0, 11, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    loss, metrics = recurft._compute_multistep_recurrent_loss(
        recurrent_module=recurrent,
        anchor_hidden=torch.randn(1, 8, 4),
        teacher_logits=torch.randn(1, 8, 11),
        inputs=inputs,
        finetuning_args=args,
        model=model,
        metadata={},
        global_step=0,
    )

    assert loss is not None
    loss.backward()
    assert recurrent.boundary.weight.grad is not None
    assert recurrent.boundary.weight.grad.abs().sum() > 0
    assert metrics["recurft_multistep_boundary_loss"] > 0.0
    assert "recurft_multistep_boundary_token_ce_k1" in metrics
    assert "recurft_multistep_boundary_teacher_top1_k3" in metrics


if __name__ == "__main__":
    test_optional_step1_logit_loss_reports_metrics_and_backpropagates()
    test_step1_logit_loss_is_backward_compatible_when_disabled()
    test_cyclic_start_selection_rotates_across_middle_anchors()
    test_multistep_boundary_loss_trains_head_on_rollout_states()
    print("4 RecurFT multi-step tests passed")
