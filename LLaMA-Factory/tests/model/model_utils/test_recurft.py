import torch
from torch import nn

from llamafactory.model.adapter import _restrict_to_existing_trainable_prefix
from llamafactory.model.model_utils.recurft import RecurFTRecurrentModule


def assert_raises_message(error_type, message: str, callback) -> None:
    try:
        callback()
    except error_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {message}")


class DummyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (self.proj(hidden_states),)


class DummyCausalLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(hidden_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def make_module(
    boundary_rank: int = 0,
    token_rank: int = 0,
    multistep_residual_rank: int = 0,
    multistep_residual_start_step: int = 2,
    multistep_step1_residual_rank: int = 0,
) -> RecurFTRecurrentModule:
    return RecurFTRecurrentModule(
        layers=[DummyLayer(hidden_size=8)],
        target_modules=["proj"],
        rank=2,
        alpha=4,
        dropout=0.0,
        metadata={"loop_layer_ids": [0], "tail_layer_ids": [0]},
        verifier_rank=2,
        verifier_alpha=4,
        verifier_dropout=0.0,
        verifier_probe_init_std=0.02,
        boundary_head_rank=boundary_rank,
        token_conditioning_rank=token_rank,
        multistep_residual_rank=multistep_residual_rank,
        multistep_residual_alpha=multistep_residual_rank * 2,
        multistep_residual_start_step=multistep_residual_start_step,
        multistep_step1_residual_rank=multistep_step1_residual_rank,
        multistep_step1_residual_alpha=multistep_step1_residual_rank * 2,
    )


def test_legacy_recurft_forward_stays_hidden_only() -> None:
    module = make_module()
    hidden = torch.randn(2, 3, 8)

    output = module(hidden)

    assert output.shape == hidden.shape
    assert not module.has_boundary_head()
    assert not module.has_token_conditioning()


def test_boundary_head_produces_trainable_vocab_logits() -> None:
    module = make_module(boundary_rank=4)
    model = DummyCausalLM(hidden_size=8, vocab_size=17)
    hidden = torch.randn(2, 3, 8, requires_grad=True)

    logits = module.boundary_logits(hidden, model)
    logits.float().mean().backward()

    assert logits.shape == (2, 3, 17)
    assert module.boundary_B.weight.grad is not None
    assert any(name.startswith("boundary_") for name in module.state_dict())


def test_batched_boundary_logits_match_single_position_projection() -> None:
    module = make_module(boundary_rank=4).eval()
    model = DummyCausalLM(hidden_size=8, vocab_size=17).eval()
    hidden = torch.randn(1, 4, 8)

    batched = module.boundary_logits(hidden, model)
    positionwise = torch.cat(
        [module.boundary_logits(hidden[:, index : index + 1], model) for index in range(4)],
        dim=1,
    )

    assert torch.allclose(batched, positionwise, atol=1e-6, rtol=1e-6)


def test_token_conditioning_requires_aligned_token_ids_and_round_trips() -> None:
    module = make_module(boundary_rank=4, token_rank=4)
    model = DummyCausalLM(hidden_size=8, vocab_size=17)
    hidden = torch.randn(2, 3, 8)
    token_ids = torch.randint(0, 17, (2, 3))

    output = module(hidden, model=model, token_ids=token_ids)
    clone = make_module(boundary_rank=4, token_rank=4)
    clone.load_state_dict(module.state_dict())

    assert output.shape == hidden.shape
    assert clone.has_boundary_head()
    assert clone.has_token_conditioning()
    assert_raises_message(
        ValueError,
        "token_ids shape",
        lambda: module(hidden, model=model, token_ids=token_ids[:, :-1]),
    )
    assert_raises_message(
        ValueError,
        "requires both",
        lambda: module(hidden, model=model),
    )


def test_stage1_heads_support_mixed_norm_and_projection_dtypes() -> None:
    module = make_module(boundary_rank=4, token_rank=4)
    model = DummyCausalLM(hidden_size=8, vocab_size=17)

    module.boundary_norm.to(torch.bfloat16)
    module.boundary_A.to(torch.float32)
    module.boundary_B.to(torch.float32)
    module.token_conditioning_norm.to(torch.bfloat16)
    module.token_conditioning_A.to(torch.float32)
    module.token_conditioning_B.to(torch.float32)
    model.embed_tokens.to(torch.bfloat16)
    model.model.norm.to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)

    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    token_ids = torch.randint(0, 17, (2, 3))
    conditioned = module._apply_token_conditioning(hidden, token_ids, model)
    logits = module.boundary_logits(conditioned, model)
    logits.float().mean().backward()

    assert conditioned.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    assert module.token_conditioning_B.weight.grad is not None
    assert module.boundary_B.weight.grad is not None


def test_recurrent_lora_merge_preserves_eval_output_and_can_unmerge() -> None:
    torch.manual_seed(0)
    module = make_module().eval()
    hidden = torch.randn(2, 3, 8)
    for child in module.modules():
        if hasattr(child, "lora_B"):
            nn.init.normal_(child.lora_B.weight, std=0.05)

    unmerged = module(hidden)
    merged_count = module.merge_lora_for_inference()
    merged = module(hidden)

    assert merged_count == 1
    assert module.recurrent_lora_is_merged()
    assert torch.allclose(unmerged, merged, atol=1e-5, rtol=1e-5)

    assert module.unmerge_lora_for_inference() == 1
    assert not module.recurrent_lora_is_merged()
    assert torch.allclose(unmerged, module(hidden), atol=1e-5, rtol=1e-5)


def test_recurrent_trainable_scope_freezes_target_base_and_verifier() -> None:
    root = nn.Module()
    root.target_adapter = nn.Linear(8, 8, bias=False)
    root.recurft_recurrent = make_module(
        boundary_rank=4,
        token_rank=4,
        multistep_residual_rank=4,
        multistep_residual_start_step=1,
    )

    retained = _restrict_to_existing_trainable_prefix(root, "recurft_recurrent.")
    trainable = {name for name, param in root.named_parameters() if param.requires_grad}

    assert trainable == set(retained)
    assert trainable
    assert all(name.startswith("recurft_recurrent.") for name in trainable)
    assert not any("base_layer" in name for name in trainable)
    assert not any("verifier" in name for name in trainable)
    assert not root.target_adapter.weight.requires_grad
    assert any("lora_A" in name for name in trainable)
    assert any("boundary_" in name for name in trainable)
    assert any("token_conditioning_" in name for name in trainable)
    assert any("multistep_residual_" in name for name in trainable)


def test_multistep_residual_is_exactly_disabled_for_step1() -> None:
    module = make_module(multistep_residual_rank=4)
    hidden = torch.randn(2, 3, 8)
    step1_before = module(hidden, rollout_step=1)
    step2_before = module(hidden, rollout_step=2)
    assert torch.equal(step1_before, step2_before)

    nn.init.normal_(module.multistep_residual_B.weight, std=0.1)
    step1_after = module(hidden, rollout_step=1)
    step2_after = module(hidden, rollout_step=2)
    assert torch.equal(step1_before, step1_after)
    assert not torch.equal(step1_after, step2_after)

    step2_after.float().mean().backward()
    assert module.multistep_residual_B.weight.grad is not None


def test_multistep_residual_can_train_the_first_rollout_step() -> None:
    module = make_module(multistep_residual_rank=4, multistep_residual_start_step=1)
    hidden = torch.randn(2, 3, 8)

    step1 = module(hidden, rollout_step=1)
    step1.float().mean().backward()

    assert module.multistep_residual_B.weight.grad is not None
    assert torch.count_nonzero(module.multistep_residual_B.weight.grad) > 0


def test_dedicated_step1_residual_does_not_change_later_rollout_steps() -> None:
    torch.manual_seed(0)
    module = make_module(multistep_residual_rank=4, multistep_step1_residual_rank=4)
    hidden = torch.randn(2, 3, 8)
    nn.init.normal_(module.multistep_step1_residual_B.weight, std=0.1)

    step1 = module._apply_multistep_residual(hidden, rollout_step=1)
    step2 = module._apply_multistep_residual(hidden, rollout_step=2)

    assert not torch.equal(step1, hidden)
    assert torch.equal(step2, hidden)


def test_dedicated_step1_residual_receives_step1_gradient_only() -> None:
    module = make_module(multistep_residual_rank=4, multistep_step1_residual_rank=4)
    hidden = torch.randn(2, 3, 8, requires_grad=True)

    module._apply_multistep_residual(hidden, rollout_step=1).sum().backward()

    assert module.multistep_step1_residual_B.weight.grad is not None
    assert torch.count_nonzero(module.multistep_step1_residual_B.weight.grad) > 0
    assert module.multistep_residual_B.weight.grad is None


if __name__ == "__main__":
    test_legacy_recurft_forward_stays_hidden_only()
    test_boundary_head_produces_trainable_vocab_logits()
    test_batched_boundary_logits_match_single_position_projection()
    test_token_conditioning_requires_aligned_token_ids_and_round_trips()
    test_stage1_heads_support_mixed_norm_and_projection_dtypes()
    test_recurrent_lora_merge_preserves_eval_output_and_can_unmerge()
    test_recurrent_trainable_scope_freezes_target_base_and_verifier()
    test_multistep_residual_is_exactly_disabled_for_step1()
    test_multistep_residual_can_train_the_first_rollout_step()
    test_dedicated_step1_residual_does_not_change_later_rollout_steps()
    test_dedicated_step1_residual_receives_step1_gradient_only()
    print("11 RecurFT smoke tests passed")
