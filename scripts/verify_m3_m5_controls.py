#!/usr/bin/env python3
"""Forward and structural smoke checks for the controlled M3/M5 ablations."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.rowcol_convmixer import (
    WeightedMultiplicativeRowColConvMixer,
    WeightedRowColConvMixer,
)


def trainable_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def rowcol_parameter_names(model):
    return [
        name
        for name, _ in model.named_parameters()
        if ".rowcol_branch." in name
    ]


def assert_forward_order(branch):
    order = []
    handles = []
    module_names = [
        "horizontal_conv",
        "vertical_conv",
        "h_norm",
        "w_norm",
        "activation",
        "bn",
    ]
    for name in module_names:
        module = getattr(branch, name)
        handles.append(
            module.register_forward_hook(
                lambda _m, _i, _o, name=name: order.append(name)
            )
        )

    try:
        branch(torch.randn(2, branch.dim, branch.feature_size, branch.feature_size))
    finally:
        for handle in handles:
            handle.remove()

    assert order == module_names, order


def assert_legacy_default_equations(original_m3_branch, original_m5_branch):
    feature_size = original_m3_branch.feature_size
    inputs = torch.randn(2, original_m3_branch.dim, feature_size, feature_size)
    with torch.no_grad():
        m3_h = original_m3_branch.horizontal_conv(inputs)
        m3_v = original_m3_branch.vertical_conv(inputs)
        expected_m3 = original_m3_branch.bn(
            original_m3_branch.activation(
                original_m3_branch.branch_weight[0] * m3_h
                + original_m3_branch.branch_weight[1] * m3_v
            )
        )
        torch.testing.assert_close(
            original_m3_branch(inputs), expected_m3, rtol=0, atol=0
        )

        m5_h = original_m5_branch.h_norm(original_m5_branch.horizontal_conv(inputs))
        m5_w = original_m5_branch.w_norm(original_m5_branch.vertical_conv(inputs))
        expected_m5 = original_m5_branch.bn(
            original_m5_branch.activation(
                (m5_h * original_m5_branch.p_w)
                * (original_m5_branch.p_h * m5_w)
            )
        )
        torch.testing.assert_close(
            original_m5_branch(inputs), expected_m5, rtol=0, atol=0
        )


def main():
    common = {
        "dim": 32,
        "depth": 2,
        "kernel_size": 5,
        "patch_size": 2,
        "image_size": 32,
        "n_classes": 10,
        "in_channels": 3,
        "local_init": 1.0,
        "rowcol_init": 0.1,
    }
    conditions = {
        "A_original_m3": WeightedRowColConvMixer(
            **common, rowcol_combine="learnable"
        ),
        "B_m3_prenorm": WeightedRowColConvMixer(
            **common, rowcol_combine="learnable", rowcol_pre_norm=True
        ),
        "C_add_control": WeightedRowColConvMixer(
            **common, rowcol_combine="sum", rowcol_pre_norm=True
        ),
        "D_mul_control": WeightedMultiplicativeRowColConvMixer(
            **common, mul_pre_norm=True, mul_pos_prior=False
        ),
        "E_original_m5": WeightedMultiplicativeRowColConvMixer(**common),
    }

    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 32, 32)
    for name, model in conditions.items():
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        assert output.shape == (2, 10), (name, output.shape)
        print(f"{name}: output={tuple(output.shape)}, params={trainable_parameter_count(model):,}")

    original_m3_branch = conditions["A_original_m3"].blocks[0].rowcol_branch
    original_m5_branch = conditions["E_original_m5"].blocks[0].rowcol_branch
    assert not hasattr(original_m3_branch, "h_norm")
    assert hasattr(original_m3_branch, "branch_weight")
    assert hasattr(original_m5_branch, "h_norm")
    assert hasattr(original_m5_branch, "p_h")
    assert hasattr(original_m5_branch, "p_w")
    assert_legacy_default_equations(original_m3_branch, original_m5_branch)

    add_control = conditions["C_add_control"]
    mul_control = conditions["D_mul_control"]
    assert list(add_control.named_parameters())
    add_parameters = [
        (name, tuple(parameter.shape))
        for name, parameter in add_control.named_parameters()
    ]
    mul_parameters = [
        (name, tuple(parameter.shape))
        for name, parameter in mul_control.named_parameters()
    ]
    assert add_parameters == mul_parameters
    assert trainable_parameter_count(add_control) == trainable_parameter_count(mul_control)
    add_state = [
        (name, tuple(value.shape)) for name, value in add_control.state_dict().items()
    ]
    mul_state = [
        (name, tuple(value.shape)) for name, value in mul_control.state_dict().items()
    ]
    assert add_state == mul_state

    controlled_names = rowcol_parameter_names(add_control)
    assert any(".h_norm." in name for name in controlled_names)
    assert any(".w_norm." in name for name in controlled_names)
    assert not any("branch_weight" in name for name in controlled_names)
    assert not any(".p_h" in name or ".p_w" in name for name in controlled_names)

    mul_control_branch = mul_control.blocks[0].rowcol_branch
    assert not hasattr(mul_control_branch, "p_h")
    assert not hasattr(mul_control_branch, "p_w")
    assert mul_control_branch.get_positional_prior_stats()["positional_prior"] is False

    assert_forward_order(conditions["B_m3_prenorm"].blocks[0].rowcol_branch)
    assert_forward_order(mul_control_branch)
    print("All controlled-ablation smoke checks passed.")


if __name__ == "__main__":
    main()
