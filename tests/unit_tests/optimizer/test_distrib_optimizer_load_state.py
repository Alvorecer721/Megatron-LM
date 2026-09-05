# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from types import SimpleNamespace

import torch

from megatron.core import optimizer as optimizer_module
from megatron.core.optimizer import distrib_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.core.optimizer.optimizer_config import OptimizerConfig


class _InnerOptimizer:
    def __init__(self, param_group):
        self.state = {}
        self._state_dict = {"state": {}, "param_groups": [param_group]}
        self.loaded_state_dict = None

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


class _PrecisionAwareInnerOptimizer:
    def __init__(self, param, param_group):
        self.param = param
        self.param_groups = [{**param_group, "params": [param]}]
        self.state = {}
        self.load_state_dict_called = False

    def state_dict(self):
        return {
            "state": {0: self.state[self.param]},
            "param_groups": [{**self.param_groups[0], "params": [0]}],
        }

    def load_state_dict(self, _state_dict):
        self.load_state_dict_called = True
        raise AssertionError("precision-aware state went through the casting loader")

    def __setstate__(self, state):
        self.state = state["state"]
        self.param_groups = state["param_groups"]

    def get_unscaled_state(self, param, name):
        return self.state[param][name]

    def set_scaled_state(self, param, name, value):
        self.state.setdefault(param, {})[name] = value


class _PrecisionAwareAdamStub:
    """Model the two-argument state initializer exposed by TE 2.15."""

    def __init__(self, params, **_kwargs):
        self.param_groups = params
        self.state = defaultdict(dict)
        self.initialize_state_calls = []

    def initialize_state(self, param, store_param_remainders):
        self.initialize_state_calls.append((param, store_param_remainders))
        self.state[param]["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)


def test_load_state_dict_allocates_dummy_optimizer_state_on_cpu(monkeypatch):
    """Dummy state must not overlap with the final restored state in GPU memory."""
    monkeypatch.setattr(distrib_optimizer, "HAVE_APEX_OR_TE", True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("dummy state queried the CUDA device")),
    )

    model_param = object()
    identifier_fields = {key: None for key in param_group_identifier_keys}
    inner_param_group = {**identifier_fields, "params": [0]}
    optimizer = _InnerOptimizer(inner_param_group)

    distributed_optimizer = object.__new__(DistributedOptimizer)
    distributed_optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    distributed_optimizer.optimizer = optimizer
    distributed_optimizer.init_state_fn = None
    distributed_optimizer.gbuf_ranges = [
        {"buffer": [{"param_map": {model_param: {"gbuf_world": range(4)}}}]}
    ]
    distributed_optimizer.model_param_group_index_map = {model_param: (0, 0)}
    distributed_optimizer.config = SimpleNamespace(
        optimizer="adam",
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        fp16=False,
    )
    distributed_optimizer.grad_scaler = None
    checkpoint_state = {"optimizer": {"param_groups": [identifier_fields]}}

    distributed_optimizer.load_state_dict(checkpoint_state)

    loaded_state = optimizer.loaded_state_dict["state"][0]
    assert set(loaded_state) == {"exp_avg", "exp_avg_sq"}
    assert all(tensor.device.type == "cpu" for tensor in loaded_state.values())
    assert all(tensor.shape == (4,) for tensor in loaded_state.values())


def test_precision_aware_init_state_passes_remainder_setting(monkeypatch):
    """The initializer must honor the TE 2.15 two-argument API."""
    monkeypatch.setattr(optimizer_module, "USING_PYTORCH_OPTIMIZER", False)
    monkeypatch.setattr(optimizer_module, "Adam", _PrecisionAwareAdamStub)

    config = OptimizerConfig(optimizer="adam", lr=0.01)
    param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    optimizer, init_state_fn = optimizer_module._get_megatron_optimizer_based_on_param_groups(
        config, model_chunks=[], param_groups=[{"params": [param]}], skip_megatron_wrapping=True
    )

    config.use_precision_aware_optimizer = True
    config.store_param_remainders = True
    init_state_fn(optimizer, config)

    assert optimizer.initialize_state_calls == [(param, True)]


def test_load_state_dict_reuses_precision_aware_optimizer_state(monkeypatch):
    """DCP loading must not send precision-aware state through TE's casting loader."""
    monkeypatch.setattr(distrib_optimizer, "HAVE_APEX_OR_TE", True)

    param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    identifier_fields = {key: None for key in param_group_identifier_keys}
    optimizer = _PrecisionAwareInnerOptimizer(param, {**identifier_fields, "lr": 1.0})
    initialized_state = {}
    init_calls = 0

    def init_state_fn(inner_optimizer, _config):
        nonlocal init_calls
        init_calls += 1
        initialized_state.update(
            {
                "exp_avg": torch.zeros(4, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(4, dtype=torch.float32),
                "master_param": torch.zeros(4, dtype=torch.int16),
            }
        )
        inner_optimizer.state[param] = initialized_state

    distributed_optimizer = object.__new__(DistributedOptimizer)
    distributed_optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    distributed_optimizer.optimizer = optimizer
    distributed_optimizer.init_state_fn = init_state_fn
    distributed_optimizer.config = SimpleNamespace(
        optimizer="adam",
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
        fp16=False,
    )
    distributed_optimizer.grad_scaler = None
    checkpoint_state = {"optimizer": {"param_groups": [{**identifier_fields, "lr": 0.25}]}}

    distributed_optimizer.load_state_dict(checkpoint_state)
    second_checkpoint_state = {"optimizer": {"param_groups": [{**identifier_fields, "lr": 0.125}]}}
    distributed_optimizer.load_state_dict(second_checkpoint_state)

    assert optimizer.load_state_dict_called is False
    assert init_calls == 1
    assert optimizer.state[param] is initialized_state
    assert optimizer.param_groups[0]["params"] == [param]
    assert optimizer.param_groups[0]["lr"] == 0.125
