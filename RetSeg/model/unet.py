"""
MONAI-style UNet with Dynamic Active Depth

Remove learnable `up_bypass` and instead ALWAYS use `up_full`.
When bypassing deeper recursion, we still build the skip concat as cat(down(x), down(x)) = 2c,
but we adapt channels (2c -> upc_full) using a *frozen* 1x1 conv adapter, then feed into `up_full`.

Benefits:
- One shared decoder path (`up_full`) across all clients (shallow and deep).
- Shallow clients still contribute gradients to the shared `up_full` (through bypass),
  reducing "two decoder branches" mismatch in FL aggregation.
- Adapter is frozen by default so it won't become a separate learnable pathway.

Notes:
- For most levels, upc_full == 2c, so the adapter becomes Identity and adds no compute.
- Only at the penultimate level (where bottom outputs channels[1]) you have upc_full = c + channels[1],
  so the adapter is a real 1x1 conv (frozen).
"""

from __future__ import annotations

import warnings
from typing import Sequence, Union, Optional

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm


# ----------------------------
# Utility adapters
# ----------------------------

class _BottomAdapter(nn.Module):
    """Adapts a plain nn.Module(bottom layer) to accept (x, active_layers)."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, active_layers: int) -> torch.Tensor:
        return self.module(x)


# ----------------------------
# Core conditional block (Option 1)
# ----------------------------

class ConditionalUNetBlock(nn.Module):
    """
    One UNet "level" that supports dynamic depth activation.

    Option 1 behavior:
    - There is only ONE up module: up_full
    - On bypass path, we replace subblock output with x_down (so concat becomes 2c),
      then run a frozen adapter (2c -> upc_full) if needed, then run up_full.
    """
    def __init__(
        self,
        down: nn.Module,
        subblock: nn.Module,
        up_full: nn.Module,
        bypass_adapter: nn.Module,
        level_idx: int,
    ):
        super().__init__()
        self.down = down
        self.subblock = subblock
        self.up_full = up_full
        self.bypass_adapter = bypass_adapter
        self.level_idx = level_idx

    def forward(self, x: torch.Tensor, active_layers: int) -> torch.Tensor:
        x_down = self.down(x)

        # Bypass deeper recursion: replace subblock output with x_down
        if self.level_idx >= active_layers:
            x_cat = torch.cat([x_down, x_down], dim=1)   # (N, 2c, ...)
            x_cat = self.bypass_adapter(x_cat)           # (N, upc_full, ...)
            return self.up_full(x_cat)

        # Normal recursion
        x_sub = self.subblock(x_down, active_layers)
        x_cat = torch.cat([x_down, x_sub], dim=1)        # (N, upc_full, ...)
        return self.up_full(x_cat)


# ----------------------------
# UNet with dynamic active depth (Option 1)
# ----------------------------

class UNet(nn.Module):
    """
    MONAI-style Residual UNet with dynamic "active_layers" that bypasses deeper recursion.

    Option 1:
    - Always use `up_full` for decoding.
    - Use a frozen 1x1 adapter on bypass path when (2c != upc_full).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[tuple, str] = Act.PRELU,
        norm: Union[tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        active_layers: Optional[int] = None,
        freeze_bypass_adapter: bool = True,  # <-- Option 1 default
    ) -> None:
        super().__init__()

        # --- Validation ---
        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")

        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("If sequence, the length of `kernel_size` should equal `spatial_dims`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("If sequence, the length of `up_kernel_size` should equal `spatial_dims`.")

        # Store config
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.strides = tuple(strides[: len(channels) - 1])  # ignore extra
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.freeze_bypass_adapter = freeze_bypass_adapter

        # Active depth default: full depth
        if active_layers is None:
            active_layers = len(self.channels)
        if not (1 <= active_layers <= len(self.channels)):
            raise ValueError(f"`active_layers` must be in [1, {len(self.channels)}], got {active_layers}.")
        self.active_layers = int(active_layers)

        # Build full model
        self.model = self._create_block(
            inc=self.in_channels,
            outc=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            is_top=True,
            level_idx=1,
        )

    # ----------------------------
    # Layers
    # ----------------------------

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )

        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, strides=1, is_top=False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: nn.Module = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def _make_1x1_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        if in_ch == out_ch:
            return nn.Identity()

        if self.dimensions == 1:
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        elif self.dimensions == 2:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        elif self.dimensions == 3:
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            raise ValueError(f"Unsupported spatial_dims={self.dimensions}")

        if self.freeze_bypass_adapter:
            for p in conv.parameters():
                p.requires_grad = False
        return conv

    # ----------------------------
    # Recursive builder (Option 1)
    # ----------------------------

    def _create_block(
        self,
        inc: int,
        outc: int,
        channels: Sequence[int],
        strides: Sequence[int],
        is_top: bool,
        level_idx: int,
    ) -> nn.Module:
        """
        Build from top to bottom recursively, returning a module that accepts (x, active_layers).
        """
        c = channels[0]
        s = strides[0]  # stride for this level down/up

        down = self._get_down_layer(inc, c, s, is_top=is_top)

        # Build subblock or bottom
        if len(channels) > 2:
            subblock = self._create_block(
                inc=c,
                outc=c,
                channels=channels[1:],
                strides=strides[1:],
                is_top=False,
                level_idx=level_idx + 1,
            )
            # Full path concat channels: cat(c, c) => 2c
            upc_full = 2 * c
        else:
            # bottom maps c -> channels[1]
            bottom = self._get_bottom_layer(c, channels[1])
            subblock = _BottomAdapter(bottom)

            # Full path concat channels: cat(down_out(c), bottom_out(channels[1])) => c + channels[1]
            upc_full = c + channels[1]

        # Single decoder module
        up_full = self._get_up_layer(upc_full, outc, s, is_top=is_top)

        # Bypass concat is always 2c; adapt 2c -> upc_full if needed
        bypass_adapter = self._make_1x1_conv(in_ch=2 * c, out_ch=upc_full)

        return ConditionalUNetBlock(
            down=down,
            subblock=subblock,
            up_full=up_full,
            bypass_adapter=bypass_adapter,
            level_idx=level_idx,
        )

    # ----------------------------
    # FL utils
    # ----------------------------

    def get_learnable_parameters(self, active_layers: Optional[int] = None):
        """
        Return parameters that SHOULD be trained given active_layers, matching execution.

        Option 1:
        - down always runs
        - if bypass: bypass_adapter runs (but is typically frozen) + up_full runs
        - if full: subblock runs + up_full runs

        We exclude frozen bypass_adapter parameters automatically (requires_grad=False).
        """
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}")

        learnable = []

        def add_params(prefix: str, mod: nn.Module):
            for n, p in mod.named_parameters(recurse=True):
                if p.requires_grad:
                    learnable.append((f"{prefix}.{n}", p))

        def collect_from_module(mod: nn.Module, prefix: str):
            if isinstance(mod, ConditionalUNetBlock):
                add_params(f"{prefix}.down", mod.down)

                if mod.level_idx >= k:
                    # bypass: adapter (usually frozen) + up_full
                    add_params(f"{prefix}.bypass_adapter", mod.bypass_adapter)
                    add_params(f"{prefix}.up_full", mod.up_full)
                    return

                # full path: recurse + up_full
                collect_from_module(mod.subblock, f"{prefix}.subblock")
                add_params(f"{prefix}.up_full", mod.up_full)
                return

            if isinstance(mod, _BottomAdapter):
                add_params(f"{prefix}.module", mod.module)
                return

            add_params(prefix, mod)

        collect_from_module(self.model, "model")

        # De-duplicate by parameter identity
        seen = set()
        unique = []
        for name, p in learnable:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append((name, p))

        return unique

    def estimate_flops(
        self,
        input_shape,
        active_layers: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """
        Device-agnostic forward FLOPs estimate on CPU.
        Returns total forward FLOPs (int).
        """
        from torch.utils.flop_counter import FlopCounterMode

        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}")

        orig_device = next(self.parameters()).device
        self_cpu = self.to("cpu").eval()

        x = torch.randn(*input_shape, device="cpu", dtype=dtype)

        with FlopCounterMode(display=False) as fc:
            _ = self_cpu.forward(x, active_layers=k)

        if hasattr(fc, "get_total_flops"):
            total_flops = int(fc.get_total_flops())
        else:
            total_flops = int(getattr(fc, "total_flops", 0))

        self.to(orig_device)
        return total_flops

    # ----------------------------
    # Forward
    # ----------------------------

    def forward(self, x: torch.Tensor, active_layers: Optional[int] = None) -> torch.Tensor:
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}.")
        return self.model(x, k)


class UNetMultiExits(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[tuple, str] = Act.PRELU,
        norm: Union[tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        freeze_bypass_adapter: bool = True,  # <-- Option 1 default
        active_layers: int = 5,
    ) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
            active_layers = active_layers,
            freeze_bypass_adapter=freeze_bypass_adapter,

        )
        self.active_layers = active_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(self.active_layers):
            outs.append(self.model(x, i+1))
        return outs

# ----------------------------
# Quick sanity test
# ----------------------------
if __name__ == "__main__":
    

    x = torch.randn(10, 3, 256, 256)

    # multi-exit UNet
    net = UNetMultiExits(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            act='RELU')
    y = net(x)
    print(f"Multi-exit UNet output shapes:")
    for i, out in enumerate(y):
        print(f"  Exit {i+1}: shape={tuple(out.shape)}") 

    # single-exit UNet with varying active_layers
    for k in [1, 2, 3, 4, 5]:
        net = UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                active_layers=k,
                act='RELU')
        flops = net.estimate_flops(input_shape=x.shape, active_layers=k)
        print(f"active_layers={k}: FLOPs={flops/1e9:.3f} GFLOPs")

    for k in [1, 2, 3, 4, 5]:
        compute_capacities = [1, 2, 1, 4, 5]
        net = UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                active_layers=compute_capacities[k-1],
                act='RELU')
        learnable = net.get_learnable_parameters(active_layers=k)
        print("Learnable parameters:")
        for name, p in learnable:
            print(f"  {name}: {tuple(p.shape)}")
        print(f"active_layers={k}: learnable params returned = {len(learnable)}")