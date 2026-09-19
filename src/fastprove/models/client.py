"""Client/debug-side holder of the hidden mixing transform."""

from __future__ import annotations

from typing import Tuple

import torch

from ..state import MixedState, decode_debug, encode_debug
from ..structured import StructuredBasis
from ..transforms import BasisDescriptor


class ObfuscatedBlockClient:
    """Client/debug-side holder of the hidden mixing transform."""

    def __init__(
        self, transform: StructuredBasis, *, debug_enabled: bool
    ) -> None:
        self._transform = transform
        self._debug_enabled = bool(debug_enabled)

    @property
    def basis(self) -> BasisDescriptor:
        """Public basis identity without matrix material."""

        return self._transform.descriptor

    def encode_debug(
        self, signal: torch.Tensor, noise: torch.Tensor
    ) -> MixedState:
        """Encode an input only on the explicitly enabled reference path."""

        return encode_debug(
            signal, noise, self._transform, enabled=self._debug_enabled
        )

    def decode_debug(
        self, state: MixedState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode an output only on the explicitly enabled reference path."""

        return decode_debug(
            state, self._transform, enabled=self._debug_enabled
        )
