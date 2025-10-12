from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import einops
from torch import Tensor

class DDPWrapper:
    def __init__(self, ddp_module: Any) -> None:
        self.ddp_module: Any = ddp_module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.ddp_module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.ddp_module, name):
            attr: Any = getattr(self.ddp_module, name)

            #  Make sure to return DDPWrapper instead of directly returning the attribute in case of a callable such as state.model.train()
            if callable(attr):

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    result: Any = attr(*args, **kwargs)
                    if result is self.ddp_module:
                        return self
                    return result

                return wrapper
            return attr

        if hasattr(self.ddp_module.module, name):
            return getattr(self.ddp_module.module, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

@dataclass
class DecoderOutput:
    """
    Container for decoder outputs (logits, masks, and auxiliary tensors).
    Fields default to None; set by the decoder as available.
    """
    low_res_masks: Optional[Tensor] = None
    high_res_masks: Optional[Tensor] = None
    obj_ptr: Optional[Tensor] = None
    pix_feat_with_mem: Optional[Tensor] = None
    masks: Optional[Tensor] = None

    def __post_init__(self):
        self.masks = self.low_res_masks

    def move_to_cpu(self) -> "DecoderOutput":
        """Safely move all present tensors to CPU and return self."""
        for field in (
            "low_res_masks",
            "high_res_masks",
            "obj_ptr",
            "object_score_logits",
            "hyper_in",
            "object_score",
            "masks",
            "pix_feat_with_mem",
        ):
            val = getattr(self, field)
            if val is not None:
                setattr(self, field, val.cpu())
        return self


@dataclass
class BackboneOutput:
    """
    Container for backbone features across a clip.

    Attributes:
        orig_size: List of original sizes [(H, W)] for each frame (length B*T).
        vision_feats: List of feature tensors per FPN level.
        vision_pos_embeds: List of positional encodings per level.
        feat_sizes: List of (H, W) per FPN level.
    """
    orig_size: List[Tuple[int, int]]
    vision_feats: List[Tensor]
    vision_pos_embeds: List[Tensor]
    feat_sizes: List[Tuple[int, int]]

    def get_current_feats(self, idx: int) -> List[Tensor]:
        """
        Slice features for the flat frame index `idx` (0 <= idx < B*T).

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        return [x[:, idx:idx + 1, :] for x in self.vision_feats]

    def get_current_pos_embeds(self, idx: int) -> List[Tensor]:
        """
        Slice positional encodings for the flat frame index `idx`.

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        return [x[:, idx:idx + 1, :] for x in self.vision_pos_embeds]

    def get_current_feats_x16(self, idx: int) -> Tensor:
        """
        Slice features for the flat frame index `idx` (0 <= idx < B*T).

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        vision_feats_16 = self.vision_feats[-1][:, idx:idx + 1, :]
        vision_feats_16 = einops.rearrange(vision_feats_16, '(h w) b c -> b c h w', h=self.feat_sizes[-1][0])
        return vision_feats_16


    def get_high_res_features(self, current_vision_feats: List[Tensor]) -> List[Tensor]:
        """
        Reformat high-resolution backbone features for the decoder.
        Operates on all levels except the last.

        Args:
            current_vision_feats: list of per-level tensors for the current frame.

        Returns:
            List of tensors shaped [B(=1), C, H, W] for the high-res levels.
        """
        high_res = []
        for x, s in zip(current_vision_feats[:-1], self.feat_sizes[:-1]):
            # x is typically [C, 1, HW] -> permute to [1, HW, C], then view to [1, C, H, W]
            x_perm = x.permute(1, 2, 0).contiguous()            # [1, HW, C]
            x_bchw = x_perm.view(x.size(1), x.size(2), *s)      # [1, C, H, W]
            high_res.append(x_bchw)
        return high_res

    def move_to_cpu(self) -> "BackboneOutput":
        """Move feature lists to CPU."""
        self.vision_feats = [x.cpu() for x in self.vision_feats]
        self.vision_pos_embeds = [x.cpu() for x in self.vision_pos_embeds]
        return self
