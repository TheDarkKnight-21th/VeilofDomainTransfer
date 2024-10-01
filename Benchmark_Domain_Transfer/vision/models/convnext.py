from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ConvNeXt",
    "ConvNeXt_Tiny_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]




def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
    "_docs": """
        These weights improve upon the results of the original paper by using a modified version of TorchVision's
        `new training recipe
        <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    """,
}


class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=236),
        meta={
            **_COMMON_META,
            "num_params": 28589128,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.520,
                    "acc@5": 96.146,
                }
            },
            "_ops": 4.456,
            "_file_size": 109.119,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_small-0c510722.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=230),
        meta={
            **_COMMON_META,
            "num_params": 50223688,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.616,
                    "acc@5": 96.650,
                }
            },
            "_ops": 8.684,
            "_file_size": 191.703,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88591464,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
            "_ops": 15.355,
            "_file_size": 338.064,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 197767336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.414,
                    "acc@5": 96.976,
                }
            },
            "_ops": 34.361,
            "_file_size": 754.537,
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Small_Weights.IMAGENET1K_V1))
def convnext_small(
    *, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
        :members:
    """
    weights = ConvNeXt_Small_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Base_Weights.IMAGENET1K_V1))
def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
        :members:
    """
    weights = ConvNeXt_Base_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", ConvNeXt_Large_Weights.IMAGENET1K_V1))
def convnext_large(
    *, weights: Optional[ConvNeXt_Large_Weights] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt Large model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Large_Weights
        :members:
    """
    weights = ConvNeXt_Large_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
