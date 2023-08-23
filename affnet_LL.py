# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
'''
Description:
Author：LL-Version-V1
LastEditTime: 2023-08-23
Description:AFFNet-Pytorch-Unofficial-Implementation
Reference：https://github.com/microsoft/TokenMixers；
Original Paper：Adaptive Frequency Filters As Efficient Global Token Mixers, ICCV 2023-https://arxiv.org/abs/2307.14008
'''
# --------------------------------------------------------
import torch
from torch import nn
import argparse
from typing import Dict, Tuple, Optional

import logger
from base_cls_LL import BaseEncoder
from affnet_config import get_configuration
from layers import ConvLayer, LinearLayer, GlobalPool, Dropout, InvertedResidual
from aff_block_LL import AFFBlock


class AffNet(BaseEncoder):

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = 1000
        classifier_dropout = 0.0

        pool_type = "mean"
        image_channels = 3
        out_channels = 16

        affnet_config = get_configuration(opts=opts)

        super().__init__(opts, *args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=affnet_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=affnet_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=affnet_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=affnet_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=affnet_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        exp_channels = min(affnet_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            use_norm=True,
        )

        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": exp_channels,
        }

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="dropout", module=Dropout(p=classifier_dropout, inplace=True)
            )
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(
                in_features=exp_channels, out_features=num_classes, bias=True
            ),
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def _make_layer(
        self,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "aff_block")
        if block_type.lower() == "aff_block":
            return self._make_affnet_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        input_channel: int, cfg: Dict, *args, **kwargs
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                inp=input_channel,
                oup=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_affnet_layer(
        self,
        input_channel,
        cfg: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)
        no_fuse = cfg.get("no_fuse", False)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                inp=input_channel,
                oup=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                # dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error(
                "Transformer input dimension should be divisible by head dimension. "
                "Got {} and {}.".format(transformer_dim, head_dim)
            )

        block.append(
            AFFBlock(
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=0.1,
                ffn_dropout=0.0,
                attn_dropout=0.1,
                head_dim=head_dim,
                no_fusion=no_fuse,
                conv_ksize=3,
                attn_norm_layer="layer_norm_2d",
            )
        )

        return nn.Sequential(*block), input_channel

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model.classification.affnet.mode",
        type=str,
        default="xx_small",
        choices=["xx_small", "x_small", "small"],
    )
    parser.add_argument(
        "--model.classification.affnet.attn-dropout",
        type=float,
        default=0.0,
        help="Dropout in attention layer. Defaults to 0.0",
    )
    parser.add_argument(
        "--model.classification.affnet.ffn-dropout",
        type=float,
        default=0.0,
        help="Dropout between FFN layers. Defaults to 0.0",
    )
    parser.add_argument(
        "--model.classification.affnet.dropout",
        type=float,
        default=0.0,
        help="Dropout in Transformer layer. Defaults to 0.0",
    )
    parser.add_argument(
        "--model.classification.affnet.attn-norm-layer",
        type=str,
        default="layer_norm",
        help="Normalization layer in transformer. Defaults to LayerNorm",
    )
    parser.add_argument(
        "--model.classification.affnet.no-fuse-local-global-features",
        action="store_true",
        help="Do not combine local and global features in MobileViT block",
    )
    parser.add_argument(
        "--model.classification.affnet.conv-kernel-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--model.classification.affnet.head-dim",
        type=int,
        default=None,
        help="Head dimension in transformer",
    )
    parser.add_argument(
        "--model.classification.affnet.number-heads",
        type=int,
        default=None,
        help="Number of heads in transformer",
    )
    args = parser.parse_args()

    # opts = get_configuration(opts=args)
    affnet = AffNet(args)
    torch.save(affnet.state_dict(), "affnet.ckpt")

    input = torch.randn(1, 3, 640, 640)

    output = affnet(input)

    print(f"output.shape：{output.shape}")