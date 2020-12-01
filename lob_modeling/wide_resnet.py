import math
import torch.nn as nn

class SequentialFeatureExtractorAbstractClass(nn.Module):
    def __init__(self, all_feat_names, feature_blocks):
        super().__init__()

        assert isinstance(feature_blocks, list)
        assert isinstance(all_feat_names, list)
        assert len(all_feat_names) == len(feature_blocks)

        self.all_feat_names = all_feat_names
        self._feature_blocks = nn.ModuleList(feature_blocks)

    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError("Empty list of output feature keys.")

        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    "Feature with name {} does not exist. "
                    "Existing features: {}.".format(key, self.all_feat_names)
                )
            elif key in out_feat_keys[:f]:
                raise ValueError(f"Duplicate output feature key: {key}.")

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward the image `x` through the network and output the asked features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. If out_feat_keys is None (
                DEFAULT) then the last feature of the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats

        return out_feats

#utils
def global_pooling(x, pool_type):
    assert x.dim() == 4
    if pool_type == "max":
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif pool_type == "avg":
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError("Unknown pooling type.")

class GlobalPooling(nn.Module):
    def __init__(self, pool_type):
        super().__init__()
        assert pool_type == "avg" or pool_type == "max"
        self.pool_type = pool_type

    def forward(self, x):
        return global_pooling(x, pool_type=self.pool_type)




class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, kernel_size=3):
        super().__init__()

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 2

        kernel_size1, kernel_size2 = kernel_size

        assert kernel_size1 == 1 or kernel_size1 == 3
        padding1 = 1 if kernel_size1 == 3 else 0
        assert kernel_size2 == 1 or kernel_size2 == 3
        padding2 = 1 if kernel_size2 == 3 else 0

        self.equalInOut = in_planes == out_planes and stride == 1

        self.convResidual = nn.Sequential()

        if self.equalInOut:
            self.convResidual.add_module("bn1", nn.BatchNorm2d(in_planes))
            self.convResidual.add_module("relu1", nn.ReLU(inplace=True))

        self.convResidual.add_module(
            "conv1",
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size1,
                stride=stride,
                padding=padding1,
                bias=False,
            ),
        )

        self.convResidual.add_module("bn2", nn.BatchNorm2d(out_planes))
        self.convResidual.add_module("relu2", nn.ReLU(inplace=True))
        self.convResidual.add_module(
            "conv2",
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size=kernel_size2,
                stride=1,
                padding=padding2,
                bias=False,
            ),
        )

        if drop_rate > 0:
            self.convResidual.add_module("dropout", nn.Dropout(p=drop_rate))

        if self.equalInOut:
            self.convShortcut = nn.Sequential()
        else:
            self.convShortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        return self.convShortcut(x) + self.convResidual(x)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()

        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):

        layers = []
        for i in range(nb_layers):
            in_planes_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(block(in_planes_arg, out_planes, stride_arg, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResnet(SequentialFeatureExtractorAbstractClass):
    def __init__(
        self,
        depth,
        widen_factor=1,
        drop_rate=0.0,
        pool="avg",
        extra_block=False,
        block_strides=[2, 2, 2, 2],
        extra_block_width_mult=1,
        num_layers=None,
    ):
        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
            64 * widen_factor * extra_block_width_mult,
        ]

        assert not ((depth is None) and (num_layers is None))

        num_blocks = 4 if extra_block else 3
        if depth is not None:
            assert (depth - 4) % 6 == 0
            n = int((depth - 4) / 6)
            num_layers = [n for _ in range(num_blocks)]
        else:
            assert isinstance(num_layers, (list, tuple))
        assert len(num_layers) == num_blocks

        block = BasicBlock

        all_feat_names = []
        feature_blocks = []

        # 1st conv before any network block
        conv1 = nn.Sequential()
        conv1.add_module(
            "Conv", nn.Conv2d(3, nChannels[0], kernel_size=3, padding=1, bias=False)
        )
        conv1.add_module("BN", nn.BatchNorm2d(nChannels[0]))
        conv1.add_module("ReLU", nn.ReLU(inplace=True))
        feature_blocks.append(conv1)
        all_feat_names.append("conv1")

        # 1st block.
        block1 = nn.Sequential()
        block1.add_module(
            "Block",
            NetworkBlock(
                num_layers[0], nChannels[0], nChannels[1], block, block_strides[0], drop_rate
            ),
        )
        block1.add_module("BN", nn.BatchNorm2d(nChannels[1]))
        block1.add_module("ReLU", nn.ReLU(inplace=True))
        feature_blocks.append(block1)
        all_feat_names.append("block1")

        # 2nd block.
        block2 = nn.Sequential()
        block2.add_module(
            "Block",
            NetworkBlock(
                num_layers[1], nChannels[1], nChannels[2], block, block_strides[1], drop_rate
            ),
        )
        block2.add_module("BN", nn.BatchNorm2d(nChannels[2]))
        block2.add_module("ReLU", nn.ReLU(inplace=True))
        feature_blocks.append(block2)
        all_feat_names.append("block2")

        # 3rd block.
        block3 = nn.Sequential()
        block3.add_module(
            "Block",
            NetworkBlock(
                num_layers[2], nChannels[2], nChannels[3], block, block_strides[2], drop_rate
            ),
        )
        block3.add_module("BN", nn.BatchNorm2d(nChannels[3]))
        block3.add_module("ReLU", nn.ReLU(inplace=True))
        feature_blocks.append(block3)
        all_feat_names.append("block3")

        # extra block.
        if extra_block:
            block4 = nn.Sequential()
            block4.add_module(
                "Block",
                NetworkBlock(
                    num_layers[3],
                    nChannels[3],
                    nChannels[4],
                    block,
                    block_strides[3],
                    drop_rate,
                ),
            )
            block4.add_module("BN", nn.BatchNorm2d(nChannels[4]))
            block4.add_module("ReLU", nn.ReLU(inplace=True))
            feature_blocks.append(block4)
            all_feat_names.append("block4")

        # global average pooling and classifier_type
        assert pool == "none" or pool == "avg" or pool == "max"
        if pool == "max" or pool == "avg":
            feature_blocks.append(GlobalPooling(pool_type=pool))
            all_feat_names.append("GlobalPooling")

        super().__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def create_model(opt):
    depth = opt["depth"]
    widen_factor = opt["widen_Factor"]
    drop_rate = opt["drop_rate"] if ("drop_rate" in opt) else 0.0
    pool = opt["pool"] if ("pool" in opt) else "avg"
    extra_block = opt["extra_block"] if ("extra_block" in opt) else False
    block_strides = opt["strides"] if ("strides" in opt) else None
    extra_block_width_mult = (
        opt["extra_block_width_mult"] if ("extra_block_width_mult" in opt) else 1
    )

    if block_strides is None:
        block_strides = [2] * 4

    return WideResnet(
        depth=depth,
        widen_factor=widen_factor,
        drop_rate=drop_rate,
        pool=pool,
        extra_block=extra_block,
        block_strides=block_strides,
        extra_block_width_mult=extra_block_width_mult,
        num_layers=None,
    )
