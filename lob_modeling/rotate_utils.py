import pickle
import random

import numpy as np
import torch
import torchnet as tnt


def load_pickle_data(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
        if isinstance(data, dict):
            #data = {k.decode("ascii"): v for k, v in data.items()}
            data = {k: v for k, v in data.items()}

    return data


def build_label_index(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class FastConfusionMeter:
    def __init__(self, k, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def get_conf_matrix_results(matrix):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]

    count_correct = np.diag(matrix)
    count_preds = matrix.sum(1)
    count_gts = matrix.sum(0)
    epsilon = np.finfo(np.float32).eps
    accuracies = count_correct / (count_gts + epsilon)
    IoUs = count_correct / (count_gts + count_preds - count_correct + epsilon)
    totAccuracy = count_correct.sum() / (matrix.sum() + epsilon)

    num_valid = (count_gts > 0).sum()
    meanAccuracy = accuracies.sum() / (num_valid + epsilon)
    meanIoU = IoUs.sum() / (num_valid + epsilon)

    result = {
        "totAccuracy": round(totAccuracy, 4),
        "meanAccuracy": round(meanAccuracy, 4),
        "meanIoU": round(meanIoU, 4),
    }
    if num_valid == 2:
        result["IoUs_bg"] = round(IoUs[0], 4)
        result["IoUs_fg"] = round(IoUs[1], 4)

    return result


class BaseMeter:
    def __init__(self, name=None, _run=None):
        """ If singleton is True, work with single values."""
        self.val = None  # saves the last value added
        self.sum = None
        self.count = 0
        self.name = name
        self._run = _run

    def reset(self):
        self.val = None
        self.sum = None
        self.count = 0

    def update(self, array):
        array = np.array(array)
        self.val = array.astype(np.float64)
        if self.count == 0:
            self.sum = np.copy(self.val)
        else:
            self.sum += self.val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count

    def log(self):
        if self.name is None:
            return
        # we can't log metrics in 2D.
        if self.sum.ndim > 1:
            return

        if self._run is None:
            return

        average = self.sum / self.count
        if average.ndim == 0:
            self._run.log_scalar(self.name, average)
        else:
            for i, scalar in enumerate(average):
                self._run.log_scalar(f"{self.name}_{i}", scalar)

    def __str__(self):
        average = self.sum / self.count
        if average.ndim == 0:
            return f"{average:.4f}"
        elif average.ndim == 1:
            return " ".join(f"{scalar:.4f}" for scalar in average)
        elif average.ndim > 1:
            raise TypeError("cannot print easily a numpy array" "with a dimension above 1")


class AverageConfMeter(BaseMeter):
    @BaseMeter.avg.getter
    def avg(self):
        return get_conf_matrix_results(self.sum)


class DAverageMeter:
    def __init__(self, name=None, _run=None):
        self.values = {}
        self.name = name
        self._run = _run

    def reset(self):
        self.values = {}

    def update(self, values):
        assert isinstance(values, dict)
        for key, val in values.items():
            if isinstance(val, (float, int, list)):
                if not (key in self.values):
                    self.values[key] = BaseMeter(self.make_name(key), self._run)
                self.values[key].update(val)
            elif isinstance(val, (tnt.meter.ConfusionMeter, FastConfusionMeter)):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter(self.make_name(key), self._run)
                self.values[key].update(val.value())
            elif isinstance(val, AverageConfMeter):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter(self.make_name(key), self._run)
                self.values[key].update(val.sum)
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter(self.make_name(key), self._run)
                self.values[key].update(val)
            else:
                raise TypeError(f"Wrong type {type(val)}")

    def average(self):
        return {k: _get_average(v) for k, v in self.values.items()}

    def __str__(self):
        return str({k: str(v) for k, v in self.values.items()})

    def make_name(self, key):
        return f"{self.name}_{key}"

    def log(self):
        for value in self.values.values():
            value.log()


def _get_average(value):
    if isinstance(value, DAverageMeter):
        return value.average()
    return value.avg


def apply_2d_rotation(input_tensor, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.

    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor.dim() >= 2

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )


def top1accuracy(output, target):
    return top1accuracy_tensor(output, target).item()


def top1accuracy_tensor(output, target):
    pred = output.max(dim=1)[1]
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


def convert_from_5d_to_4d(tensor_5d):
    _, _, channels, height, width = tensor_5d.size()
    return tensor_5d.view(-1, channels, height, width)


def convert_from_6d_to_5d(tensor_6d):
    d0, d1, d2, d3, d4, d5 = tensor_6d.size()
    return tensor_6d.view(d0 * d1, d2, d3, d4, d5)


def convert_from_6d_to_4d(tensor_6d):
    return convert_from_5d_to_4d(convert_from_6d_to_5d(tensor_6d))


def add_dimension(tensor, dim_size):
    assert (tensor.size(0) % dim_size) == 0
    return tensor.view([dim_size, tensor.size(0) // dim_size,] + list(tensor.size()[1:]))


def crop_patch(image, y, x, patch_height, patch_width):
    _, image_height, image_width = image.size()
    y_top, y_bottom = y, y + patch_height
    x_left, x_right = x, x + patch_width

    assert y_top >= 0 and y_bottom <= image_height
    assert x_left >= 0 and x_right <= image_width

    patch = image[:, y_top:y_bottom, x_left:x_right].contiguous()

    return patch


def standardize_image(images):
    num_dims = images.dim()

    if not (num_dims == 4 or num_dims == 3):
        raise ValueError("The input tensor must have 3 or 4 dimnsions.")

    if num_dims == 3:
        images = images.unsqueeze(dim=0)

    batch_size, channels, height, width = images.size()
    images_flat = images.view(batch_size, -1)
    mean_values = images_flat.mean(dim=1, keepdim=True)
    std_values = images_flat.std(dim=1, keepdim=True) + 1e-5
    images_flat = (images_flat - mean_values) / std_values

    images = images_flat.view(batch_size, channels, height, width)

    if num_dims == 3:
        assert images.size(0) == 1
        images = images.squeeze(dim=0)

    return images


def image_to_patches(image, is_training, split_per_side, patch_jitter=0):
    """Crops split_per_side x split_per_side patches from input image.

    Args:
        image: input image tensor with shape [c, h, w].
        is_training: is training flag.
        split_per_side: split of patches per image side.
        patch_jitter: jitter of each patch from each grid.
    Returns:
        Patches: 4D tensor with shape
        [num_patches, num_channels, patch_height, patch_width], where
        num_patches = split_per_side * split_per_side
    """
    _, image_height, image_width = image.size()
    assert patch_jitter >= 0
    grid_height = image_height // split_per_side
    grid_width = image_width // split_per_side
    patch_height = grid_height - patch_jitter
    patch_width = grid_width - patch_jitter

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            y, x = i * grid_height, j * grid_width

            if patch_jitter > 0:
                if is_training:
                    dy = random.randint(0, patch_jitter)
                    dx = random.randint(0, patch_jitter)
                    y += dy
                    x += dx
                else:
                    y += patch_jitter // 2
                    x += patch_jitter // 2

            patches.append(crop_patch(image, y, x, patch_height, patch_width))

    return torch.stack(patches, dim=0)



import numpy as np
import torch
import torch.nn.functional as F



def rotation_task(rotation_classifier, features, labels_rotation):
    """Applies the rotation prediction head to the given features."""
    scores = rotation_classifier(features)
    assert scores.size(1) == 4
    loss = F.cross_entropy(scores, labels_rotation)

    return scores, loss


def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot


def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot

