from __future__ import annotations

import os
from threading import Lock

import cv2
import numpy as np
import torch
import vapoursynth as vs
from torchvision.transforms.functional import normalize

from .dpt_depth import DPTDepthModel

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def midas(
    clip: vs.VideoNode, device_index: int | None = None, num_streams: int = 1, model: int = 2, grayscale: bool = False
) -> vs.VideoNode:
    """Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer

    :param clip:            Clip to process. Only RGBH and RGBS formats are supported.
                            RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:    Device ordinal of the GPU.
    :param num_streams:     Number of CUDA streams to enqueue the kernels.
    :param model:           Model to use.
                            0 = dpt_swin2_tiny_256
                            1 = dpt_large_384
                            2 = dpt_swin2_large_384
                            3 = dpt_beit_large_512
    :param grayscale:       Use a grayscale colormap instead of the inferno one.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("midas: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("midas: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("midas: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("midas: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("midas: setting num_streams greater than `core.num_threads` is useless")

    if model not in range(4):
        raise vs.Error("midas: model must be 0, 1, 2, or 3")

    if os.path.getsize(os.path.join(model_dir, "dpt_beit_large_512.pt")) == 0:
        raise vs.Error("midas: model files have not been downloaded. run 'python -m vsmidas' first")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case 0:
            model_name = "dpt_swin2_tiny_256.pt"
            module = DPTDepthModel(backbone="swin2t16_256")
            net_w, net_h = 256, 256
        case 1:
            model_name = "dpt_large_384.pt"
            module = DPTDepthModel(backbone="vitl16_384")
            net_w, net_h = 384, 384
        case 2:
            model_name = "dpt_swin2_large_384.pt"
            module = DPTDepthModel(backbone="swin2l24_384")
            net_w, net_h = 384, 384
        case 3:
            model_name = "dpt_beit_large_512.pt"
            module = DPTDepthModel(backbone="beitl16_512")
            net_w, net_h = 512, 512

    parameters = torch.load(os.path.join(model_dir, model_name), map_location="cpu")
    if "optimizer" in parameters:
        parameters = parameters["model"]

    module.load_state_dict(parameters)
    module.eval().to(device, memory_format=torch.channels_last)
    if clip.format.bits_per_sample == 16:
        module.half()

    if grayscale:
        new_format = clip.format.replace(color_family=vs.GRAY)
    else:
        new_format = clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=8)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device)
            normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

            output = module(img).unsqueeze(0)

            if not output.isfinite().all():
                output.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                vs.core.log_message(
                    vs.MESSAGE_TYPE_WARNING, "midas: non-finite values present. inference in FP32 mode is recommended"
                )

            output_min = output.min()
            output_max = output.max()

            if output_max - output_min > torch.finfo(output.dtype).eps:
                output = (output - output_min) / (output_max - output_min)
            else:
                output.zero_()

            if not grayscale:
                output = (output * 255).type(torch.uint8)

            return tensor_to_frame(output, grayscale, f[1].copy())

    resized_clip = clip.resize.Bicubic(net_w, net_h)
    new_clip = clip.std.BlankClip(net_w, net_h, format=new_format, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([resized_clip, new_clip], inference), clip_src=[resized_clip, new_clip]
    ).resize.Bicubic(clip.width, clip.height)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, grayscale: bool, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()

    if not grayscale:
        array = np.transpose(array, (1, 2, 0))
        array = cv2.applyColorMap(array, cv2.COLORMAP_INFERNO)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        array = np.transpose(array, (2, 0, 1))

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])

    return frame
