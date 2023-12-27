import argparse
import gc
import math
from typing import List

import deepspeed
import torch


class DSPipeline:
    """
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    """

    def __init__(
        self,
        model,
        dtype=torch.float16,
        device=-1,
    ):
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")
        self.device = torch.device("cpu")

        self.model = model

        self.model.eval()

    def __call__(self, inputs=["test"], **kwargs):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        outputs = self.generate_outputs(input_list, **kwargs)
        return outputs

    def generate_outputs(self, inputs=["test"], **generate_kwargs):

        self.model.to(self.device)
        self.model(torch.ones((8, 8)))
        return ['some outputs']


def init_model(
    args: argparse.Namespace, world_size: int, local_rank: int, model_init
) -> DSPipeline:
    """Initialize the deepspeed model"""

    data_type = getattr(torch, args.dtype)
    model = model_init()

    pipe = DSPipeline(
            model = model,
        dtype=data_type,
        device=local_rank,
    )

    gc.collect()

    pipe.model = deepspeed.init_inference(
        pipe.model,
        dtype=data_type,
        mp_size=world_size,
        replace_with_kernel_inject=args.use_kernel,
        replace_method=args.replace_method,
        max_tokens=args.max_tokens,
        save_mp_checkpoint_path=None,
    )
    return pipe


def generate(
    input_sentences: List[str], pipe: DSPipeline, batch_size: int, **generate_kwargs
) -> List[str]:
    """Generate predictions using a DSPipeline"""
    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))

    inputs = input_sentences[:batch_size]
    outputs = pipe(inputs, **generate_kwargs)
    return outputs
