import os

import ray
import ray.util
from ray.air import ScalingConfig, Checkpoint
from ray.train.batch_predictor import BatchPredictor

from deepspeed_predictor import DeepSpeedPredictor



import dataclasses  as dc

@dc.dataclass
class DistributedConfig:
    num_worker_groups: int = 2
    num_groups: int = 2
    num_cpus: int = 1
    num_gpus: int= 0
    num_cpus_per_worker_group:int = 2

    batch_size: int = 2
    dtype: str = 'float16'

    use_kernel: bool = True
    replace_method: str= ''
    max_tokens: int = 1024

    


class DistributedModel:

    def __init__(self, predictor, config):
        self.predictor = predictor
        self.config  = config
    def inference(self, df):
        ds = (
            ray.data.from_pandas(df)
            .repartition(self.config.num_cpus_per_worker_group * 2)
            .random_shuffle()
            .fully_executed()
        )


        pred = self.predictor.predict(
            ds,
            batch_size=1,
            num_cpus_per_worker=2,
            min_scoring_workers=self.config.num_worker_groups,
            max_scoring_workers=self.config.num_worker_groups,
        )
        return pred


def distributed(model_func, config):
    runtime_env = {"working_dir": os.path.dirname(__file__)}
    ray.init(runtime_env=runtime_env, num_gpus=0, num_cpus=8)

    group_scaling_config = ScalingConfig(
        use_gpu=False,
        num_workers=config.num_cpus_per_worker_group,
        trainer_resources={"CPU": 2},
    )
    batch_predictor = BatchPredictor(
            Checkpoint.from_dict({'args': config}),
        DeepSpeedPredictor,
        scaling_config=group_scaling_config,
        model_init=model_func
    )
    return DistributedModel(batch_predictor, config)
