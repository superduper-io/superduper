**`superduper.ext.torch.training`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/torch/training.py)

## `TorchTrainer` 

```python
TorchTrainer(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     key: 'ModelInputType',
     select: 'Query',
     transform: 't.Optional[t.Callable]' = None,
     metric_values: Dict = None,
     signature: 'Signature' = '*args',
     data_prefetch: 'bool' = False,
     prefetch_size: 'int' = 1000,
     prefetch_factor: 'int' = 100,
     in_memory: 'bool' = True,
     compute_kwargs: 't.Dict' = None,
     objective: Callable,
     loader_kwargs: Dict = None,
     max_iterations: int = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
     no_improve_then_stop: int = 5,
     download: bool = False,
     validation_interval: int = 100,
     listen: str = 'objective',
     optimizer_cls: str = 'Adam',
     optimizer_kwargs: Dict = None,
     optimizer_state: Optional[Dict] = None,
     collate_fn: Optional[Callable] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| key | Model input type key. |
| select | Model select query for training. |
| transform | (optional) transform callable. |
| metric_values | Metric values |
| signature | Model signature. |
| data_prefetch | Boolean for prefetching data before forward pass. |
| prefetch_size | Prefetch batch size. |
| prefetch_factor | Prefetch factor for data prefetching. |
| in_memory | If training in memory. |
| compute_kwargs | Kwargs for compute backend. |
| objective | Objective function |
| loader_kwargs | Kwargs for the dataloader |
| max_iterations | Maximum number of iterations |
| no_improve_then_stop | Number of iterations to wait for improvement before stopping |
| download | Whether to download the data |
| validation_interval | How often to validate |
| listen | Which metric to listen to for early stopping |
| optimizer_cls | Optimizer class |
| optimizer_kwargs | Kwargs for the optimizer |
| optimizer_state | Latest state of the optimizer for contined training |
| collate_fn | Collate function for the dataloader |

Configuration for the PyTorch trainer.

