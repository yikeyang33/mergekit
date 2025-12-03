import torch
from typing import Any, Dict, List, Optional
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


def muon_process(delta: torch.Tensor) -> torch.Tensor:
    """
    Apply Muon optimization processing to a tensor.
    Delta W = U Sigma V^T
    """
    if len(delta.shape) < 2:
        return delta

    original_dtype = delta.dtype
    delta = delta.to(torch.float32)
    
    # SVD decomposition
    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        delta_muon = U @ Vh
    except RuntimeError:
        # Fallback if SVD fails (e.g. on some devices or specific shapes)
        return delta.to(original_dtype)

    return delta_muon.to(original_dtype)


class MuonMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        # Sort models by 'order' parameter to ensure correct sequence
        # W_0, W_1, ..., W_N
        keys = sorted(
            tensors.keys(),
            key=lambda k: self.tensor_parameters[k]["order"]
        )

        ordered_tensors = [tensors[key] for key in keys]
        
        # Weights for the gradients (W_i - W_{i-1})
        # Note: The first model W_0 doesn't have a weight in the formula W_0 + sum(w_i * Muon(delta))
        # But in our config, we might assign weights to all.
        # Let's assume the weight parameter on model i corresponds to w_i in the formula.
        # For i=0, weight is ignored or treated as 1.0 for the base.
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        rectify_embed_sizes(self.weight_info, ordered_tensors)

        # Ensure all tensors are on the same device and type
        device = ordered_tensors[0].device
        dtype = ordered_tensors[0].dtype
        
        # Initialize merged state with W_0
        merged_state = ordered_tensors[0].clone()
        
        # Iterate through the sequence
        # Formula: W_{merged} = W_0 + sum_{i=1}^{N-1} w_i * Muon(W_i - W_{i-1})
        # Note: The 'weights' list from the user script corresponds to the gradient weights.
        # In the user script:
        # i=0: merged_state = W_0
        # i>0: delta = W_i - W_{i-1}; merged_state += weights[i-1] * Muon(delta)
        # Here weights[i] in our list corresponds to the weight assigned to model i.
        
        for i in range(1, len(ordered_tensors)):
            current_tensor = ordered_tensors[i]
            prev_tensor = ordered_tensors[i-1]
            
            delta = current_tensor - prev_tensor
            
            # Apply Muon processing
            if len(delta.shape) >= 2:
                delta = muon_process(delta)
            
            # Apply weight
            # The weight for this step should be the weight associated with the current model (or the gradient step)
            w = weights[i]
            
            merged_state += w * delta

        return merged_state

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MuonMerge(MergeMethod):
    def name(self) -> str:
        return "muon"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Muon Merge"

    def parameters(self) -> List[ConfigParameterDef]:
        return []

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="order", required=True),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return MuonMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
        )
