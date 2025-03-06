# components/__init__.py
from src.components.distances.gaussian import GaussianDistance
from src.components.layers.linear import LinearLayer
from src.components.layers.cfconv import CFConv
from src.components.models.GraphVAMPNet import GraphVampNet
from src.components.losses.vampnet_loss import vampnet_loss
from src.components.scores.vamp_score import vamp_score

try:
    import torch_scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("Warning: torch_scatter not found. GraphVampNet functionality will be limited.")

__all__ = [
    'GaussianDistance',
    'LinearLayer',
    'CFConv',
    'vampnet_loss',
    'vamp_score'
]

if HAS_TORCH_SCATTER:
    __all__.append('GraphVampNet')
