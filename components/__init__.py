# components/__init__.py
from components.distances.gaussian import GaussianDistance
from components.layers.linear import LinearLayer
from components.layers.cfconv import CFConv
from components.models.GraphVAMPNet import GraphVampNet

try:
    import torch_scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("Warning: torch_scatter not found. GraphVampNet functionality will be limited.")

__all__ = [
    'GaussianDistance',
    'LinearLayer',
    'CFConv'
]

if HAS_TORCH_SCATTER:
    __all__.append('GraphVampNet')
