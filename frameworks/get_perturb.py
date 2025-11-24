import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

def get_perturb(self, image_embeds: torch.Tensor) -> torch.Tensor:
    
    raw_delta = self.perturb_net(image_embeds)  # [B, embed_dim]
    eps = self.config.get('perturb_epsilon', 1.0)
    norm   = raw_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta  = raw_delta / norm * eps
    
    return delta
