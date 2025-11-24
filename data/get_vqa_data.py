from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import Counter
from PIL import Image
import torch

def _soft_score(n: int) -> float:
    return min(1.0, n / 3.0)  # 目前未使用，先保留

class VQAHFDataset(Dataset):
    def __init__(self, hf_split, ans2id=None, image_size=None, return_pil=True):
        self.ds = hf_split
        self.ans2id = ans2id
        self.image_size = image_size
        self.return_pil = return_pil
    
    def __len__(self): 
        return len(self.ds)
    
    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"]  # HF VQAv2 通常是 PIL.Image

        if not self.return_pil:
            # 统一转 RGB，再按需 resize 到 (image_size, image_size)，最后转 Tensor[3,H,W] in [0,1]
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            img = TF.to_tensor(img)  # -> float32 [0,1], CHW

            # 灰度/带 alpha 的情况已通过 convert("RGB") 规避；此处再保险一下
            if img.dim() == 3 and img.size(0) == 1:
                img = img.expand(3, -1, -1)
            elif img.dim() == 3 and img.size(0) == 4:
                img = img[:3, ...]

        # 其余字段照旧
        return {"image": img, "question": ex["question"], "answers": ex["answers"]}

def _majority_answer(answers_list):
    toks = []
    for a in answers_list:
        s = (a.get("answer") or "").strip().lower()
        if s:
            toks.append(s)
    if not toks:
        return "unknown"
    return Counter(toks).most_common(1)[0][0]

def vqa_collate(batch):
    # 若 Dataset 返回的是 Tensor（推荐：return_pil=False），可直接 stack
    if torch.is_tensor(batch[0]["image"]):
        imgs = torch.stack([b["image"] for b in batch], dim=0)   # [B,3,H,W]
    else:
        # 兼容 return_pil=True 的情况（不推荐）：此处转一次
        imgs = torch.stack([TF.to_tensor(b["image"].convert("RGB")) for b in batch], dim=0)

    qs = [b["question"] for b in batch]
    ans_txt = [_majority_answer(b["answers"]) for b in batch]    # list[str]

    return {"image": imgs, "question": qs, "answer_text": ans_txt}
