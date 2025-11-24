import os, csv, math, random
from collections import defaultdict
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image

from frameworks.framework_januspro import JanusProAdvFramework
from PIL import Image, ImageDraw, ImageFont

from matplotlib import cycler

class HardBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self.data: List[Dict] = []

    def __len__(self): 
        return len(self.data)

    def push_many(self, items: List[Dict]):
        if not items:
            return
        self.data.extend(items)
        self.data.sort(key=lambda x: float(x["H"]), reverse=True)
        if len(self.data) > self.capacity:
            self.data = self.data[: self.capacity]

    def sample(self, n: int, temperature: float = 2.0, pop: bool = False) -> List[Dict]:
        if len(self.data) == 0:
            return []
        k = min(int(n), len(self.data))

        idx = torch.arange(len(self.data), dtype=torch.float32)
        inv_rank = (len(self.data) - 1) - idx
        probs = torch.softmax(inv_rank / max(1e-6, float(temperature)), dim=0)
        idxs = torch.multinomial(probs, num_samples=k, replacement=False).tolist()
        items = [self.data[i] for i in idxs]
        if pop:
            for i in sorted(idxs, reverse=True):
                self.data.pop(i)
        return items


class CLIPScorer:
    def __init__(self, device, name="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        import open_clip
        self.model, _, self.pp = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=device
        )
        self.tok = open_clip.get_tokenizer(name)
        self.device = device

    @torch.no_grad()
    def score(self, images: torch.Tensor, texts: List[str], micro_bs: int = 32) -> torch.Tensor:

        assert images.dim() == 4 and images.size(1) == 3
        N = images.size(0)
        target = 224

        import torch.nn.functional as F

        _, _, H, W = images.shape
        scale = target / min(H, W)
        newH, newW = int(round(H * scale)), int(round(W * scale))
        x = F.interpolate(images, size=(newH, newW), mode="bicubic", align_corners=False, antialias=True)
        y0 = max(0, (newH - target) // 2)
        x0 = max(0, (newW - target) // 2)
        x = x[:, :, y0:y0+target, x0:x0+target]

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        tok = self.tok(texts)           
        tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()} if isinstance(tok, dict) else tok.to(self.device)

        outs = []
        for i in range(0, N, max(1, micro_bs)):
            xi = x[i:i+micro_bs]
            ti = {k: v[i:i+micro_bs] for k, v in tok.items()} if isinstance(tok, dict) else tok[i:i+micro_bs]
            fe_i = F.normalize(self.model.encode_image(xi), dim=-1)
            fe_t = F.normalize(self.model.encode_text(ti), dim=-1)
            outs.append((fe_i * fe_t).sum(dim=-1))
        return torch.cat(outs, 0)


class AdvTrainer:
    def __init__(self, config: Dict):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_distributed else 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}" if self.is_distributed else "cuda")
            if self.is_distributed:
                torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.cfg = config

        _base = JanusProAdvFramework(
            model_path=config.get("model_path", "deepseek-ai/Janus-Pro-7B"),
            num_answers=int(config["num_answers"]),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            finetune_last_k=config.get("finetune_last_k", 0),
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device=self.device,
            eps_max=config.get("eps_max", 0.02),
        )
        self.base = _base
    
        self.model = DDP(
            _base, device_ids=[self.local_rank], output_device=self.local_rank,
            find_unused_parameters=True,          
            broadcast_buffers=False,
            gradient_as_bucket_view=True       

        )

        self.base.img_gen_size    = int(config.get("img_gen_size", 256))
        self.base.img_patch_size  = int(config.get("img_patch_size", 16))
        self.base.cfg_weight      = float(config.get("cfg_weight", 5.0))
        self.base.gen_temperature = float(config.get("gen_temperature", 1.0))

        self._disc_params = [p for p in self.base.mm.language_model.parameters() if p.requires_grad]
        self._gen_params  = list(self.base.gen_params)

        self.disc_opt = torch.optim.AdamW(
            self._disc_params,
            lr=float(config.get("disc_lr", 3e-6)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            betas=(0.9, 0.999),
        )
        self.gen_opt = torch.optim.AdamW(
            self._gen_params,
            lr=float(config.get("gen_lr", 3e-2)),
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )

        amp_enabled = bool(config.get("use_amp", True))
        lm_dtype = (self.model.module.mm.language_model.dtype
                    if hasattr(self.model, "module") else self.model.mm.language_model.dtype)
        if lm_dtype in (torch.float16, torch.bfloat16):
            amp_enabled = False
        self.use_amp = amp_enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.workdir = config.get("logdir", "logs_januspro")
        self.ckpt_dir = os.path.join(self.workdir, "checkpoints")
        self.metrics_dir = os.path.join(self.workdir, "metrics")
        self.vis_dir = os.path.join(self.workdir, "visualizations")
        self.game_vis_dir = os.path.join(self.vis_dir, "game")
        self.decoded_dir = os.path.join(self.vis_dir, "decoded_samples")
        for d in [self.workdir, self.ckpt_dir, self.metrics_dir, self.vis_dir, self.game_vis_dir, self.decoded_dir]:
            os.makedirs(d, exist_ok=True)

        self.log_stride    = int(config.get("log_stride"))
        self.plot_every    = int(config.get("plot_interval"))
        self.save_every    = int(config.get("save_interval"))
        self.csv_log_stride = self.plot_every

        self.train_csv_path = os.path.join(self.metrics_dir, "train_log.csv")
        if (self.rank == 0) and (not os.path.exists(self.train_csv_path)):
            with open(self.train_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "total", "gen_CE_adv", "disc_CE", "eps"])

        self._palette = {
            "disc_path": "#2170b5",  
            "gen_path":  "#ef3b2c",  
            "loss_1": "#2878b5",
            "loss_2": "#9ac9db",
            "loss_3": "#f8ac8c",
            "loss_4": "#c82423",
            "loss_5": "#ff8884",
            "eps":    "#4D4D4D",
            "grid":   "#D0D7DE",
        }
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 220,
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#2D333B",
            "xtick.color": "#2D333B",
            "ytick.color": "#2D333B",
            "axes.grid": True,
            "grid.color": "#D0D7DE",
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        })
        from matplotlib import cycler as _cycler
        plt.rcParams["axes.prop_cycle"] = _cycler(color=[
            self._palette["loss_1"], self._palette["loss_2"], self._palette["loss_3"],
            self._palette["loss_4"], self._palette["loss_5"],
        ])
        self.path_plot_ema_beta = float(config.get("path_plot_ema_beta", 0.8))

        self._path_stride = int(config.get("path_stride", max(10, self.log_stride)))
        self.disc_clip = float(config.get("disc_clip_norm", 0.0) or 0.0)
        self.gen_clip  = float(config.get("gen_clip_norm",  0.0) or 0.0)
        self._path_xy_D: List[Tuple[float,float]] = []
        self._path_xy_G: List[Tuple[float,float]] = []
        self._deltaJ_steps, self._deltaJ_D, self._deltaJ_G = [], [], []

        self.eps_history: List[Tuple[int, float]] = []
        self.eps_max = float(config.get("eps_max", 0.02))

        self.step = 0
        self.loss_history = defaultdict(list)
        self._loss_buf = defaultdict(lambda: {"sum": 0.0, "n": 0})
        for k in ["gen_ce_adv_ma", "disc_ce_decoded_ma", "disc_ce_clean_ma", "disc_ce_hard_ma"]:
            self._loss_buf[k] = {"sum": 0.0, "n": 0}
            self.loss_history[k] = []

        self.gdbg_csv_path = os.path.join(self.metrics_dir, "gdbg_log.csv")
        if self.rank == 0 and (not os.path.exists(self.gdbg_csv_path)):
            with open(self.gdbg_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step","||∇logeps||","Δlogeps","eps","CE_adv@G","ΔJ_D","ΔJ_G"])

        self.d_updates = 1
        self.g_updates = 1

        self.gen_lambda = float(config.get("gen_lambda", 120.0))

        self.use_decoded_dpass  = bool(config.get("use_decoded_dpass", True))
        self.use_clean_dpass    = bool(config.get("use_clean_dpass", True))
        self.d_decoded_weight   = float(config.get("d_decoded_weight", 1.0))
        self.d_clean_weight     = float(config.get("d_clean_weight",   0.5))

        self.clip_tau    = float(config.get("clip_tau", 0.30))
        self.clip_lambda = float(config.get("clip_lambda", 0.2))
        self.cand_K      = int(config.get("cand_K", 3))
        self.hard_topk   = int(config.get("hard_topk", 1))
        self.buffer_size = int(config.get("buffer_size", 4096))
        self.buffer_temp = float(config.get("buffer_temp", 2.0))
        self.hard_bs     = int(config.get("hard_bs", 8))
        self.decoded_micro_bs = int(config.get("decoded_micro_bs", 2))
        self.hard_push_max_per_step = int(config.get("hard_push_max_per_step", 0))  # 0=不限制
        self.mine_stride = int(config.get("mine_stride", 1))       

        self.hard_buf = HardBuffer(self.buffer_size)
        self.clip = CLIPScorer(
            self.device,
            name=config.get("clip_name", "ViT-B-16"),
            pretrained=config.get("clip_ckpt", "laion2b_s34b_b88k"),
        )

        self._flat_idx = {}
        offset = 0
        for p in (self._disc_params + self._gen_params):
            n = p.numel()
            self._flat_idx[id(p)] = (offset, offset + n)
            offset += n
        self._flat_dim = offset
        rp_seed = int(config.get("rp_seed", 114514))
        torch.manual_seed(rp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(rp_seed)
            torch.cuda.manual_seed_all(rp_seed)
        self._shared_v1 = torch.randn(self._flat_dim, device=self.device, dtype=torch.float32)
        self._shared_v1 = self._shared_v1 / self._shared_v1.norm().clamp_min(1e-8)
        v2r = torch.randn_like(self._shared_v1)
        v2r = v2r - (v2r * self._shared_v1).sum() * self._shared_v1
        self._shared_v2 = v2r / v2r.norm().clamp_min(1e-8)

        self.base.reg_enable = bool(config.get("reg_enable", True))
        self.base.reg_l2     = float(config.get("reg_l2", 0.0))
        self.base.reg_tv     = float(config.get("reg_tv", 0.0))
        self.base.reg_eps    = float(config.get("reg_eps", 0.0))

        self.hard_dir = os.path.join(self.vis_dir, "hard_samples")
        os.makedirs(self.hard_dir, exist_ok=True)
        self.hard_csv = os.path.join(self.metrics_dir, "hard_samples.csv")
        if (self.rank == 0) and (not os.path.exists(self.hard_csv)):
            with open(self.hard_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "step","global_idx","qid","cand_k","H","CE","CLIP","tau","lambda",
                    "th_mode","th_value","picked","img_path","question","answer"
                ])
        self.hard_thresh_mode   = str(config.get("hard_thresh_mode", "quantile")).lower()
        self.hard_thresh_q      = float(config.get("hard_thresh_q", 0.80))
        self.hard_thresh_value  = float(config.get("hard_thresh_value", 0.0))
        self.hard_save_max_step = int(config.get("hard_save_max_per_step", 16))


    @staticmethod
    def _quantile(x: torch.Tensor, q: float) -> float:
        q = min(0.999, max(0.001, float(q)))
        return float(torch.quantile(x.detach().float(), q).item())
    
    @staticmethod
    def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
        x = x.detach().clamp(0,1).cpu()
        if x.dim()==3 and x.size(0) in (1,3,4):
            if x.size(0)==1: x = x.expand(3,-1,-1)
            if x.size(0)==4: x = x[:3]
            arr = (x.permute(1,2,0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, "RGB")
        raise ValueError("expect CHW tensor")

    @staticmethod
    def _draw_panel(orig: Image.Image, hard: Image.Image,
                    question: str, gt: str, pred: str) -> Image.Image:
        from PIL import Image, ImageDraw, ImageFont

        H = max(orig.height, hard.height)
        if orig.height != H:
            orig = orig.resize((orig.width, H), Image.BILINEAR)
        if hard.height != H:
            hard = hard.resize((hard.width, H), Image.BILINEAR)

        pad = 8
        text_h = 120
        W = orig.width + hard.width + pad * 3   
        Htot = H + text_h + pad * 2

        canvas = Image.new("RGB", (W, Htot), (255, 255, 255))
        canvas.paste(orig, (pad, pad))
        canvas.paste(hard, (orig.width + pad * 2, pad))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        x_text = pad
        y_text = H + pad + 2
        maxw = W - pad * 2

        def wrap_line(s: str, max_width: int):
            out, cur = [], ""
            for ch in s:
                w, _ = draw.textsize(cur + ch, font=font)
                if w > max_width:
                    out.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                out.append(cur)
            return out

        lines = []
        lines += wrap_line(f"Q: {question}", maxw)
        lines += wrap_line(f"GT: {gt}", maxw)
        if pred:
            lines += wrap_line(f"Pred: {pred}", maxw)

        for ln in lines:
            draw.text((x_text, y_text), ln, fill=(20, 20, 20), font=font)
            y_text += 20

        return canvas

        
    @torch.no_grad()
    def _save_hard_visuals(
        self,
        orig_batch_tensor: torch.Tensor,     
        cand_imgs: torch.Tensor,            
        questions: List[str],
        answers: List[str],
        preds: List[str],
        orig_indices: List[int],             
        tag: str,
    ):
        if self.rank != 0 or len(orig_indices) == 0:
            return
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np, torch.nn.functional as F

        os.makedirs(self.hard_dir, exist_ok=True)

        def _to_pil(x: torch.Tensor) -> Image.Image:
            x = x.detach().clamp(0,1).cpu()
            if x.dim()==3 and x.size(0) in (1,3,4):
                if x.size(0)==1: x = x.expand(3,-1,-1)
                if x.size(0)==4: x = x[:3]
                arr = (x.permute(1,2,0).numpy() * 255).astype(np.uint8)
                return Image.fromarray(arr, "RGB")
            raise ValueError("expect CHW tensor")

        H, W = cand_imgs.shape[-2], cand_imgs.shape[-1]
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        panels = []
        pad = 6
        footer_h = 64
        maxh = H + footer_h + 2 * pad
        maxw = (W * 2) + 3 * pad

        for j in range(len(orig_indices)):
            i0 = int(orig_indices[j])                      
            raw_pil  = _to_pil(orig_batch_tensor[i0])
            hard_pil = _to_pil(cand_imgs[j])

            canvas = Image.new("RGB", (maxw, maxh), (255,255,255))
            canvas.paste(raw_pil,  (pad, pad))
            canvas.paste(hard_pil, (pad*2 + W, pad))

            draw = ImageDraw.Draw(canvas)
            x0, ytxt = pad, pad + H + 4

            def _wrap(text, width=56):
                out, cur = [], ""
                for ch in text:
                    cur += ch
                    if len(cur) >= width:
                        out.append(cur); cur = ""
                if cur: out.append(cur)
                return "\n".join(out)

            q  = f"Q: {questions[j] if j < len(questions) else questions[0]}"
            gt = f"GT: {answers[j]   if j < len(answers)   else answers[0]}"
            pr = f"Pred: {preds[j]   if j < len(preds)     else ''}"
            block = _wrap(q) + "\n" + _wrap(gt) + "\n" + _wrap(pr)
            draw.text((x0, ytxt), block, fill=(10,10,10), font=font)

            panels.append(canvas)

        tot_h = sum(p.size[1] for p in panels) + (len(panels)-1)*pad
        grid = Image.new("RGB", (maxw, tot_h), (255,255,255))
        y = 0
        for p in panels:
            grid.paste(p, (0,y)); y += p.size[1] + pad

        out_path = os.path.join(self.hard_dir, f"{tag}.png")
        grid.save(out_path, quality=92)


    def _wrap(text, maxw):
            lines, cur = [], ""
            for ch in text:
                w,_ = draw.textsize(cur + ch, font=font)
                if w > maxw:
                    lines.append(cur); cur = ch
                else:
                    cur += ch
            if cur: lines.append(cur)
            return lines

    def _hard_threshold(self, H: torch.Tensor) -> float:
        if self.hard_thresh_mode == "absolute":
            return float(self.hard_thresh_value)
        return self._quantile(H, self.hard_thresh_q)

    @torch.no_grad()
    def _project_delta_shared(self, which: str, old_params: List[torch.Tensor]) -> Tuple[float, float]:
        v1, v2 = self._shared_v1, self._shared_v2
        d = torch.zeros(self._flat_dim, device=self.device, dtype=torch.float32)
        plist = self._disc_params if which == "D" else self._gen_params
        for p, old in zip(plist, old_params):
            s, e = self._flat_idx[id(p)]
            d[s:e] = (p.data.detach().float().view(-1) - old.view(-1).to(d.device))
        return float((d * v1).sum().item()), float((d * v2).sum().item())

    @torch.no_grad()
    def _append_path(self, which: str, dx: float, dy: float, normalize: bool = True):
        if normalize:
            n = (dx*dx + dy*dy) ** 0.5
            if n > 1e-12:
                dx, dy = dx / n, dy / n
        path = self._path_xy_D if which == "D" else self._path_xy_G
        if not path: path.append((0.0, 0.0))
        x0, y0 = path[-1]
        path.append((x0 + dx, y0 + dy))


    def _read_eps(self) -> Optional[float]:
        try:
            mod = getattr(self.base, "perturb", None)
            if mod is None or getattr(mod, "log_eps", None) is None:
                return None
            with torch.no_grad():
                eps01 = torch.sigmoid(mod.log_eps.detach().float())
                eps = float((mod.eps_max * eps01).item())
                if not (0.0 <= eps <= max(1.0, float(mod.eps_max) * 1.5)):
                    print(f"[WARN] eps out of expected range: {eps} (eps_max={mod.eps_max})")
                    eps = max(0.0, min(float(mod.eps_max), eps))
                return eps
        except Exception:
            return None


    @torch.no_grad()
    def _mine_and_buffer(self, batch: Dict):
        imgs_batch = batch["image"]
        assert torch.is_tensor(imgs_batch) and imgs_batch.dim() == 4 and imgs_batch.size(1) == 3, \
            "Expect batch['image'] as Tensor[B,3,H,W] (把预处理放进Dataset+collate后，这里永远为Tensor)"

        qs  = batch["question"]
        ans = batch["answer_text"]
        B   = len(qs)
        if B == 0 or self.cand_K <= 0:
            return

        Ht, Wt = imgs_batch.shape[-2], imgs_batch.shape[-1] 

        prompts, cand_meta = [], []
        for i in range(B):
            for k in range(self.cand_K):
                prompts.append(qs[i])
                cand_meta.append((i, k))  
        # [N,3,h,w] in [0,1]
        cand_imgs = self.base._official_generate_batch(
            prompts, inject_adv=True,
            seed=114514 + self.step * 97,
            img_size=self.base.img_gen_size,
            patch_size=self.base.img_patch_size,
            cfg_weight=self.base.cfg_weight,
            # temperature=self.base.gen_temperature,
        )

        if cand_imgs.shape[-2:] != (Ht, Wt):
            cand_imgs = torch.nn.functional.interpolate(
                cand_imgs, size=(Ht, Wt), mode="bilinear", align_corners=False
            )

        cand_qs = [qs[i]  for (i, _) in cand_meta]
        cand_as = [ans[i] for (i, _) in cand_meta]
        N = cand_imgs.size(0)
       s_clip = self.clip.score(cand_imgs, cand_qs, micro_bs=self.decoded_micro_bs)  # [N], cosine 相似度

        ce_vals = []
        mb = max(1, self.decoded_micro_bs)
        tmp_batch = {"image": cand_imgs, "question": cand_qs, "answer_text": cand_as}
        for i in range(0, N, mb):
            sub = {k: (v[i:i+mb] if torch.is_tensor(v) else v[i:i+mb]) for k, v in tmp_batch.items()}
            ce, _ = self.model(sub)
            ce_vals.extend([float(ce.detach())] * min(mb, N - i))
        ce_vals = torch.tensor(ce_vals, device=self.device, dtype=torch.float32)

        clip_hinge = torch.clamp(self.clip_tau - s_clip, min=0.0)
        H = ce_vals + self.clip_lambda * clip_hinge
        th = (self._hard_threshold(H))
        mask = (H >= th) | (s_clip <= self.clip_tau)
        chosen_idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()

        by_src = {}
        for idx in chosen_idx:
            i0, k0 = cand_meta[idx]
            by_src.setdefault(i0, []).append(idx)
        picked_final = []
        for i0, lst in by_src.items():
            if not lst: continue
            lst = sorted(lst, key=lambda t: float(H[t]), reverse=True)[:max(1, self.hard_topk)]
            picked_final.extend(lst)

        save_visual = (self.rank == 0) and (((self.step + 1) % max(1, int(self.cfg.get("plot_interval", 10))) == 0))
        pred_cache: Dict[int, str] = {}

        if save_visual and picked_final:
            viz_ids = picked_final[: self.hard_save_max_step]
            viz_imgs = cand_imgs[viz_ids]  # Tensor[Nv,3,H,W]
            viz_qs   = [cand_qs[j] for j in viz_ids]
            preds = self.base.infer_answers_batch(
                images=viz_imgs,
                questions=viz_qs,
                max_new_tokens=64,
                temperature=0.0,
            )
            for jid, p in zip(viz_ids, preds):
                pred_cache[jid] = p

        items = []
        saved_cnt = 0
        if self.rank == 0:
            import csv as _csv

        for j in picked_final:
            i0, k0 = cand_meta[j]
            H_j  = float(H[j].item())
            CE_j = float(ce_vals[j].item())
            C_j  = float(s_clip[j].item())

            items.append({
                "image": cand_imgs[j].detach().cpu(),  
                "question": cand_qs[j],
                "answer_text": cand_as[j],
                "H": H_j,
            })
            img_path = ""
            if save_visual and self.rank == 0 and (saved_cnt < self.hard_save_max_step):
                try:
                    orig_pil = self._tensor_to_pil(imgs_batch[i0])
                    hard_pil = self._tensor_to_pil(cand_imgs[j])
                    pred_txt = pred_cache.get(j, "")
                    panel = self._draw_panel(
                        orig=orig_pil, hard=hard_pil,
                        question=cand_qs[j], gt=cand_as[j], pred=pred_txt
                    )
                    img_path = os.path.join(
                        self.hard_dir,
                        f"step_{self.step+1:07d}_qid{i0}_k{k0}_H{H_j:.3f}_CE{CE_j:.3f}_CLIP{C_j:.3f}.jpg"
                    )
                    panel.save(img_path, quality=92)
                    saved_cnt += 1
                except Exception:
                    pass

            if self.rank == 0:
                with open(self.hard_csv, "a", newline="") as f:
                    _csv.writer(f).writerow([
                        self.step + 1, j, i0, k0, H_j, CE_j, C_j,
                        float(self.clip_tau), float(self.clip_lambda),
                        self.hard_thresh_mode,
                        (self.hard_thresh_value if self.hard_thresh_mode=="absolute" else self.hard_thresh_q),
                        1, (img_path if (save_visual and saved_cnt <= self.hard_save_max_step) else ""),
                        cand_qs[j], cand_as[j]
                    ])

        if self.rank == 0:
            remained = set(range(N)) - set(picked_final)
            for j in list(remained)[: max(0, self.hard_save_max_step - saved_cnt)]:
                i0, k0 = cand_meta[j]
                H_j  = float(H[j].item()); CE_j = float(ce_vals[j].item()); C_j = float(s_clip[j].item())
                with open(self.hard_csv, "a", newline="") as f:
                    _csv.writer(f).writerow([
                        self.step + 1, j, i0, k0, H_j, CE_j, C_j,
                        float(self.clip_tau), float(self.clip_lambda),
                        self.hard_thresh_mode,
                        (self.hard_thresh_value if self.hard_thresh_mode=="absolute" else self.hard_thresh_q),
                        0, "", cand_qs[j], cand_as[j]
                    ])

        if self.hard_push_max_per_step > 0 and len(items) > self.hard_push_max_per_step:
            items.sort(key=lambda d: float(d["H"]), reverse=True)
            items = items[: self.hard_push_max_per_step]
        self.hard_buf.push_many(items)

    @torch.no_grad()
    def _plot_eps_curve(self):
        if self.rank != 0 or len(self.eps_history) == 0: return
        s, v = zip(*self.eps_history)
        plt.figure(figsize=(9.5, 4.4))
        plt.plot(s, v, label=f"eps (max={self.eps_max:g})", lw=1.6, marker="o",
                markevery=max(1, len(s)//30), color=self._palette["eps"])
        plt.xlabel("Global step"); plt.ylabel("eps")
        plt.title("Adversarial Budget (eps)")
        ymax = max(1e-6, float(self.eps_max)) * 1.1
        plt.ylim(0.0, ymax)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "eps_curve.png")); plt.close()


    @staticmethod
    def _ema(values, beta=0.6):
        out, m = [], None
        for t, v in enumerate(values, 1):
            m = v if m is None else beta * m + (1 - beta) * v
            out.append(m / (1 - beta ** t))
        return out

    @torch.no_grad()
    def _plot_game_geometry(self):
        if self.rank != 0:
            return

        have_D = len(self._path_xy_D) >= 2
        have_G = len(self._path_xy_G) >= 2
        if not (have_D or have_G):
            return

        import numpy as np
        import matplotlib.pyplot as plt

        max_pts = int(self.cfg.get("path_plot_max_points", 200))

        def _decimate(arr: np.ndarray, k: int) -> np.ndarray:
            if arr.shape[0] <= k:
                return arr
            step = max(1, arr.shape[0] // k)
            return arr[::step]

        def _to_np(path_list):
            return np.asarray(path_list, dtype=np.float32)

        D = _to_np(self._path_xy_D) if have_D else None
        G = _to_np(self._path_xy_G) if have_G else None
        Dd = _decimate(D, max_pts) if have_D else None
        Gd = _decimate(G, max_pts) if have_G else None

        xs, ys = [], []
        if have_D:
            xs.append(D[:, 0]); ys.append(D[:, 1])
        if have_G:
            xs.append(G[:, 0]); ys.append(G[:, 1])
        xs = np.concatenate(xs) if xs else np.array([0.0])
        ys = np.concatenate(ys) if ys else np.array([0.0])

        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        dx = max(1e-8, xmax - xmin)
        dy = max(1e-8, ymax - ymin)
        span = max(dx, dy)
        pad = 0.08 * span
        xlim = (xmin - pad, xmax + pad)
        ylim = (ymin - pad, ymax + pad)

        plt.figure(figsize=(6.6, 6.6))
        ax = plt.gca()

        # D-path
        if have_D:
            ax.plot(D[:, 0], D[:, 1],
                    lw=2.0, color=self._palette["disc_path"], alpha=0.9, label="D-path")
            ax.scatter(Dd[:, 0], Dd[:, 1], s=8, color=self._palette["disc_path"], alpha=0.9)
            ax.scatter(D[0, 0], D[0, 1], s=36, facecolors="none",
                    edgecolors=self._palette["disc_path"], linewidths=1.6, zorder=5)
            ax.scatter(D[-1, 0], D[-1, 1], s=36, marker="x",
                    color=self._palette["disc_path"], linewidths=1.6, zorder=5)

        if have_G:
            ax.plot(G[:, 0], G[:, 1],
                    lw=2.0, color=self._palette["gen_path"], alpha=0.9, label="G-path")
            ax.scatter(Gd[:, 0], Gd[:, 1], s=8, color=self._palette["gen_path"], alpha=0.9)
            ax.scatter(G[0, 0], G[0, 1], s=36, facecolors="none",
                    edgecolors=self._palette["gen_path"], linewidths=1.6, zorder=5)
            ax.scatter(G[-1, 0], G[-1, 1], s=36, marker="x",
                    color=self._palette["gen_path"], linewidths=1.6, zorder=5)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.28, linestyle="--", color=self._palette["grid"])
        ax.set_xlabel("Proj-1")
        ax.set_ylabel("Proj-2")
        ax.set_title("Optimization Path (true step lengths)")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.game_vis_dir, "opt_path_2d.png"))
        plt.close()

        if len(self._deltaJ_steps) >= 2:
            xs = np.array(self._deltaJ_steps, dtype=np.int64)
            dD = np.array(self._deltaJ_D, dtype=np.float32)
            dG = np.array(self._deltaJ_G, dtype=np.float32)
            plt.figure(figsize=(9.5, 4.0))
            plt.plot(xs, dD, label="ΔJ_D", lw=1.6, color=self._palette["disc_path"])
            plt.plot(xs, dG, label="ΔJ_G", lw=1.6, color=self._palette["gen_path"])
            plt.axhline(0.0, color="#999", lw=1.0)
            plt.xlabel("Global step"); plt.ylabel("ΔJ ≈ ⟨∇J, Δθ⟩")
            plt.title("First-order Objective Change")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(self.game_vis_dir, "deltaJ_est.png")); plt.close()


    @torch.no_grad()
    def _plot_losses(self):
        if self.rank != 0: return
        keys = ["gen_ce_adv_ma", "disc_ce_decoded_ma", "disc_ce_clean_ma", "disc_ce_hard_ma"]
        if not any(self.loss_history.get(k) for k in keys): return

        plt.figure(figsize=(11.5, 5.0))
        if self.loss_history.get("gen_ce_adv_ma"):
            x = (np.arange(len(self.loss_history["gen_ce_adv_ma"])) + 1) * self.log_stride
            plt.plot(x, self.loss_history["gen_ce_adv_ma"], lw=1.8, label="G: CE(adv) (↑)", color=self._palette["loss_4"])
        if self.loss_history.get("disc_ce_decoded_ma"):
            x = (np.arange(len(self.loss_history["disc_ce_decoded_ma"])) + 1) * self.log_stride
            plt.plot(x, self.loss_history["disc_ce_decoded_ma"], lw=1.8, label="D: CE(hard) (↓)", color=self._palette["loss_1"])
        if self.loss_history.get("disc_ce_hard_ma"):
            x = (np.arange(len(self.loss_history["disc_ce_hard_ma"])) + 1) * self.log_stride
            plt.plot(x, self.loss_history["disc_ce_hard_ma"], lw=1.8, label="D: CE(adv) (↓)", color=self._palette["loss_3"])

        plt.xlabel("Global step"); plt.ylabel("Loss / CE")
        plt.title("G vs D")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f"loss_curve_sep_{self.step}.png")); plt.close()

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        num_epochs = int(self.cfg.get("num_epochs", 1))
        lam = self.gen_lambda

        for epoch in range(num_epochs):
            if self.is_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            pbar = tqdm(total=len(train_loader), desc=f"Train[{epoch+1}/{num_epochs}]", dynamic_ncols=True) if self.rank == 0 else None

            for batch in train_loader:
                batch = self._to_device(batch)

                self.disc_opt.zero_grad(set_to_none=True)
                self.gen_opt.zero_grad(set_to_none=True)
                for p in self._gen_params:  p.requires_grad = True
                for p in self._disc_params: p.requires_grad = False

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    self.base._adv_flag = True
                    loss_adv_G, aux = self.model(batch)
                    self.base._adv_flag = False
                    reg_term = aux.get("reg", torch.tensor(0.0, device=self.device, dtype=loss_adv_G.dtype))
                    gen_loss = - lam * loss_adv_G + reg_term

                old_gen_params = [p.data.detach().clone().float() for p in self._gen_params]
                self.scaler.scale(gen_loss).backward()
                self.scaler.unscale_(self.gen_opt)
                if self.gen_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._gen_params, self.gen_clip)
                g_logeps = getattr(self.base.perturb, "log_eps", None)
                grad_norm = (0.0 if (g_logeps is None or g_logeps.grad is None)
                             else float(g_logeps.grad.detach().abs().mean().item()))
                logeps_old = float(self.base.perturb.log_eps.detach().item()) if hasattr(self.base.perturb, "log_eps") else 0.0

                gen_grads_before = []
                for p in self._gen_params:
                    if p.grad is None:
                        gen_grads_before.append(None)
                    else:
                        gen_grads_before.append((-1.0 / max(lam, 1e-12)) * p.grad.detach().float().clone())

                self.scaler.step(self.gen_opt)
                self.scaler.update()

                gdx, gdy = self._project_delta_shared("G", old_gen_params)
                if (self.step + 1) % self._path_stride == 0:
                    self._append_path("G", gdx, gdy)

                deltaG = 0.0
                for p, gJ, old in zip(self._gen_params, gen_grads_before, old_gen_params):
                    if gJ is None: continue
                    d = (p.data.detach().float() - old)
                    deltaG += float((gJ * d).sum().item())

                logeps_new = float(self.base.perturb.log_eps.detach().item()) if hasattr(self.base.perturb, "log_eps") else logeps_old
                dlogeps = (logeps_new - logeps_old)
                eps_now = self._read_eps() or 0.0
                gen_ce_this = float(loss_adv_G.detach())

                if self.rank == 0:
                    with open(self.gdbg_csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            self.step + 1, float(grad_norm), float(dlogeps),
                            float(eps_now), float(loss_adv_G.detach()),
                            0.0, float(deltaG)
                        ])

                if (self.step + 1) % max(1, self.mine_stride) == 0:
                    self._mine_and_buffer(batch)

                disc_ce_this = 0.0
                deltaJ_D_this = 0.0
                dxD_acc, dyD_acc = 0.0, 0.0

                if self.use_decoded_dpass:
                    with torch.no_grad():
                        adv_only_imgs = self.base._make_decoded_images(batch, mode="adv_only")

                    if (self.rank == 0) and ((self.step == 0) or ((self.step + 1) % max(1, int(self.cfg.get("save_decoded_every", 10))) == 0)):
                        self.base._save_decoded_samples(batch["image"], adv_only_imgs, tag=f"step_{self.step+1:07d}")

                    self.disc_opt.zero_grad(set_to_none=True)
                    self.gen_opt.zero_grad(set_to_none=True)
                    for p in self._gen_params:  p.requires_grad = False
                    for p in self._disc_params: p.requires_grad = True

                    noisy_batch = dict(batch)
                    noisy_batch["image"] = adv_only_imgs

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        self.base._adv_flag = False
                        loss_decoded, _ = self.model(noisy_batch)
                        self.base._adv_flag = False
                        disc_loss_decoded = self.d_decoded_weight * loss_decoded

                    old_disc_params_dec = [p.data.detach().clone().float() for p in self._disc_params]
                    self.scaler.scale(disc_loss_decoded).backward()
                    self.scaler.unscale_(self.disc_opt)
                    if self.disc_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                    disc_grads_before_dec = [p.grad.detach().float().clone() if p.grad is not None else None for p in self._disc_params]
                    self.scaler.step(self.disc_opt); self.scaler.update()

                    dx_dec, dy_dec = self._project_delta_shared("D", old_disc_params_dec)
                    dxD_acc += dx_dec; dyD_acc += dy_dec

                    delta_dec = 0.0
                    for p, g, old in zip(self._disc_params, disc_grads_before_dec, old_disc_params_dec):
                        if g is None: continue
                        d = (p.data.detach().float() - old)
                        delta_dec += float((g * d).sum().item())
                    deltaJ_D_this += delta_dec

                    disc_ce_this += float(loss_decoded.detach())
                    self._loss_buf["disc_ce_decoded_ma"]["sum"] += float(loss_decoded.detach())
                    self._loss_buf["disc_ce_decoded_ma"]["n"]   += 1

                if self.use_clean_dpass:
                    self.disc_opt.zero_grad(set_to_none=True)
                    self.gen_opt.zero_grad(set_to_none=True)
                    for p in self._gen_params:  p.requires_grad = False
                    for p in self._disc_params: p.requires_grad = True

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        self.base._adv_flag = False
                        loss_clean, _ = self.model(batch)
                        self.base._adv_flag = False
                        disc_loss_clean = self.d_clean_weight * loss_clean

                    old_disc_params_clean = [p.data.detach().clone().float() for p in self._disc_params]
                    self.scaler.scale(disc_loss_clean).backward()
                    self.scaler.unscale_(self.disc_opt)
                    if self.disc_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                    disc_grads_before_clean = [p.grad.detach().float().clone() if p.grad is not None else None for p in self._disc_params]
                    self.scaler.step(self.disc_opt); self.scaler.update()

                    dx_cln, dy_cln = self._project_delta_shared("D", old_disc_params_clean)
                    dxD_acc += dx_cln; dyD_acc += dy_cln

                    delta_cln = 0.0
                    for p, g, old in zip(self._disc_params, disc_grads_before_clean, old_disc_params_clean):
                        if g is None: continue
                        d = (p.data.detach().float() - old)
                        delta_cln += float((g * d).sum().item())
                    deltaJ_D_this += delta_cln

                    disc_ce_this += float(loss_clean.detach())
                    self._loss_buf["disc_ce_clean_ma"]["sum"] += float(loss_clean.detach())
                    self._loss_buf["disc_ce_clean_ma"]["n"]   += 1

                hard_items = self.hard_buf.sample(n=self.hard_bs, temperature=self.buffer_temp, pop=True)

                if hard_items:
        
                    if torch.is_tensor(batch["image"]):
                        Ht, Wt = batch["image"].shape[-2], batch["image"].shape[-1]
                    else:
                        im0 = batch["image"][0]
                        if torch.is_tensor(im0):
                            Ht, Wt = im0.shape[-2], im0.shape[-1]
                        else:
                            Wt, Ht = im0.size
                    
                    resized_imgs = []
                    for it in hard_items:
                        t = it["image"].detach().float().clamp(0, 1).cpu()
                        if t.dim() == 3 and t.size(0) == 1:
                            t = t.expand(3, -1, -1)      
                        if t.dim() == 3 and t.size(0) == 4:
                            t = t[:3, ...]              
                        if t.shape[-2:] != (Ht, Wt):
                            t = F.interpolate(
                                t.unsqueeze(0), size=(Ht, Wt),
                                mode="bilinear", align_corners=False
                            ).squeeze(0)
                        resized_imgs.append(t)
                    
                    hb = {
                        "image": torch.stack(resized_imgs, 0).to(self.device),
                        "question": [it["question"] for it in hard_items],
                        "answer_text": [it["answer_text"] for it in hard_items],
                    }

                    self.disc_opt.zero_grad(set_to_none=True)
                    
                    for p in self._gen_params:  p.requires_grad = False
                    for p in self._disc_params: p.requires_grad = True

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        self.base._adv_flag = False
                        loss_hard, _ = self.model(hb)
                        self.base._adv_flag = False
                        disc_loss_hard = 1.0 * loss_hard
                    
                    old_disc_params_hard = [p.data.detach().clone().float() for p in self._disc_params]
                    self.scaler.scale(disc_loss_hard).backward()
                    self.scaler.unscale_(self.disc_opt)
                    if self.disc_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._disc_params, self.disc_clip)
                    disc_grads_before_hard = [p.grad.detach().float().clone() if p.grad is not None else None for p in self._disc_params]
                    self.scaler.step(self.disc_opt); self.scaler.update()

                    dx_h, dy_h = self._project_delta_shared("D", old_disc_params_hard)
                    dxD_acc += dx_h; dyD_acc += dy_h
                    delta_h = 0.0
                    for p, g, old in zip(self._disc_params, disc_grads_before_hard, old_disc_params_hard):
                        if g is None: continue
                        d = (p.data.detach().float() - old)
                        delta_h += float((g * d).sum().item())
                    deltaJ_D_this += delta_h

                    disc_ce_this += float(loss_hard.detach())
                    self._loss_buf["disc_ce_hard_ma"]["sum"] += float(loss_hard.detach())
                    self._loss_buf["disc_ce_hard_ma"]["n"]   += 1

                for p in self._disc_params: p.requires_grad = True

                eps_val = self._read_eps()
                if eps_val is not None:
                    self.eps_history.append((self.step + 1, eps_val))

                d_pass_cnt = max(1, int(self.use_decoded_dpass) + int(self.use_clean_dpass)) + (1 if self.hard_bs > 0 else 0)
                self._loss_buf["gen_ce_adv_ma"]["sum"] += float(gen_ce_this)
                self._loss_buf["gen_ce_adv_ma"]["n"]   += 1
                self._loss_buf["disc_ce_ma"]["sum"]    += (disc_ce_this / max(1, d_pass_cnt))
                self._loss_buf["disc_ce_ma"]["n"]      += 1

                self._deltaJ_steps.append(self.step + 1)
                self._deltaJ_D.append(float(deltaJ_D_this))
                self._deltaJ_G.append(float(deltaG))

                if (self.step + 1) % self.log_stride == 0:
                    for k, b in self._loss_buf.items():
                        if b["n"] > 0:
                            self.loss_history[k].append(b["sum"] / b["n"])
                            b["sum"], b["n"] = 0.0, 0

                if (self.rank == 0) and ((self.step + 1) % max(self.plot_every, self.log_stride) == 0):
                    self._plot_eps_curve()
                    self._plot_game_geometry()
                    self._plot_losses()
                if (self.step + 1) % self._path_stride == 0:
                    self._append_path("D", dxD_acc, dyD_acc)

                self.step += 1
                if self.rank == 0 and pbar:
                    pbar.set_postfix_str(
                        f"D(CE)={(disc_ce_this/max(1,d_pass_cnt)):.4f} | "
                        f"G(CE)={gen_ce_this:.4f} | "
                        f"eps={(eps_val if eps_val is not None else 0.0):.4f}/{self.eps_max:g} | "
                        f"buf={len(self.hard_buf)}"
                    )
                    pbar.update(1)

                if (self.step % self.save_every == 0) and self.rank == 0:
                    self.save_ckpt(step_tag=f"step_{self.step}")

                if self.rank == 0 and ((self.step + 1) % self.csv_log_stride == 0):
                    with open(self.train_csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            self.step + 1,
                            float(disc_ce_this + gen_ce_this),
                            float(gen_ce_this),
                            float(disc_ce_this / max(1, d_pass_cnt)),
                            ("" if eps_val is None else float(eps_val)),
                        ])

            if pbar: pbar.close()

    @torch.no_grad()
    def _save_adv_samples(self, batch: Dict, questions=None):
        return

    def save_ckpt(self, step_tag: str):
        if self.rank != 0: return
        path = os.path.join(self.ckpt_dir, f"{step_tag}.pt")
        to_save = {
            "step": self.step,
            "model": self.base.state_dict(),
            "perturb_only": self.base.perturb.state_dict(),
            "disc_opt": self.disc_opt.state_dict(),
            "gen_opt": self.gen_opt.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(to_save, path)
        print(f"[CKPT] saved => {path}")

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return out
