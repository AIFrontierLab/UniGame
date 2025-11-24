import os
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from data.get_vqa_data import VQAHFDataset
from trainers.trainer_januspro import AdvTrainer  
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import Counter

def init_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True
    return False

def build_vocab_fast_hf(hf_train, topk: int):
    cnt = Counter(
        (a.get("answer") or "").strip().lower()
        for answers in hf_train["answers"]
        for a in answers
        if a.get("answer")
    )
    vocab = [a for a, _ in cnt.most_common(topk)]
    ans2id = {a: i for i, a in enumerate(vocab)}
    return vocab, ans2id

def _majority_answer(answers_list):
    toks = [a.get("answer").strip().lower() for a in answers_list if a.get("answer")]
    return Counter(toks).most_common(1)[0][0] if toks else "unknown"

def vqa_collate_generative(batch):
    imgs, qs, ans_txt = [], [], []
    for b in batch:
        imgs.append(b["image"])
        qs.append(b["question"])
        ans_txt.append(_majority_answer(b["answers"]))
    return {
        "image": torch.stack(imgs, 0),
        "question": qs,
        "answer_text": ans_txt,
    }

def _make_loader(dataset, batch_size, sampler, num_workers, collate_fn):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs.update(pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)

def main():
    ddp = init_ddp()
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "training_config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_answers = int(config.get("num_answers", 3129))
    LOCAL_VQAV2 = "your path"
    ds_train = load_dataset(LOCAL_VQAV2, split="train", trust_remote_code=True).train_test_split(test_size=0.1, seed=42)["train"]
    ds_val = load_dataset(LOCAL_VQAV2, split="validation", trust_remote_code=True).train_test_split(test_size=0.9, seed=42)["test"]
    
    _, ans2id = build_vocab_fast_hf(ds_train, topk=num_answers)
    train_dataset = VQAHFDataset(ds_train, ans2id, image_size=384, return_pil=False)
    val_dataset = VQAHFDataset(ds_val, ans2id, image_size=384, return_pil=False)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp else None
    
    num_workers = int(config.get("num_workers", 4))
    train_bs = int(config.get("batch_size", 8))
    val_bs = int(config.get("val_batch_size", train_bs))

    train_loader = _make_loader(
        dataset=train_dataset,
        batch_size=train_bs,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=vqa_collate_generative,
    )
    val_loader = _make_loader(
        dataset=val_dataset,
        batch_size=val_bs,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=vqa_collate_generative,
    )

    trainer = AdvTrainer(config)
    print("Starting training…", "(DDP)" if ddp else "")
    trainer.train(train_loader, val_loader)

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
