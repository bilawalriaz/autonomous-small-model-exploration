"""Check if clean/corrupt pairs actually produce different predictions."""
import sys
sys.path.insert(0, "src")
import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.utils import load_json, PROJECT_ROOT

bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
model = bundle.model
t = bundle.tokenizer
model.eval()

pairs = load_json(str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"))

for p in pairs:
    cids = t(p["clean"], return_tensors="pt")["input_ids"].to(model.device)
    xids = t(p["corrupt"], return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        cl = model(cids).logits
        xl = model(xids).logits
    cp = torch.softmax(cl[0, -1], dim=-1)
    xp = torch.softmax(xl[0, -1], dim=-1)
    kl = torch.nn.functional.kl_div(xp.log(), cp, reduction="sum").item()
    ct = [(t.decode([i]), round(v.item(), 4)) for i, v in zip(torch.topk(cp, 3).indices, torch.topk(cp, 3).values)]
    xt = [(t.decode([i]), round(v.item(), 4)) for i, v in zip(torch.topk(xp, 3).indices, torch.topk(xp, 3).values)]
    print(f"{p['id']}: KL={kl:.4f}")
    print(f"  clean:  {ct}")
    print(f"  corrupt: {xt}")
