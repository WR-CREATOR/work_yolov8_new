import torch

nl=8
batch1 = {"image": torch.randn(nl, 3, 640, 640), "target": torch.randn(nl, 15, 4)}
batch1["batch_idx"] = torch.zeros(nl)

batch2 = {"image": torch.randn(nl, 3, 640, 640), "target": torch.randn(nl, 15, 4)}
batch2["batch_idx"] = torch.zeros(nl)
batch = [batch1, batch2]

values = list(zip(*[list(b.values()) for b in batch]))
print(values)
