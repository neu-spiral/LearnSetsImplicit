import torch

from utils.pytorch_helper import set_value_according_index, move_to_device


def compute_metrics(loader, infer_func, v_size, device):
    jc_list = []
    for batch_num, batch in enumerate(loader):
        V_set, S_set = move_to_device(batch, device)

        q = infer_func(V_set, S_set.shape[0])
        # print(f"S_set shape 0: {S_set.shape[0]}")  # 128
        # print(f"S_set shape -1: {S_set.shape[-1]}")  # 100
        _, idx = torch.topk(q, S_set.shape[-1], dim=1, largest=True)

        pre_list = []
        for i in range(len(idx)):
            pre_mask = torch.zeros([S_set.shape[-1]]).to(device)
            # print(f"Pre mask shape: {pre_mask.shape}")
            ids = idx[i][:int(torch.sum(S_set[i]).item())]
            pre_mask[ids] = 1
            pre_list.append(pre_mask.unsqueeze(0))
            # print(f"idx length: {len(idx)}")
        pre_mask = torch.cat(pre_list, dim=0)
        true_mask = S_set

        # print(f"True mask shape: {true_mask.shape}")

        intersection = true_mask * pre_mask
        union = true_mask + pre_mask - intersection
        jc = intersection.sum(dim=-1) / union.sum(dim=-1)
        jc_list.append(jc)

    jca = torch.cat(jc_list, 0).mean(0).item()
    return jca * 100
