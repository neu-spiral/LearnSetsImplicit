# import EquiVSetFlax.utils.flax_evaluation.compute_metrics as flav_metrics
# import EquiVSet.utils.evaluation.compute_metrics as torch_metrics
import numpy as np
import torch
import jax.numpy as jnp
from jax import random
import jax
from torch.utils.data import DataLoader, TensorDataset



def torch_loss(q, S, neg_S):  # Eq. (5) in the paper
    loss = - torch.sum((S * torch.log(q + 1e-12) + (1 - S) * torch.log(1 - q + 1e-12)) * neg_S, dim=-1)
    return loss.mean()

def jnp_loss(q, S, neg_S):  # Eq. (5) in the paper
    loss = - jnp.sum((S * jnp.log(q + 1e-12) + (1 - S) * jnp.log(1 - q + 1e-12)) * neg_S, axis=-1)
    return loss.mean()

def torch_eval(S_set, q):
    _, idx = torch.topk(q, S_set.shape[-1], dim=1, largest=True)


    pre_list = []
    for i in range(len(idx)):
        pre_mask = torch.zeros([S_set.shape[-1]])
        ids = idx[i][:int(torch.sum(S_set[i]).item())]
        pre_mask[ids] = 1
        pre_list.append(pre_mask.unsqueeze(0))

    pre_mask = torch.cat(pre_list, dim=0)
    print(ids, pre_mask)
    true_mask = S_set

    intersection = true_mask * pre_mask
    union = true_mask + pre_mask - intersection
    jc = intersection.sum(dim=-1) / union.sum(dim=-1)
    print(f'torch jc:{jc}')
    return jc


#### partial jit didnt work
# @partial(jax.jit, static_argnames=("k",))
# def get_pre_mask(pre_mask, id, k):
#     ids = id[:k]
#     pre_mask = pre_mask.at[ids].set(1)
#     return pre_mask

def update_mask(pre_mask, id_):
    return jax.lax.cond(jnp.isnan(id_),  # Check if id_ is NaN
                    lambda _: pre_mask,  # If NaN, return pre_mask unchanged
                    lambda _: pre_mask.at[id_.astype(int)].set(1),  # If not NaN, set the corresponding index to 1
                    None)

@jax.jit
def jnp_eval(S_set, q):
    _, idx = jax.lax.top_k(q, S_set.shape[-1])
    # idx = jnp.argpartition(q, -S_set.shape[-1], axis=1)
    # call(lambda x: print(f"shape[0] {x}"), S_set.shape[0])  # 128
    # call(lambda x: print(f"shape[-1] {x}"), S_set.shape[-1])  # 100
    # call(lambda x: print(f"len idx {x}"), len(idx))  # 100

    #### partial jit didn't work
    # pre_list = []
    # # s_size = jnp.sum(S_set[0]).astype(int)  # needs s_size as input
    # for i in range(len(idx)):
    #     pre_mask = jnp.zeros([S_set.shape[-1]])
    #     k = (jnp.sum(S_set[i])).astype(int)
    #     pre_mask = get_pre_mask(pre_mask, idx[i], k)
    #       # this needs static slicing
    #     pre_list.append(jnp.expand_dims(pre_mask, axis=0))

    #### failed loop as they do not allow row assignment pre_mask.at[i:].add in jit
    # pre_mask = jnp.zeros(S_set.shape)
    #
    # def body(i, pre_mask):
    #     s_size = jnp.sum(S_set[i]).astype(int)
    #     pre_mask_row = jnp.zeros([S_set.shape[-1]])
    #     mask = jnp.where(jnp.arange(S_set.shape[-1]) < s_size, True, jnp.nan)
    #     ids = idx[i] * mask
    #     for _, id_ in enumerate(ids):
    #         pre_mask_row = update_mask(pre_mask_row, id_)
    #     pre_mask = pre_mask.at[i:].add(pre_mask_row)
    #     return pre_mask
    #
    # final_i, pre_list = jax.lax.fori_loop(0, len(idx), body, pre_mask)


    ####  Working version, but slow compilation
    pre_list = []
    for i in range(len(idx)):
        s_size = jnp.sum(S_set[i]).astype(int)
        pre_mask_row = jnp.zeros([S_set.shape[-1]])
        mask = jnp.where(jnp.arange(S_set.shape[-1]) < s_size, True, jnp.nan)
        ids = idx[i] * mask  # #output: [id1, id2, id3, Nan,...]
        for _, id_ in enumerate(ids):
            pre_mask_row = update_mask(pre_mask_row, id_) # assign all non-Nan idx to mask
        pre_list.append(jnp.expand_dims(pre_mask_row, axis=0))
    pre_mask = jnp.concatenate(pre_list, axis=0)
    true_mask = S_set

    intersection = true_mask * pre_mask
    union = true_mask + pre_mask - intersection
    jc = intersection.sum(axis=-1) / union.sum(axis=-1)
    # print(f'jnp jc {jc}')  # loss
    jax.debug.print("jc is {jc}", jc=jc)
    return jc

# Dummy data
batch_size = 32
v_size = 100
num_samples = 1000
num_features = 10

V_set = np.load('./history/V_set.npy')
S_set = np.load('./history/S_set.npy')
q = np.load('./history/q.npy')
# S = np.load('./history/S.npy')
# neg_S = np.load('./history/neg_S.npy')

# V_set_jnp = jnp.array(V_set)
S_set_jnp = jnp.array(S_set)
q_jnp = jnp.array(q)
# S_jnp = jnp.array(S)
# neg_S_jnp = jnp.array(neg_S)

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Run both implementations

result_jax = jnp_eval(S_set_jnp, q_jnp)
result_torch = torch_eval(torch.tensor(S_set), torch.tensor(q))
#
# # Check if the results are equal

# print(type(q), type(S), type(neg_S))
# result_torch = torch_loss(torch.from_numpy(q), torch.from_numpy(S), torch.tensor(neg_S))
# result_jnp = jnp_loss(q_jnp, S_jnp, neg_S_jnp)
assert jnp.allclose(result_jax, jnp.array(result_torch)), "Results are not equal!"
print("Both implementations produce the same result.")
print(result_torch, result_jnp)