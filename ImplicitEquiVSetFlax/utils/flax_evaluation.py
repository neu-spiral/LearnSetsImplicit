import jax
import jax.numpy as jnp
from utils.flax_helper import set_value_according_index


def compute_metrics(loader, infer_func, v_size, device):
    jc_list = []
    for batch_num, batch in enumerate(loader):
        V_set, S_set = batch

        q = infer_func(V_set, S_set.shape[0])
        idx = jnp.argpartition(q, S_set.shape[-1], axis=1)

        pre_list = []
        for i in range(len(idx)):
            pre_mask = jnp.zeros([S_set.shape[-1]])
            ids = idx[i][:int(jnp.sum(S_set[i]))]
            pre_mask[ids] = 1
            pre_list.append(jnp.expand_dims(pre_mask, axis=0))
        pre_mask = jnp.concatenate(pre_list, axis=0)
        true_mask = S_set

        intersection = true_mask * pre_mask
        union = true_mask + pre_mask - intersection
        jc = intersection.sum(axis=-1) / union.sum(axis=-1)
        jc_list.append(jc)

    jca = jnp.concatenate(jc_list, axis=0).mean(0)
    return jca * 100