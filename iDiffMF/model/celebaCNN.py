from jax import numpy as jnp
from flax import linen as nn

class CelebaCNN(nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(256)
    @nn.compact
    def __call__(self, x):
        x = self._forward_features(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        return x

    def _forward_features(self, x):
        for i in range(3):
            x = self.apply_conv(x, i)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')  # Max pooling
        return x

    def apply_conv(self, x, layer_idx):
        kernel = [3, 4, 5][layer_idx]
        kernel_size = (kernel, kernel)
        return nn.Conv(features=[32, 64, 128][layer_idx],
                       kernel_size=kernel_size,
                       strides=(2,2),
                       padding='SAME')(x)