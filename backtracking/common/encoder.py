import gin
import numpy as np

@gin.configurable(
  denylist=['input_shape'])
class RandomProjectionEncoder(object):
    def __init__(self, input_shape, latent_dim=3, precision=6):
      self.input_dim = np.prod(input_shape)
      self.proj = np.random.normal(loc=0, scale=1./ np.sqrt(latent_dim), 
                      size=(latent_dim, self.input_dim))
      self.precision = precision

    def __call__(self, x):    
      return tuple(np.around(np.dot(self.proj, x.flatten()), self.precision))
      
@gin.configurable(
  denylist=['input_shape'])
class AtariRandomProjectionEncoder(RandomProjectionEncoder):

  def __init__(self, input_shape, latent_dim=3, precision=6):  
      self.input_dim = np.prod(input_shape[1:])
      self.proj = np.random.normal(loc=0, scale=1./ np.sqrt(latent_dim), 
                      size=(latent_dim, self.input_dim))
      self.precision = precision

  def __call__(self, x):
      return tuple(np.around(np.dot(self.proj, x.__array__()[-1, :, :].flatten()), self.precision))

@gin.configurable(
  denylist=['input_shape'])
class IdentityEncoder(object):
    def __init__(self, input_shape, latent_dim=3, precision=6):
      pass

    def __call__(self, x):    
      return x