import torch
import numpy as np
import einops


def RGB2SH(rgb):
    c1 = 0.282095
    return (rgb - 0.5) / c1

def SH2RGB(sh):
    c1 = 0.282095
    return sh *c1 + 0.5

def eval_sh_point(n, env):
    '''
    Computes lighting from eq.(3) in:
    An Efficient Representation for Irradiance Environment Maps by Ravi Ramamoorthi
    https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf.

    This function evaluates the SH value for a point, used to lit an object with a single point from the environment map.

    Args:
       n - Normal vector, [B, 3] or [W, H, 3]
       env - Vector with SH coefficients, [3x9] or [B, 3, 9]

    Outputs:
       Evaluated SH value for the given normal vector, [B, 3] or [W, H, 3]
    '''
    
    c1 = 0.282095
    c2 = 0.488603
    c3 = 1.092548
    c4 = 0.315392
    c5 = 0.546274
    
    c = env
    if len(c.shape)==1:
        c=c.unsqueeze(0) #expand for batch
    if c.shape[1]!=9: # by default we need transpose, but just a check
        c=torch.transpose(c, -1, -2)

    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]

    irradiance = (
        c[:,0] * c1 +
        c[:,1] * c2*y +
        c[:,2] * c2*z +
        c[:,3] * c2*x +
        c[:,4] * c3*x*y +
        c[:,5] * c3*y*z +
        c[:,6] * c4*(3*z*z-1) +
        c[:,7] * c3*x*z +
        c[:,8] * c5*(x*x-y*y)
    )
    return irradiance

def eval_sh_hemisphere(n, env):
    '''
    Computes lighting using eq.(12, 13) from:
    "An Efficient Representation for Irradiance Environment Maps" by Ravi Ramamoorthi
    https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf.

    This is the closed-form solution for the integral over the hemisphere.
    Used when the object is lit by light coming from all directions within the hemisphere.

    Args:
        n - Normal vector, [B, 3] or [W, H, 3]
        env - Vector with SH coefficients, [3x9] or [B, 3, 9]

    Outputs:
        Evaluated SH value for the given normal vector, [B, 3] or [W, H, 3]
    '''
  
    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708

    c = env
    if len(c.shape)==1:
        c=c.unsqueeze(0) #expand for batch
    if c.shape[1]!=9: # by default we need transpose, but just a check
        c=torch.transpose(c, -1, -2)

    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]

    irradiance = (
        c1 * c[:,8] * (x ** 2 - y ** 2) +
        c3 * c[:,6] * (z ** 2) +
        c4 * c[:,0] -
        c5 * c[:,6] +
        2 * c1 * c[:,4] * x * y +
        2 * c1 * c[:,7] * x * z +
        2 * c1 * c[:,5] * y * z +
        2 * c2 * c[:,3] * x +
        2 * c2 * c[:,1] * y +
        2 * c2 * c[:,2] * z
    )
    return irradiance

def eval_sh_shadowed(shs_gauss, sh_scene):
    """
    Evaluates the dot product for SH coefficients.

    Args:
       sh_gauss: SH coefficients for Gaussians, [..., 3x9]
       sh_scene: SH coefficients for the environment map, [3x9]

    Returns:
       [..., C]: Dot product result with preserved batch dimensions.
    """
    assert shs_gauss.shape[-1] == sh_scene.shape[-1]
    return einops.einsum(shs_gauss, sh_scene, 'b i j, i j->b i')