dmc2gym                   1.0.0                    pypi_0    pypi
gym                       0.26.2                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
isaacgym                  1.0rc4                    dev_0    <develop>

gym 本來是25，但裝不了，先裝26
最後一步 seed() 
AttributeError: 'FurnitureDummy' object has no attribute 'seed'
又裝回25 就可以??

/home/by4212/anaconda3/envs/sim/lib/python3.8/site-packages/gym/core.py:256: DeprecationWarning: WARN: Function `env.seed(seed)` is marked as deprecated and will be removed in the future. Please use `env.reset(seed=seed)` instead

