"""Test script to prove that PyTorch can use CUDA GPU for faster operations"""

import torch
import time


def experiment(device='CPU', num_trials=1000, shape=None):
    if shape is None:
        shape = (2**12, 2**12)
    # Start
    start_time = time.time()
    # Instantiate the tensors
    if device == 'CPU':
        a = torch.rand(*shape)
    elif device == 'GPU':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device is not available!')
        a = torch.rand(*shape).cuda()
    else:
        raise ValueError('Invalid computation device specified!')
    # Do the computations
    for i in range(num_trials):
        a += a
    # End
    elapsed_time = time.time()-start_time
    print(device + ' time = %.12f' % elapsed_time)
    return elapsed_time


# Run the CPU experiment
experiment('CPU')

# Run the GPU experiment multiple times in order to warm up overhead from copying to GPU memory.
# The last 2 trials should be about the same runtime and much faster than CPU
experiment('GPU')
experiment('GPU')
experiment('GPU')
