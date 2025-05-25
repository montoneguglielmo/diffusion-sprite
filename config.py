# network hyperparameters
N_FEAT = 64  # 64 hidden dimension feature
N_CFEAT = 5  # context vector is of size 5
HEIGHT = 16  # 16x16 image
SAVE_DIR = './weights/'

# diffusion hyperparameters
TIMESTEPS = 500
BETA1 = 1e-4
BETA2 = 0.02

# training hyperparameters
BATCH_SIZE = 100
N_EPOCH = 32
LRATE = 1e-3

# Make all variables available when importing *
__all__ = [
    'N_FEAT',
    'N_CFEAT',
    'HEIGHT',
    'SAVE_DIR',
    'TIMESTEPS',
    'BETA1',
    'BETA2',
    'BATCH_SIZE',
    'N_EPOCH',
    'LRATE'
]

