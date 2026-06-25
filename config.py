# In config.py
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
NUM_WORKERS = 4 if USE_CUDA else 0
BATCH_SIZE = 16 if USE_CUDA else 4