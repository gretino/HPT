import torch
import numpy as np
import copy
from torch.utils.data import Dataset
from hpt.utils.normalizer import LinearNormalizer


class ECGDataset(Dataset):
    def __init__(
    self,
    data: np.ndarray,               # Shape: (num_samples, sequence_length, num_leads)
    labels: np.ndarray,             # Shape: (num_samples,)
    normalizer: LinearNormalizer = None,
    domain_name: str = 'ecg_dataset',
    observation_horizon: int = 1000,
    pad_before: int = 0,
    pad_after: int = 0,
    normalize_state: bool = True,
    **kwargs,
    ):
        self.data = data                    # (num_samples, sequence_length, num_leads)
        self.labels = labels                # (num_samples,)
        self.normalizer = normalizer
        self.domain_name = domain_name
        self.observation_horizon = observation_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.normalize_state = normalize_state

    # Compute normalization parameters if normalizer is not provided
        if self.normalizer is None and self.normalize_state:
            self.normalizer = LinearNormalizer()
            # Prepare data for normalizer
            data_dict = {'state': self.data}
            self.normalizer.fit(data_dict, last_n_dims=1, mode='gaussian')
        elif not self.normalize_state:
            self.normalizer = None

    def __len__(self):
        return len(self.data)
    
    def get_normalizer(self):
        return self.normalizer
    
    def __getitem__(self, idx: int):
        # Get the ECG data sequence
        state = self.data[idx]  # Shape: (sequence_length, num_leads)
        
        # Apply padding if necessary
        if self.pad_before > 0 or self.pad_after > 0:
            state = np.pad(
                state,
                ((self.pad_before, self.pad_after), (0, 0)),
                mode='constant',
                constant_values=0,
            )
        
        # Limit to observation_horizon
        state = state[:self.observation_horizon]

        # Normalize the state data if normalizer is provided
        if self.normalizer and self.normalize_state:
            state = self.normalizer['state'].normalize(state)
        
        # Create the sample dictionary
        sample = {
            'domain': self.domain_name,
            'data': state,
            'label': self.labels[idx],
        }
        
        return sample