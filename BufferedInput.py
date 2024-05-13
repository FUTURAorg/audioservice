import datetime
import numpy as np
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader

class BufferedInput:
    def __init__(self, frame_len = 1, frame_overlap=2, clear_after = 5):
        self.sr = 16000
        self.frame_len = frame_len
        self.n_frame_len = int(self.sr * self.frame_len)
        
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)

        self._buffer = np.zeros(shape=self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        
        self.last_update = datetime.datetime.now()
        self.clear_after = clear_after
        
        self.reset()
    
    def update_buffer(self, frame):
        if (datetime.datetime.now() - self.last_update).total_seconds() >= self.clear_after:
            self.reset()
        
        self._buffer[:-self.n_frame_len] = self._buffer[self.n_frame_len:]
        self._buffer[-self.n_frame_len:] = frame
        
        self.last_update = datetime.datetime.now()
    
    @property
    def buffer(self):
        return self._buffer
    
    def reset(self):
        self._buffer=np.zeros(shape=self._buffer.shape, dtype=np.float32)


class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1