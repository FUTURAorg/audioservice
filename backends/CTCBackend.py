import copy
import datetime

import numpy as np
import torch
from .BufferedInput import AudioDataLayer, BufferedInput
from torch.utils.data import DataLoader
from nemo.core.classes import IterableDataset

from backends.abc.backend import ASRBackend

from nemo.collections.asr.models import EncDecCTCModelBPE

class CTCBackend(ASRBackend):
    
    def __init__(self):
        start_load = datetime.datetime.now()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = EncDecCTCModelBPE.restore_from('E:/Programming/ThesisBackend/audioservice/models/ctcsmall.nemo', map_location=self.device)
        self.model = EncDecCTCModelBPE.from_pretrained(model_name="stt_ru_conformer_ctc_large")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.name = 'CTCBackend'
        
        cfg = copy.deepcopy(self.model._cfg)
        self.vocab = list(cfg.decoder.vocabulary)
        self.vocab.append('_')

        self.data_layer = AudioDataLayer(sample_rate=16000)
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)
        
        self.prev_char = ''
        print(f"Loaded model conformerCTC on device {self.device} in {(datetime.datetime.now() - start_load).total_seconds()} seconds")
        
        
    def infer_signal(self, signal):
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        
        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(self.model.device), audio_signal_len.to(self.model.device)
        log_probs, _, _ = self.model.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        return log_probs
    
    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s
    
    def _decode(self, frame):
        logits = self.infer_signal(frame).cpu().numpy()[0]
        
        decoded = self._greedy_decoder(
            logits, 
            self.vocab
        )
        
        return decoded
    
    def greedy_merge(self, s):
        s_merged = ''
        self.prev_char = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged
    
    @torch.no_grad()
    def transcribe(self, frame):
        t = self.greedy_merge(self._decode(frame))
        t = t.replace('▁', ' ')
        return t, 'ru'
    


class NVIDIACTCBackend(ASRBackend):
    def __init__(self):
        start_load = datetime.datetime.now()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EncDecCTCModelBPE.from_pretrained(model_name="stt_ru_conformer_ctc_large")
        self.model.to(self.device)
        print(f"Loaded model conformerCTC on device {self.device} in {(datetime.datetime.now() - start_load).total_seconds()} seconds")
    
    def transcribe(self, frame):
        logits = self.infer_signal(frame).cpu().numpy()[0]
        
        return decoded