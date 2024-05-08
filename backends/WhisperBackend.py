import torch
import whisper
import datetime

from backends.abc.backend import ASRBackend

class WhisperBackend(ASRBackend):
    def __init__(self):
        start_load = datetime.datetime.now()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model("medium", device=self.device)
        print(f"Loaded model Whisper on device {self.device} in {(datetime.datetime.now() - start_load).total_seconds()} seconds")
        
    
    @torch.no_grad()
    def transcribe(self, signal):
        tmp = self.model.transcribe(signal)
        
        text = tmp['text']
        lang = tmp['language']
        
        return text, lang