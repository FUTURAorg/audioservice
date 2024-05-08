import abc

class ASRBackend(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def transcribe(self, data) -> str:
        pass