
from concurrent import futures
import grpc
import redis

from BufferedInput import BufferedInput
from backends.CTCBackend import CTCBackend
from backends.WhisperBackend import WhisperBackend

from futuracommon.protos import audioservice_pb2
from futuracommon.protos import audioservice_pb2_grpc

import numpy as np


rd = redis.Redis(host="127.0.0.1", port=6379, db=0)
model = WhisperBackend()

class AudioStreamer(audioservice_pb2_grpc.AudioStreamerServicer):
    
    def __init__(self) -> None:
        super().__init__()
        self.buffer = BufferedInput()
    
    def StreamAudio(self, request_iterator, context):
        for audio_chunk in request_iterator:
            signal = np.frombuffer(audio_chunk.audio_data, dtype=np.float32)
            self.buffer.update_buffer(signal)
            
            text, lang = model.transcribe(self.buffer.buffer)
            print(text, lang)
        
            if lang == 'ru':
                rd.hset(name=audio_chunk.client_id, key="transribe", value=text)
            
            print("Received audio chunk of size:", len(audio_chunk.audio_data))
            
        return audioservice_pb2.StreamStatus(success=True, message="Stream received successfully")



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audioservice_pb2_grpc.add_AudioStreamerServicer_to_server(AudioStreamer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
