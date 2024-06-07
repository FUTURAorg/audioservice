from concurrent import futures
import os
import grpc
import logging 

from backends.BufferedInput import BufferedInput

from futuracommon.protos import audioservice_pb2, audioservice_pb2_grpc
from futuracommon.protos import nlp_pb2_grpc, nlp_pb2
from futuracommon.protos import healthcheck_pb2, healthcheck_pb2_grpc
from futuracommon.protos import backend_change_pb2, backend_change_pb2_grpc
from futuracommon.SessionManager.RedisManager import RedisSessionManager

import numpy as np

from backends.Holder import BackendHolder

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AUDIOSERVICE')


NLP_SERVICE_ADDR = f'{os.environ.get("NLP_SERVICE_HOST", "nlpservice")}'
REDIS_HOST = os.environ.get("REDIS_HOST", "session_manager")
REDIS_PORT = 6379
REDIS_DB = 0

sessionManager = RedisSessionManager(REDIS_HOST, REDIS_PORT, REDIS_DB)


class AudioStreamer(audioservice_pb2_grpc.AudioStreamerServicer):
    
    def __init__(self) -> None:
        super().__init__()
        self.buffer = BufferedInput(frame_len=0.5, frame_overlap=6, clear_after=3)
        self.prev_text = ''
        self.backend = BackendHolder()
    
    def StreamAudio(self, request_iterator, context):
        channel = grpc.insecure_channel(NLP_SERVICE_ADDR)
        stub = nlp_pb2_grpc.NLPServiceStub(channel)
        
        for audio_chunk in request_iterator:
            signal = np.frombuffer(audio_chunk.audio_data, dtype=np.float32)
            self.buffer.update_buffer(signal)
            
            text, lang = self.backend.transcribe(self.buffer.buffer)
            print(text, lang)
        
            if lang == 'ru' and text != '' and text != self.prev_text:
                self.prev_text = text
                sessionManager.save(cliend_id=audio_chunk.client_id, key="q", value=text)
                stub.NotifySuccess(nlp_pb2.SuccessNotification(client_id=audio_chunk.client_id))
            
            print("Received audio chunk of size:", len(audio_chunk.audio_data))
        
        channel.close()
        return audioservice_pb2.StreamStatus(success=True, message="Stream received successfully")


class HealthServicer(healthcheck_pb2_grpc.HealthServiceServicer):
    def Check(self, request, context):
        backend = BackendHolder()
        return healthcheck_pb2.HealthResponse(status=1, current_backend=f"{backend.name} ({backend.device})")


class BackendChangerService(backend_change_pb2_grpc.BackendServiceServicer):
    def __init__(self):
        self.backends = BackendHolder()
    
    def ChangeBackend(self, request, context):
        success, message = self.backends.change_backend(request.backend_name)
        return backend_change_pb2.BackendStatus(success=success, message=message)
    
    def GetBackendList(self, request, context):
        backends = self.backends.list_backends()
        return backend_change_pb2.BackendList(backends=backends)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audioservice_pb2_grpc.add_AudioStreamerServicer_to_server(AudioStreamer(), server)
    healthcheck_pb2_grpc.add_HealthServiceServicer_to_server(HealthServicer(), server)
    backend_change_pb2_grpc.add_BackendServiceServicer_to_server(BackendChangerService(), server)
    server.add_insecure_port('[::]:50050')
    server.start()
    logger.info("Listening...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
