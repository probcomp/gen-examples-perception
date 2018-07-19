import socket
import json
import select
import sys
import math
import time
import pdb
from abc import ABC, abstractmethod

def _send(socket, data):
    try:
        serialized = json.dumps(data)
        print("_send: ", serialized)
    except (TypeError, ValueError):
        raise Exception('You can only send JSON-serializable data')
    socket.send(('%d\n' % len(serialized)).encode())
    socket.sendall(serialized.encode())

def _recv(socket):
    # https://github.com/mdebbar/jsonsocket/blob/master/jsonsocket.py
    # read length of data, character by character, until EOL
    length_str = ''
    char = socket.recv(1).decode()
    while char != '\n':
        length_str += char
        char = socket.recv(1).decode()
    length = int(length_str)
    view = memoryview(bytearray(length))
    next_offset = 0
    while length - next_offset > 0:
        recv_size = socket.recv_into(view[next_offset:], length - next_offset)
        next_offset += recv_size
    try:
        deserialized = json.loads(view.tobytes().decode())
        print("_recv: ", deserialized)
    except (TypeError, ValueError):
        raise Exception('Data received was not in JSON format')
    return deserialized

class JSONServer(ABC):

    def __init__(self, port):
        self.host = ''
        self.port = port

    @abstractmethod
    def process(self, request):
        pass

    def run(self):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((self.host, self.port))
        serversocket.listen(0)
        while True:
            (sock, address) = serversocket.accept()
            request = _recv(sock)
            print("Server received request: ", request)
            response = self.process(request)
            print("Server sending response: ", response)
            _send(sock, response)
            sock.close()
        

class JSONClient(object):

    def __init__(self, port):
        self.host = 'localhost'
        self.port = port
        
    def execute(self, request):
        print("Client sending request: ", request)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        _send(sock, request)
        response = _recv(sock)
        print("Client received response: ", response)
        sock.close()
        return response

