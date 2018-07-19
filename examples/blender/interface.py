#import bpy
import socket
import json
import select
import sys
import math
import time
import pdb

def _send(socket, data):
    try:
        serialized = json.dumps(data)
    except (TypeError, ValueError), e:
        raise Exception('You can only send JSON-serializable data')
    socket.send('%d\n' % len(serialized))
    socket.sendall(serialized)

def _recv(socket):
    # https://github.com/mdebbar/jsonsocket/blob/master/jsonsocket.py
    # read length of data, character by character, until EOL
    length_str = ''
    char = socket.recv(1)
    while char != '\n':
        length_str += char
        char = socket.recv(1)
    length = int(length_str)
    view = memoryview(bytearray(length))
    next_offset = 0
    while length - next_offset > 0:
        recv_size = socket.recv_into(view[next_offset:], length - next_offset)
        next_offset += recv_size
    try:
        deserialized = json.loads(view.tobytes())
    except (TypeError, ValueError), e:
        raise Exception('Data received was not in JSON format')
    return deserialized

class Server(object):

    def __init__(self, port):
        self.host = 'localhost'
        self.port = port
        self.connect()

    def connect(self):
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((self.host, self.port))
        self.serversocket.listen(0)

    def run(self):
        while True:
            (socket, address) = self.serversocket.accept()
            request = _recv(socket)
            print("Server received request: ", request)
            #response = self.process(request)
            response = { "foo" : "bar" }
            print("Server sending response: ", response)
            _send(socket, response)
            socket.close()


class Client(object):

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

