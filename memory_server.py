#! /usr/bin/env python3

import sys

import rpyc


class MemoryService(rpyc.Service):

    def on_connect(self):
        self._blob = {}

    def on_diconnect(self):
        pass

    def exposed_blob(self, caller_id, key, data=None):
        if not caller_id or (not caller_id in self._blob and not data): return ValueError('Invalid caller_id')
        if not data: return self._blob[caller_id].get(key, ValueError('Specified key does not exist'))
        if not caller_id in self._blob: self._blob[caller_id] = {}
        self._blob[caller_id][key] = data
        return True

if __name__ == '__main__':
    port = 9999 if len(sys.argv) < 2 else int(sys.argv[1])
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(MemoryService, 'localhost', port=port)
    print('Running MemoryService on port %s. Use Ctrl+C to quit.' % (port))
    t.start()
    print('Killing MemoryService.')
