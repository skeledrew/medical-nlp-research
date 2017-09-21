#


import os, inspect

import rpyc


class CrunchService(rpyc.Service):

    def __init__(self):
        super(CrunchService, self).__init__()
        self.users_list = os.listdir('/home')

    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    def invalid_user(self, user):
        print('Invalid user {} tried to connect!'.format(user))
        return None

    def exposed_run(self, obj, user):
        if not user in self.users_list: return invalid_user(user)
        if inspect.isfunction(obj): return obj()
        if inspect.isclass(obj):
            obj.run()
            return obj

if __name__ == '__main__':
    port = 9999 if len(sys.argv) < 2 else sys.argv[1]
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(CrunchService, 'localhost', port=port)
    print('Running CrunchService. Use Ctrl+C to quit.')
    t.start()
    print('Killing CrunchService.')
