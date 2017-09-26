#! /home/aphillips5/envs/nlpenv/bin/python3


import os, inspect, sys, time

import rpyc


class CrunchService(rpyc.Service):

    def on_connect(self):
        self.users_list = os.listdir('/home')
        self.log_file = os.environ['HOME'] + '/crunch_server.log'

    def on_disconnect(self):
        pass

    def invalid_user(self, user):
        msg = '%s: Invalid user "%s" tried to connect!' % (current_time(), user)
        write_log(msg, log=self.log_file)
        return None

    def exposed_run(self, user, obj, args=[], kwargs={}):
        if not user in self.users_list: return self.invalid_user(user)
        write_log('%s: Crunching %s, %s, %s...' % (current_time(), repr(obj), repr(args), repr(kwargs)), log=self.log_file)
        res = None
        if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj): res = obj(*args, **kwargs)
        if inspect.isclass(obj):
            res = obj.run()
        write_log('%s: Finished crunching!' % (current_time()), log=self.log_file)
        return res

    def exposed_cpu_count(self):
        return os.cpu_count()

def current_time():
    # from common
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

def write_log(msg, print_=True, log=None):
    # from common

    with open(log, 'a') as lf:
        lf.write(msg + '\n')
    if print_: print(msg)
    return


if __name__ == '__main__':
    port = 9999 if len(sys.argv) < 2 else sys.argv[1]
    from rpyc.utils.server import ForkingServer

    t = ForkingServer(CrunchService, 'localhost', port=port)
    print('Running CrunchService. Use Ctrl+C to quit.')
    t.start()
    print('Killing CrunchService.')
