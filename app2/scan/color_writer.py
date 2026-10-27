# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded asynchronous JPEG writes, with explicit errors and save-time draining."""
import os,queue,threading

class ColorWriter:
    def __init__(self,limit=64):
        self.queue=queue.Queue(limit);self.error=None;self.written=0;self.bytes=0;self.dropped=0
        self.thread=threading.Thread(target=self._run,name='saved-color-writer',daemon=True);self.thread.start()
    def submit(self,path,jpeg):
        if self.error:self.dropped+=1;return False
        try:self.queue.put_nowait((path,jpeg));return True
        except queue.Full:self.dropped+=1;return False
    def _run(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                path,jpeg=item;temp=path+'.tmp.color'
                try:
                    with open(temp,'wb') as f:f.write(jpeg)
                    os.replace(temp,path);self.written+=1;self.bytes+=len(jpeg)
                finally:
                    if os.path.exists(temp):os.unlink(temp)
            except OSError as e:self.error=str(e)
            finally:self.queue.task_done()
    def flush(self):
        self.queue.join()
        if self.error:raise OSError('Color could not be saved: '+self.error)
    def close(self):
        self.queue.put(None);self.queue.join();self.thread.join(2)
