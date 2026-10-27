# SPDX-License-Identifier: AGPL-3.0-or-later
"""An idle light reminder scaled to observed scan passes, suspended during capture."""
class LightTimer:
    def __init__(self):
        self.ask_at=None;self.off_at=None;self.scan_at=None;self.longest=0.;self.last_tick=None
        # Retain the previous lease until a complete scan pass supplies its duration.
        self.idle_wait=300.;self.reply_wait=120.
    def start(self,now):self.ask_at=now+self.idle_wait;self.off_at=None;self.last_tick=now
    def stop(self):self.ask_at=None;self.off_at=None;self.scan_at=None;self.last_tick=None
    def extend(self,now):
        if self.off_at is None or now>=self.off_at:return False
        self.ask_at=now+self.idle_wait;self.off_at=None;return True
    def activity(self,now,active):
        if active:
            if self.scan_at is None:self.scan_at=now
            if self.last_tick is not None:
                delta=max(0,now-self.last_tick)
                if self.ask_at is not None:self.ask_at+=delta
            self.off_at=None
        elif self.scan_at is not None:
            self.longest=max(self.longest,now-self.scan_at);self.scan_at=None
            self.idle_wait=max(self.idle_wait,self.longest);self.reply_wait=max(self.reply_wait,self.longest)
            if self.ask_at is not None:self.ask_at=now+self.idle_wait
        self.last_tick=now
    def tick(self,now):
        if self.scan_at is not None:return False
        if self.ask_at is not None and now>=self.ask_at and self.off_at is None:self.off_at=now+self.reply_wait
        return self.off_at is not None and now>=self.off_at
    def remaining(self,now):return None if self.off_at is None or self.scan_at is not None else max(0,int(self.off_at-now+.999))
