"""Pair time-stamped camera frames and run independent LightNav sessions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import math

from lightnav_vln import exchange, predict


class FrameSet:
    def __init__(self, names, max_age=0.45, max_skew=0.08):
        self.names = tuple(names)
        self.max_age_ns = int(max_age * 1e9)
        self.max_skew_ns = int(max_skew * 1e9)
        self.latest = {}
        self.last_stamp = dict.fromkeys(self.names, 0)

    def put(self, name, stamp, jpeg):
        if stamp > self.last_stamp[name]:
            self.latest[name] = (stamp, jpeg)
            self.last_stamp[name] = stamp

    def take(self, now):
        if any(name not in self.latest for name in self.names):
            return None
        stamps = [self.latest[name][0] for name in self.names]
        oldest, newest = min(stamps), max(stamps)
        if newest > now or now - oldest >= self.max_age_ns or newest - oldest > self.max_skew_ns:
            return None
        frames = {name: bytes(self.latest[name][1]) for name in self.names}
        self.latest.clear()
        return oldest, frames


def combine(waypoints):
    front = waypoints["forward"]
    stopped = [name for name, wp in waypoints.items() if wp is None]
    if stopped:
        return None, "STOP " + "+".join(stopped)
    if "chest" not in waypoints:
        return front, "forward"
    chest = waypoints["chest"]
    # Compare coarse intent only. The chest camera has a different pitch and
    # field of view. The forward camera supplies the metric base waypoint.
    lengths = math.hypot(*front[:2]), math.hypot(*chest[:2])
    opposed_translation = min(lengths) >= 0.03 and (
        front[0] * chest[0] + front[1] * chest[1] < 0.5 * lengths[0] * lengths[1]
    )
    opposed_turn = min(abs(front[2]), abs(chest[2])) >= 0.1 and front[2] * chest[2] < 0
    if opposed_translation or opposed_turn:
        return (0.0, 0.0, 0.0), "hold: directions disagree"
    return front, "forward with chest check"


class ViewSessions(ExitStack):
    def __init__(self, url, names, max_age):
        super().__init__()
        import websocket

        self.sockets = {}
        try:
            for name in names:
                ws = websocket.create_connection(url, timeout=2, http_no_proxy=["127.0.0.1"])
                self.callback(ws.close)
                exchange(ws, "login", {"clientId": "jjy-base-" + name})
                exchange(ws, "reset", {})
                ws.settimeout(max_age)
                self.sockets[name] = ws
            self.pool = self.enter_context(ThreadPoolExecutor(max_workers=2)) if len(names) > 1 else None
        except BaseException:
            self.close()
            raise

    def step(self, seq, frames, instruction):
        if self.pool is None:
            wp = predict(self.sockets["forward"], seq, frames["forward"], instruction)
            return combine({"forward": wp})
        requests = {
            name: self.pool.submit(predict, ws, seq, frames[name], instruction)
            for name, ws in self.sockets.items()
        }
        return combine({name: request.result() for name, request in requests.items()})
