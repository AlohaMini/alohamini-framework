import base64
import json
import math
import unittest
from unittest.mock import Mock

from lightnav_vln import exchange, predict, velocity


class LightNavProtocolTests(unittest.TestCase):
    def next_reply(self, **changes):
        data = {"rc": 0, "seq": 3, "stop": False, "actions": {"actions": [[0.375, -0.5, math.pi]]}}
        data.update(changes)
        return Mock(recv=Mock(return_value=json.dumps({"action": "next", "data": data})))

    def test_frame_and_first_waypoint(self):
        ws = self.next_reply()
        self.assertEqual(velocity(predict(ws, 3, b"jpeg", "turn right")), (0.15, -0.15, 0.5))
        request = json.loads(ws.send.call_args.args[0])
        self.assertEqual(request["action"], "next")
        self.assertEqual(request["data"]["instruction"], "turn right")
        self.assertEqual(base64.b64decode(request["data"]["image"]), b"jpeg")

    def test_stop(self):
        self.assertIsNone(predict(self.next_reply(stop=True), 3, b"jpeg", "arrive"))

    def test_invalid_responses(self):
        for change in ({"rc": 500}, {"seq": 2}, {"stop": "false"}, {"actions": {}},
                       {"actions": {"actions": [[1, 2, float("nan")]]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                predict(self.next_reply(**change), 3, b"jpeg", "turn")
        with self.assertRaises(ValueError):
            exchange(self.next_reply(), "login", {})
