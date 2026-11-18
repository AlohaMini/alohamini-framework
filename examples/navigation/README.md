# Zero-shot VLN with LightNav-0

Use a [LightNav-0 navigation server](https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md)
started with `--task vln` in its own GPU environment. The adapter runs with the
[AlohaMini ROS 2 bridge](../../docs/ros2.md) using the ROS system Python and
the `python3-websocket` (websocket-client) package. The forward compressed
image topic must be publishing, with image timestamps in the ROS clock domain.

Preview one instruction with the command bridge disabled:

```bash
python3 examples/navigation/lightnav_vln.py --server ws://127.0.0.1:8051 \
  --instruction "Go down the corridor and stop at the door on the left."
ros2 topic echo /alohamini/lightnav/preview_cmd_vel  # in a second ROS terminal
```

The client opens one session per instruction, sends at most four images per
second, and uses the first robot-local waypoint. Forward and left map to
`Twist.linear.x/y` in m/s; counterclockwise yaw maps to `angular.z` in rad/s.
LightNav's reference normalization is scaled to 0.15 m/s per translation axis
and 0.5 rad/s in yaw. A stale frame yields zero velocity. Goal arrival ends the
episode. Socket and protocol failures terminate it; the bridge also enforces
its independent velocity watchdog.

For supervised hardware testing after preview, enable the bridge's velocity
command channel as described in the ROS guide and add `--drive` to the
client invocation. Only the base velocity topic is published. This example
does not provide odometry-based MPC or an obstacle detector.

Run the wire protocol and conversion tests without ROS or a model:

```bash
python3 -m unittest discover -s examples/navigation
```
