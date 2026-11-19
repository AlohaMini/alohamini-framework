# Zero-shot VLN with LightNav-0

Use a [LightNav-0 navigation server](https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md)
started with `--task vln` in its own GPU environment. The adapter runs with the
[AlohaMini ROS 2 bridge](../../docs/ros2.md). From the checkout root, load the
repository's ROS Conda environment and install the WebSocket client in that
environment:

```bash
source ros2/env.bash
python -m pip install websocket-client
```

For a ROS environment installed in a custom prefix, export
`ALOHAMINI_ROS_ENV=/absolute/path/to/environment` before sourcing the script.
The forward compressed image topic must be publishing, with image timestamps
in the ROS clock domain.

Preview one instruction with the command bridge disabled:

```bash
python3 examples/navigation/lightnav_vln.py --server ws://127.0.0.1:8051 \
  --instruction "Go down the corridor and stop at the door on the left."
```

In a second terminal with the same ROS environment loaded, run
`ros2 topic echo /alohamini/lightnav/preview_cmd_vel` to observe the output.

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

### Workstation dual-camera trial

The optional jjy launcher opens one LightNav WebSocket session per camera on a
shared model server. Each session receives its own chronological JPEG sequence
and the same instruction. Only pairs with capture times within 80 ms enter a
prediction. The oldest image must still be under 450 ms old when the reply is
used. This requires aligned Pi and workstation clocks.

The forward session supplies the robot-local waypoint. A stop from either view
ends the trial. If both predict at least 3 cm of translation and their headings
differ by more than 60 degrees, or their turns have opposite signs and both
exceed 0.1 rad in magnitude (including equality), the client holds zero velocity
for that pair. These thresholds are experimental. A chest stop or disagreement
can indicate a viewpoint mismatch. Neither is a distance measurement.

With the Pi Host publishing both cameras (`--cameras forward chest`), the jjy
workstation launcher accepts `--dual-camera --preview` for read-only evaluation
and `--dual-camera` for an attended trial. Single-camera operation is the default.
