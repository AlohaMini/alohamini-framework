# Zero-shot VLN with LightNav-0

Use a [LightNav-0 navigation server](https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md)
started with `--task vln` in its own GPU environment. The adapter runs with the
[AlohaMini ROS 2 bridge](../../docs/ros2.md). From the checkout root, load the
repository's ROS Conda environment and install the WebSocket client in that
environment:

```bash
source ros2/env.bash
python -m pip install websocket-client onnxruntime opencv-python-headless
```

Set `ALOHAMINI_FLOOR_MODEL` to a local copy of
[Xenova/segformer-b0-finetuned-ade-512-512](https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/tree/d3e5499fa8701ff0453ca940a8dfeae39b2f1504)
containing `config.json`, `preprocessor_config.json`, and `onnx/model.onnx`.
The site default is `~/workspace/jjy/models/segformer-b0-ade`.

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
client invocation. Only the base velocity topic is published. The forward RGB guard estimates floor coverage and local contrast in a trapezoid
in the lower center of the 640 x 480 frame. On obstruction the adapter checks the lower center and two side regions. With a
visible side route it holds a turn direction and scales requested forward speed
to 25% (at most 0.045 m/s) for up to one second per encounter. Close obstacles
suppress translation. The avoidance yaw target is at least 0.32 rad/s and scales
the requested magnitude by 1.25, within the configured yaw cap. Without a
visible side route all axes hold at zero. Releasing translation takes at least
three fresh clear frames spanning 0.5 seconds. Unknown image geometry or poor
visibility holds all axes. The region and thresholds depend on camera mounting
and floor appearance. This estimate has no depth or swept-footprint measurement.

Run the wire protocol and conversion tests without ROS or a model:

```bash
python3 -m unittest discover -s examples/navigation
```
