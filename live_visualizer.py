#!/usr/bin/env python3
"""
Live web-based scene graph visualizer — subscribes to ROS2 topics and
serves a real-time 2D view in the browser. No RViz needed.

Shows zones (colored polygons), objects (dots), robot pose (arrow),
and agent debug log (tool calls, responses) in real time.

Usage:
    python3 live_visualizer.py
    # Then open http://localhost:8080 in your browser
"""

import json
import math
import signal
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


# ---------------------------------------------------------------------------
# Shared state (updated by ROS2 callbacks, read by HTTP handler)
# ---------------------------------------------------------------------------

state = {
    "zones": [],
    "objects": [],
    "robot": {"x": 0, "y": 0, "yaw_deg": 0, "zone": None},
    "debug_log": [],       # last N debug messages
    "goal": None,          # current nav goal {x, y}
}
state_lock = threading.Lock()

MAX_LOG_ENTRIES = 50

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Scene Graph — Live</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; }
  #map-panel { flex: 1; display: flex; flex-direction: column; padding: 10px; }
  #log-panel { width: 380px; background: #16213e; border-left: 2px solid #0f3460;
               display: flex; flex-direction: column; }
  #log-header { padding: 10px 14px; font-weight: bold; font-size: 14px;
                background: #0f3460; color: #e94560; }
  #log-content { flex: 1; overflow-y: auto; padding: 8px; font-size: 12px;
                 font-family: 'Courier New', monospace; }
  .log-entry { padding: 4px 6px; border-bottom: 1px solid #1a1a3e; white-space: pre-wrap; word-break: break-all; }
  .log-tool { color: #00d2ff; }
  .log-speech { color: #4caf50; }
  .log-voice { color: #ff9800; }
  .log-pose { color: #9c27b0; }
  .log-init { color: #aaa; }
  .log-error { color: #f44336; }
  canvas { flex: 1; border-radius: 8px; }
  #status { padding: 6px 10px; font-size: 12px; color: #888; text-align: center; }
  #info { display: flex; gap: 15px; padding: 6px 10px; font-size: 12px; color: #aaa; }
</style>
</head>
<body>
<div id="map-panel">
  <canvas id="c"></canvas>
  <div id="info">
    <span id="robot-info">Robot: waiting...</span>
    <span id="objects-info">Objects: 0</span>
    <span id="zones-info">Zones: 0</span>
  </div>
  <div id="status">Connecting...</div>
</div>
<div id="log-panel">
  <div id="log-header">Agent Debug Log</div>
  <div id="log-content" id="log"></div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const logDiv = document.getElementById('log-content');
const statusDiv = document.getElementById('status');
let lastLogLen = 0;

const ZONE_COLORS = {
  'kitchen': {fill: 'rgba(76,175,80,0.15)', stroke: '#4CAF50'},
  'living_room': {fill: 'rgba(33,150,243,0.15)', stroke: '#2196F3'},
  'hallway': {fill: 'rgba(255,152,0,0.15)', stroke: '#FF9800'},
};
const DEFAULT_ZONE_COLOR = {fill: 'rgba(150,150,150,0.15)', stroke: '#999'};

function resize() {
  canvas.width = canvas.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.clientHeight * window.devicePixelRatio;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
}
window.addEventListener('resize', resize);
resize();

function worldToScreen(x, y, bounds, w, h) {
  const margin = 40;
  const scaleX = (w - 2*margin) / (bounds.maxX - bounds.minX || 1);
  const scaleY = (h - 2*margin) / (bounds.maxY - bounds.minY || 1);
  const scale = Math.min(scaleX, scaleY);
  const cx = w/2, cy = h/2;
  const worldCx = (bounds.minX + bounds.maxX)/2;
  const worldCy = (bounds.minY + bounds.maxY)/2;
  return {
    x: cx + (x - worldCx) * scale,
    y: cy - (y - worldCy) * scale,  // flip Y
    scale: scale
  };
}

function getBounds(data) {
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const z of data.zones) {
    for (const v of (z.vertices_world||[])) {
      minX = Math.min(minX, v[0]); minY = Math.min(minY, v[1]);
      maxX = Math.max(maxX, v[0]); maxY = Math.max(maxY, v[1]);
    }
  }
  for (const o of data.objects) {
    const p = o.map_position || {};
    minX = Math.min(minX, p.x||0); minY = Math.min(minY, p.y||0);
    maxX = Math.max(maxX, p.x||0); maxY = Math.max(maxY, p.y||0);
  }
  if (!isFinite(minX)) { minX=-1; minY=-1; maxX=11; maxY=6; }
  const pad = 1.5;
  return {minX: minX-pad, minY: minY-pad, maxX: maxX+pad, maxY: maxY+pad};
}

function draw(data) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const bounds = getBounds(data);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let gx = Math.floor(bounds.minX); gx <= Math.ceil(bounds.maxX); gx++) {
    const p = worldToScreen(gx, 0, bounds, w, h);
    ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, h); ctx.stroke();
  }
  for (let gy = Math.floor(bounds.minY); gy <= Math.ceil(bounds.maxY); gy++) {
    const p = worldToScreen(0, gy, bounds, w, h);
    ctx.beginPath(); ctx.moveTo(0, p.y); ctx.lineTo(w, p.y); ctx.stroke();
  }

  // Zones
  for (const z of data.zones) {
    const verts = z.vertices_world || [];
    if (verts.length < 3) continue;
    const zc = ZONE_COLORS[z.name] || DEFAULT_ZONE_COLOR;

    ctx.beginPath();
    const p0 = worldToScreen(verts[0][0], verts[0][1], bounds, w, h);
    ctx.moveTo(p0.x, p0.y);
    for (let i=1; i<verts.length; i++) {
      const p = worldToScreen(verts[i][0], verts[i][1], bounds, w, h);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = zc.fill;
    ctx.fill();
    ctx.strokeStyle = zc.stroke;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Zone label
    let cx=0, cy=0;
    for (const v of verts) { cx += v[0]; cy += v[1]; }
    cx /= verts.length; cy /= verts.length;
    const pc = worldToScreen(cx, cy, bounds, w, h);
    ctx.fillStyle = zc.stroke;
    ctx.globalAlpha = 0.5;
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(z.display_name || z.name, pc.x, pc.y);
    ctx.globalAlpha = 1.0;
  }

  // Nav goal
  if (data.goal) {
    const pg = worldToScreen(data.goal.x, data.goal.y, bounds, w, h);
    ctx.beginPath();
    ctx.arc(pg.x, pg.y, 12, 0, Math.PI*2);
    ctx.strokeStyle = '#f44336';
    ctx.lineWidth = 2;
    ctx.setLineDash([4,4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(244,67,54,0.3)';
    ctx.fill();
    // Crosshair
    ctx.beginPath();
    ctx.moveTo(pg.x-8, pg.y); ctx.lineTo(pg.x+8, pg.y);
    ctx.moveTo(pg.x, pg.y-8); ctx.lineTo(pg.x, pg.y+8);
    ctx.strokeStyle = '#f44336'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  // Objects
  for (const o of data.objects) {
    const p = o.map_position || {};
    const ps = worldToScreen(p.x||0, p.y||0, bounds, w, h);

    ctx.beginPath();
    ctx.arc(ps.x, ps.y, 8, 0, Math.PI*2);
    ctx.fillStyle = '#E91E63';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(o.label, ps.x, ps.y - 14);
  }

  // Robot
  const r = data.robot;
  if (r) {
    const pr = worldToScreen(r.x, r.y, bounds, w, h);
    const yaw = -(r.yaw_deg||0) * Math.PI / 180;  // negate for screen coords

    // Robot body
    ctx.beginPath();
    ctx.arc(pr.x, pr.y, 10, 0, Math.PI*2);
    ctx.fillStyle = '#9C27B0';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Heading arrow
    const arrowLen = 25;
    const ax = pr.x + arrowLen * Math.cos(yaw);
    const ay = pr.y + arrowLen * Math.sin(yaw);
    ctx.beginPath();
    ctx.moveTo(pr.x, pr.y);
    ctx.lineTo(ax, ay);
    ctx.strokeStyle = '#CE93D8';
    ctx.lineWidth = 3;
    ctx.stroke();
    // Arrowhead
    const headLen = 8;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen*Math.cos(yaw-0.4), ay - headLen*Math.sin(yaw-0.4));
    ctx.lineTo(ax - headLen*Math.cos(yaw+0.4), ay - headLen*Math.sin(yaw+0.4));
    ctx.closePath();
    ctx.fillStyle = '#CE93D8';
    ctx.fill();

    // Label
    ctx.fillStyle = '#CE93D8';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('ROBOT', pr.x, pr.y + 22);
  }

  // Info bar
  document.getElementById('robot-info').textContent =
    `Robot: (${r.x.toFixed(1)}, ${r.y.toFixed(1)}) yaw=${(r.yaw_deg||0).toFixed(0)} zone=${r.zone||'?'}`;
  document.getElementById('objects-info').textContent = `Objects: ${data.objects.length}`;
  document.getElementById('zones-info').textContent = `Zones: ${data.zones.length}`;
}

function getLogClass(text) {
  if (text.includes('[tool]')) return 'log-tool';
  if (text.includes('[speech]')) return 'log-speech';
  if (text.includes('[voice]')) return 'log-voice';
  if (text.includes('[pose]')) return 'log-pose';
  if (text.includes('[init]')) return 'log-init';
  if (text.includes('[error]')) return 'log-error';
  return '';
}

function updateLog(entries) {
  if (entries.length === lastLogLen) return;
  // Only add new entries
  const newEntries = entries.slice(lastLogLen);
  for (const e of newEntries) {
    const div = document.createElement('div');
    div.className = 'log-entry ' + getLogClass(e);
    div.textContent = e;
    logDiv.appendChild(div);
  }
  lastLogLen = entries.length;
  logDiv.scrollTop = logDiv.scrollHeight;
}

async function poll() {
  try {
    const resp = await fetch('/state');
    const data = await resp.json();
    draw(data);
    updateLog(data.debug_log || []);
    statusDiv.textContent = `Connected — ${new Date().toLocaleTimeString()}`;
    statusDiv.style.color = '#4caf50';
  } catch(e) {
    statusDiv.textContent = 'Disconnected';
    statusDiv.style.color = '#f44336';
  }
}

setInterval(poll, 200);  // 5 Hz update
poll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/state":
            with state_lock:
                data = json.dumps(state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def log_message(self, format, *args):
        pass  # suppress HTTP logs


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class VisualizerNode(Node):
    def __init__(self, frame_id, base_frame):
        super().__init__("live_visualizer")
        self._frame_id = frame_id
        self._base_frame = base_frame

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(String, "/scene_graph/objects", self._on_objects, latched_qos)
        self.create_subscription(String, "/scene_graph/zones", self._on_zones, latched_qos)
        self.create_subscription(String, "/scene_graph/agent_debug", self._on_debug, 10)

        # Subscribe to goal_pose to show nav target
        from geometry_msgs.msg import PoseStamped
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)

        # Poll TF for robot pose
        self.create_timer(0.1, self._update_robot_pose)

        self.get_logger().info("Live visualizer node started")

    def _on_objects(self, msg):
        try:
            data = json.loads(msg.data)
            with state_lock:
                state["objects"] = data.get("objects", [])
        except json.JSONDecodeError:
            pass

    def _on_zones(self, msg):
        try:
            data = json.loads(msg.data)
            with state_lock:
                state["zones"] = data.get("zones", [])
        except json.JSONDecodeError:
            pass

    def _on_debug(self, msg):
        with state_lock:
            state["debug_log"].append(msg.data)
            if len(state["debug_log"]) > MAX_LOG_ENTRIES:
                state["debug_log"] = state["debug_log"][-MAX_LOG_ENTRIES:]

    def _on_goal(self, msg):
        with state_lock:
            state["goal"] = {
                "x": msg.pose.position.x,
                "y": msg.pose.position.y,
            }

    def _update_robot_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._frame_id, self._base_frame, rclpy.time.Time()
            )
            pos = t.transform.translation
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

            # Determine zone
            current_zone = None
            with state_lock:
                for zone in state["zones"]:
                    verts = zone.get("vertices_world", [])
                    if len(verts) >= 3 and point_in_polygon(pos.x, pos.y, verts):
                        current_zone = zone["name"]
                        break
                state["robot"] = {
                    "x": round(pos.x, 3),
                    "y": round(pos.y, 3),
                    "yaw_deg": round(yaw_deg, 1),
                    "zone": current_zone,
                }
        except (LookupException, ExtrapolationException):
            pass


def point_in_polygon(x, y, vertices):
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live web-based scene graph visualizer")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--base-frame", default="base_link")
    args = parser.parse_args()

    # Start HTTP server in background thread
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    print(f"Live visualizer: http://localhost:{args.port}")

    # Start ROS2 node
    rclpy.init()
    node = VisualizerNode(args.frame_id, args.base_frame)

    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
