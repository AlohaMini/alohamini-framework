import { describe, el, fmt, iconButton, t } from "./ui.js";

export const colors = ["#4d61ee", "#e15128", "#0a9c83", "#bd3d8f", "#9371ce", "#147fad"];
export const accents = ["#4d61ee", "#e06f39", "#00a58c", "#bd4595", "#3879d7"];
const padding = { left: 59, right: 12, top: 10, bottom: 24 };

function nearest(points, x) {
  let lo = 0,
    hi = points.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (points[mid][0] < x) lo = mid + 1;
    else hi = mid;
  }
  if (!lo) return 0;
  if (lo === points.length) return lo - 1;
  return x - points[lo - 1][0] < points[lo][0] - x ? lo - 1 : lo;
}

function ticks(low, high, intervals, integer = false) {
  const target = (high - low) / intervals;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const factor = [1, 2, 2.5, 5, 10].find((n) => n * magnitude >= target) || 10;
  const step = integer ? Math.max(1, Math.ceil(factor * magnitude)) : factor * magnitude;
  const first = Math.ceil(low / step - 1e-10),
    last = Math.floor(high / step + 1e-10);
  const result = [];
  for (let i = 0; i <= last - first && i < 20; i++) result.push((first + i) * step);
  return result;
}

// Canvas resolution follows devicePixelRatio; DOM overlays keep hover independent of repaint.
export class Plot {
  constructor(container, onZoom, onExpand) {
    this.onZoom = onZoom;
    this.card = el("article", "card chart-card");
    this.heading = el("div", "chart-heading");
    this.title = el("h3");
    this.meta = el("span", "chart-meta");
    this.heading.append(this.title, this.meta);
    if (onExpand) {
      this.expand = iconButton("expand", t("expand"), onExpand);
      this.expand.classList.add("expand-plot");
      this.heading.append(this.expand);
    }
    this.area = el("div", "plot");
    this.canvas = el("canvas");
    this.canvas.tabIndex = 0;
    this.cross = el("div", "crosshair");
    this.tip = el("div", "plot-tip");
    this.tip.setAttribute("role", "status");
    this.brush = el("div", "brush");
    this.brush.hidden = true;
    this.area.append(this.canvas, this.cross, this.tip, this.brush);
    this.legend = el("div", "legend");
    this.card.append(this.heading, this.area, this.legend);
    container.append(this.card);
    this.lines = [];
    this.hiddenLines = new Set();
    this.hide();
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.area);
    this.canvas.onpointermove = (e) => {
      if (!this.domain) return;
      const x = this.pixel(e);
      if (this.start != null) {
        this.brush.hidden = false;
        this.brush.style.left = Math.min(x, this.start) + "px";
        this.brush.style.width = Math.abs(x - this.start) + "px";
      } else this.inspect(this.toValue(x));
    };
    this.canvas.onpointerleave = () => {
      if (this.start == null) this.hide();
    };
    this.canvas.onpointerdown = (e) => {
      if (!this.domain || !this.onZoom || e.button !== 0) return;
      this.hide();
      this.start = this.pixel(e);
      this.canvas.setPointerCapture(e.pointerId);
    };
    this.canvas.onpointerup = (e) => {
      if (this.start == null) return;
      const from = this.start,
        to = this.pixel(e);
      this.start = null;
      this.brush.hidden = true;
      if (Math.abs(to - from) > 8)
        this.onZoom(this.toValue(Math.min(from, to)), this.toValue(Math.max(from, to)));
      else this.inspect(this.toValue(to));
    };
    this.canvas.onpointercancel = () => {
      this.start = null;
      this.brush.hidden = true;
    };
    this.canvas.onblur = () => this.hide();
    this.canvas.onkeydown = (e) => {
      if (e.key === "Escape") return this.hide();
      if (!this.domain || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const pts = this.lines.find((line) => line.points.length)?.points;
      if (!pts) return;
      let i = this.hoverX == null ? pts.length - 1 : nearest(pts, this.hoverX);
      if (e.key === "Home") i = 0;
      else if (e.key === "End") i = pts.length - 1;
      else i = Math.max(0, Math.min(pts.length - 1, i + (e.key === "ArrowLeft" ? -1 : 1)));
      this.inspect(pts[i][0]);
    };
  }
  update({
    title,
    description,
    axis,
    lines,
    featured = false,
    accent = colors[0],
    range,
    unit,
    zeroY = false,
  }) {
    this.axis = axis;
    this.zeroY = zeroY;
    const visibleLines = () =>
      lines.length < 2 ? lines : lines.filter((line) => !this.hiddenLines.has(line.id));
    this.lines = visibleLines();
    this.range = range;
    this.title.textContent = title;
    this.card.classList.toggle("featured", featured);
    this.card.style.setProperty("--accent", accent);
    describe(this.title, description);
    if (this.expand) {
      this.expand.title = t("expand") + " · " + title;
      this.expand.setAttribute("aria-label", this.expand.title);
    }
    this.canvas.setAttribute(
      "aria-label",
      title + " · " + t(axis) + (this.onZoom ? " · " + t("selectHint") : ""),
    );
    let count = 0,
      visible = 0;
    this.legend.replaceChildren();
    for (const line of lines) {
      if (!line.trend) {
        count += line.count;
        visible += line.points.length;
      }
      const item = el(lines.length > 1 ? "button" : "span", "legend-item");
      if (lines.length > 1) {
        item.type = "button";
        item.setAttribute("aria-pressed", String(!this.hiddenLines.has(line.id)));
        item.title = t("toggleSeries") + " · " + line.label;
        item.onclick = () => {
          if (this.hiddenLines.has(line.id)) this.hiddenLines.delete(line.id);
          else this.hiddenLines.add(line.id);
          item.setAttribute("aria-pressed", String(!this.hiddenLines.has(line.id)));
          this.lines = visibleLines();
          this.hide();
          this.draw();
        };
      }
      const swatch = el("i", "swatch" + (line.dash ? " dash" : ""));
      swatch.style.setProperty("--ink", line.color);
      item.append(swatch, document.createTextNode(line.label));
      this.legend.append(item);
    }
    this.legend.append(el("span", "plot-axis", t(axis) + (unit ? " · " + unit : "")));
    this.meta.textContent = count
      ? visible < count
        ? t("sampled") + " · " + fmt(visible) + " / " + fmt(count)
        : t("raw") + " · " + fmt(count)
      : "";
    this.draw();
  }
  pixel(e) {
    const rect = this.canvas.getBoundingClientRect();
    return Math.min(rect.width - padding.right, Math.max(padding.left, e.clientX - rect.left));
  }
  toValue(px) {
    const d = this.domain;
    return d.xmin + ((px - padding.left) * (d.xmax - d.xmin)) / d.width;
  }
  draw() {
    const { width: w, height: h } = this.area.getBoundingClientRect();
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    const canvas = this.canvas;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    let xmin = Infinity,
      xmax = -Infinity,
      ymin = Infinity,
      ymax = -Infinity;
    for (const { points } of this.lines)
      for (const [x, y] of points) {
        xmin = Math.min(xmin, x);
        xmax = Math.max(xmax, x);
        if (y != null) {
          ymin = Math.min(ymin, y);
          ymax = Math.max(ymax, y);
        }
      }
    ctx.font = '12px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = "#64718a";
    if (!Number.isFinite(ymin)) {
      this.domain = null;
      this.hide();
      ctx.textAlign = "center";
      ctx.fillText(t("noData"), w / 2, h / 2);
      return;
    }
    if (this.range) [xmin, xmax] = this.range;
    if (xmax === xmin) {
      xmin -= 0.5;
      xmax += 0.5;
    }
    if (this.zeroY) {
      ymin = Math.min(0, ymin);
      ymax = Math.max(0, ymax);
    }
    const margin = ymin === ymax ? Math.abs(ymin) * 0.08 || 0.5 : (ymax - ymin) * 0.07;
    const floor = ymin >= 0 ? 0 : -Infinity;
    ymin = Math.max(floor, ymin - margin);
    ymax += margin;
    const width = w - padding.left - padding.right,
      height = h - padding.top - padding.bottom;
    this.domain = { xmin, xmax, width };
    const X = (x) => padding.left + ((x - xmin) * width) / (xmax - xmin);
    const Y = (y) => padding.top + ((ymax - y) * height) / (ymax - ymin);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#e2e7f2";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const value of ticks(ymin, ymax, 5)) {
      const y = Y(value);
      ctx.beginPath();
      ctx.setLineDash([3, 4]);
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      ctx.fillText(fmt(value, true), padding.left - 7, y, padding.left - 9);
    }
    const intervals = Math.max(2, Math.min(8, Math.floor(width / 96)));
    for (const x of ticks(xmin, xmax, intervals, this.axis !== "elapsed_s")) {
      ctx.textAlign =
        X(x) < padding.left + 25 ? "left" : X(x) > w - padding.right - 25 ? "right" : "center";
      ctx.fillText(fmt(x, true), X(x), h - 9);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(padding.left - 3, padding.top - 3, width + 6, height + 6);
    ctx.clip();
    for (const line of this.lines) {
      ctx.beginPath();
      ctx.strokeStyle = line.color;
      ctx.fillStyle = line.color;
      ctx.lineWidth = line.trend ? 2.6 : line.dash ? 2.2 : line.faint ? 1 : 1.7;
      ctx.lineJoin = "round";
      ctx.setLineDash(line.dash ? [6, 4] : []);
      let move = true;
      for (const [x, y] of line.points) {
        if (y == null) {
          move = true;
          continue;
        }
        if (move) ctx.moveTo(X(x), Y(y));
        else ctx.lineTo(X(x), Y(y));
        move = false;
      }
      ctx.stroke();
      if (line.points.length <= 24) {
        for (const [x, y] of line.points)
          if (y != null) {
            ctx.beginPath();
            ctx.arc(X(x), Y(y), 2.8, 0, Math.PI * 2);
            ctx.fill();
          }
      }
    }
    ctx.restore();
    if (this.hoverX != null) this.inspect(this.hoverX);
  }
  inspect(x) {
    if (!this.domain) return;
    const lines = this.lines.filter((line) => line.points.length);
    const near = lines.map((line) => line.points[nearest(line.points, x)]);
    if (!near.length) return;
    const at = near.reduce((a, b) => (Math.abs(a[0] - x) <= Math.abs(b[0] - x) ? a : b))[0];
    const d = this.domain,
      px = padding.left + ((at - d.xmin) * d.width) / (d.xmax - d.xmin);
    this.hoverX = at;
    this.cross.style.left = px + "px";
    this.cross.hidden = false;
    const labels = [t(this.axis) + "  " + fmt(at)];
    near.forEach(([x, y], i) => {
      if (x === at) labels.push(lines[i].label + "   " + fmt(y));
    });
    this.tip.textContent = labels.join("\n");
    this.tip.hidden = false;
    const available = this.area.clientWidth - this.tip.offsetWidth - 4;
    this.tip.style.left = Math.max(2, Math.min(available, px + 12)) + "px";
    this.tip.style.top = "8px";
  }
  hide() {
    this.cross.hidden = true;
    this.tip.hidden = true;
    this.hoverX = null;
  }
  destroy() {
    this.observer.disconnect();
    this.card.remove();
  }
}
