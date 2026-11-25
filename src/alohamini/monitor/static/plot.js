import { describe, el, fmt, iconButton, onThemeChange, t, theme } from "./ui.js";

// Vibrant, high-contrast palettes tuned for both light & dark backgrounds
const lightColors = ["#2563eb", "#ea580c", "#059669", "#c026d3", "#7c3aed", "#0891b2"];
const darkColors = ["#60a5fa", "#fb923c", "#34d399", "#e879f9", "#a78bfa", "#22d3ee"];

export const colors = lightColors;
export const accents = lightColors;
export function paletteColor(index) {
  const list = theme === "dark" ? darkColors : lightColors;
  return list[((index % list.length) + list.length) % list.length];
}

const padding = { left: 60, right: 14, top: 14, bottom: 24 };

// Global cross-chart synchronized hover hub
const activePlots = new Set();
export function broadcastHover(source, axis, x) {
  for (const plot of activePlots) {
    if (plot !== source && plot.syncGroup && plot.syncGroup === source.syncGroup && plot.axis === axis) {
      if (x == null) plot.hide(false);
      else plot.inspect(x, false);
    }
  }
}

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
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return [];
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

function logTicks(minVal, maxVal) {
  const lo = Math.floor(Math.log10(minVal));
  const hi = Math.ceil(Math.log10(maxVal));
  const out = [];
  for (let e = lo; e <= hi; e++) {
    const v = 10 ** e;
    if (v >= minVal * 0.999 && v <= maxVal * 1.001) out.push(v);
    if (hi - lo <= 2) {
      for (const m of [2, 5]) {
        const vm = m * v;
        if (vm >= minVal && vm <= maxVal) out.push(vm);
      }
    }
  }
  return out.sort((a, b) => a - b);
}

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function withAlpha(hex, alpha) {
  if (!hex || !hex.startsWith("#") || hex.length !== 7) return hex;
  const r = parseInt(hex.slice(1, 3), 16),
    g = parseInt(hex.slice(3, 5), 16),
    b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// High-DPI Canvas 2D line chart with log/robust scaling, reference lines, events & synced crosshairs.
export class Plot {
  constructor(container, onZoom, onExpand, { syncGroup = null, onReset = null } = {}) {
    this.onZoom = onZoom;
    this.onReset = onReset;
    this.syncGroup = syncGroup;
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
    this.allLines = [];
    this.bands = [];
    this.events = [];
    this.refs = [];
    this.hiddenLines = new Set();
    this.highlightX = null;
    this.hide(false);
    activePlots.add(this);
    this.unsubTheme = onThemeChange(() => this.draw());
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.area);

    this.canvas.onpointermove = (e) => {
      if (!this.domain) return;
      const x = this.pixel(e);
      if (this.start != null) {
        this.brush.hidden = false;
        this.brush.style.left = Math.min(x, this.start) + "px";
        this.brush.style.width = Math.abs(x - this.start) + "px";
      } else this.inspect(this.toValue(x), true);
    };
    this.canvas.onpointerleave = () => {
      if (this.start == null) this.hide(true);
    };
    this.canvas.onpointerdown = (e) => {
      if (!this.domain || !this.onZoom || e.button !== 0) return;
      this.hide(true);
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
      else this.inspect(this.toValue(to), true);
    };
    this.canvas.onpointercancel = () => {
      this.start = null;
      this.brush.hidden = true;
    };
    this.canvas.ondblclick = () => {
      if (this.onReset) this.onReset();
    };
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        if (!this.domain || !this.onZoom || (!e.ctrlKey && !e.shiftKey)) return;
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1.4 : 0.7;
        const center = this.toValue(this.pixel(e));
        const halfLeft = (center - this.domain.xmin) * factor;
        const halfRight = (this.domain.xmax - center) * factor;
        this.onZoom(center - halfLeft, center + halfRight);
      },
      { passive: false },
    );
    this.canvas.onblur = () => this.hide(true);
    this.canvas.onkeydown = (e) => {
      if (e.key === "Escape") return this.hide(true);
      if (!this.domain || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const pts = this.lines.find((line) => line.points.length)?.points;
      if (!pts) return;
      let i = this.hoverX == null ? pts.length - 1 : nearest(pts, this.hoverX);
      if (e.key === "Home") i = 0;
      else if (e.key === "End") i = pts.length - 1;
      else i = Math.max(0, Math.min(pts.length - 1, i + (e.key === "ArrowLeft" ? -1 : 1)));
      this.inspect(pts[i][0], true);
    };
  }

  update({
    title,
    description,
    axis,
    lines,
    bands = [],
    events = [],
    refs = [],
    robustBounds = null,
    accent = paletteColor(0),
    range,
    unit,
    yScale = "fit",
    zeroY = false,
  }) {
    this.axis = axis;
    this.yScale = zeroY ? "zero" : yScale;
    this.allLines = lines;
    this.bands = bands;
    this.events = events;
    this.refs = refs;
    this.robustBounds = robustBounds;
    const visibleLines = () =>
      lines.length < 2 ? lines : lines.filter((line) => !this.hiddenLines.has(line.id));
    this.lines = visibleLines();
    this.range = range;
    this.title.textContent = title;
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
        count += line.count || line.points.length;
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
          this.hide(false);
          this.draw();
        };
      }
      const swatch = el("i", "swatch" + (line.dash ? " dash" : ""));
      swatch.style.setProperty("--ink", line.displayColor || line.color);
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

  flashStep(step) {
    this.highlightX = step;
    this.draw();
    this.card.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => {
      if (this.highlightX === step) {
        this.highlightX = null;
        this.draw();
      }
    }, 3200);
  }

  pixel(e) {
    const rect = this.canvas.getBoundingClientRect();
    return Math.min(rect.width - padding.right, Math.max(padding.left, e.clientX - rect.left));
  }

  toValue(px) {
    const d = this.domain;
    return d.xmin + ((px - padding.left) * (d.xmax - d.xmin)) / d.width;
  }

  toPngDataUrl() {
    const { width: w, height: h } = this.area.getBoundingClientRect();
    const out = document.createElement("canvas");
    const dpr = 2;
    out.width = Math.round(w * dpr);
    out.height = Math.round((h + 36) * dpr);
    const ctx = out.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = cssVar("--panel", "#ffffff");
    ctx.fillRect(0, 0, w, h + 36);
    ctx.fillStyle = cssVar("--ink", "#181c24");
    ctx.font = '500 13px "Inter", "Ubuntu Sans", "Noto Sans CJK SC", sans-serif';
    ctx.fillText(this.title.textContent || "", 14, 22);
    ctx.drawImage(this.canvas, 0, 32, w, h);
    return out.toDataURL("image/png");
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

    const gridColor = cssVar("--grid", "#e8ebf0");
    const axisInk = cssVar("--muted", "#5a6270");
    const blueInk = cssVar("--blue", "#2563eb");
    const warnInk = cssVar("--warn", "#b45309");
    const okInk = cssVar("--ok", "#15803d");
    const errInk = cssVar("--err", "#b91c1c");
    const isLog = this.yScale === "log";

    let xmin = Infinity,
      xmax = -Infinity,
      ymin = Infinity,
      ymax = -Infinity;
    for (const line of this.lines) {
      for (const [x, y] of line.points) {
        xmin = Math.min(xmin, x);
        xmax = Math.max(xmax, x);
        if (y != null && Number.isFinite(y) && (!isLog || y > 0)) {
          ymin = Math.min(ymin, y);
          ymax = Math.max(ymax, y);
        }
      }
    }
    for (const band of this.bands) {
      for (const [x, lo, hi] of band.points) {
        xmin = Math.min(xmin, x);
        xmax = Math.max(xmax, x);
        if (Number.isFinite(lo) && (!isLog || lo > 0)) ymin = Math.min(ymin, lo);
        if (Number.isFinite(hi) && (!isLog || hi > 0)) ymax = Math.max(ymax, hi);
      }
    }

    ctx.font = '400 11px "Inter", "Ubuntu Sans", sans-serif';
    ctx.fillStyle = axisInk;
    if (!Number.isFinite(ymin)) {
      this.domain = null;
      this.hide(false);
      ctx.textAlign = "center";
      ctx.fillText(t("noData"), w / 2, h / 2);
      return;
    }

    if (this.range) [xmin, xmax] = this.range;
    if (xmax === xmin) {
      xmin -= 0.5;
      xmax += 0.5;
    }

    // Apply Y scaling strategy
    if (this.yScale === "robust" && this.robustBounds) {
      const [rlo, rhi] = this.robustBounds;
      if (Number.isFinite(rlo) && Number.isFinite(rhi) && rhi > rlo) {
        ymin = rlo;
        ymax = rhi;
      }
    }
    for (const ref of this.refs) {
      if (this.yScale !== "robust" && Number.isFinite(ref.value) && (!isLog || ref.value > 0)) {
        // Keep clip threshold visible if reasonably close to data scale
        if (ref.value <= ymax * 2.5 && ref.value >= ymin * 0.2) {
          ymin = Math.min(ymin, ref.value);
          ymax = Math.max(ymax, ref.value);
        }
      }
    }
    if (this.yScale === "zero" && !isLog) {
      ymin = Math.min(0, ymin);
      ymax = Math.max(0, ymax);
    }

    const width = w - padding.left - padding.right,
      height = h - padding.top - padding.bottom;
    let Y, yTicks;
    if (isLog) {
      const loLog = Math.log10(Math.max(ymin, 1e-12));
      const hiLog = Math.log10(Math.max(ymax, ymin * 1.1));
      const pad = Math.max(0.08, (hiLog - loLog) * 0.08);
      const yMinLog = loLog - pad,
        yMaxLog = hiLog + pad;
      Y = (y) => padding.top + ((yMaxLog - Math.log10(Math.max(y, 1e-12))) * height) / (yMaxLog - yMinLog);
      yTicks = logTicks(10 ** yMinLog, 10 ** yMaxLog);
    } else {
      const margin = ymin === ymax ? Math.abs(ymin) * 0.1 || 0.5 : (ymax - ymin) * 0.08;
      const floor = ymin >= 0 && this.yScale !== "robust" ? 0 : -Infinity;
      ymin = Math.max(floor, ymin - margin);
      ymax += margin;
      Y = (y) => padding.top + ((ymax - y) * height) / (ymax - ymin);
      yTicks = ticks(ymin, ymax, 5);
    }

    this.domain = { xmin, xmax, width, ymin, ymax, isLog };
    const X = (x) => padding.left + ((x - xmin) * width) / (xmax - xmin);

    // Horizontal & subtle vertical grid
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridColor;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const value of yTicks) {
      const y = Y(value);
      if (y < padding.top - 2 || y > h - padding.bottom + 2) continue;
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      ctx.fillStyle = axisInk;
      ctx.fillText(fmt(value, true), padding.left - 7, y, padding.left - 9);
    }

    const intervals = Math.max(2, Math.min(8, Math.floor(width / 96)));
    const xTicks = ticks(xmin, xmax, intervals, this.axis !== "elapsed_s");
    for (const x of xTicks) {
      const px = X(x);
      ctx.beginPath();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = withAlpha(gridColor, 0.7);
      ctx.moveTo(px, padding.top);
      ctx.lineTo(px, h - padding.bottom);
      ctx.stroke();
      ctx.fillStyle = axisInk;
      ctx.textAlign =
        px < padding.left + 24 ? "left" : px > w - padding.right - 24 ? "right" : "center";
      ctx.fillText(fmt(x, true), px, h - 9);
    }

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(padding.left - 2, padding.top - 2, width + 4, height + 4);
    ctx.clip();

    // 1. Vertical event markers (only when X axis is step)
    if (this.axis === "step" && this.events.length) {
      let lastLabelX = -100;
      for (const ev of this.events) {
        if (ev.step < xmin || ev.step > xmax) continue;
        const px = X(ev.step);
        const color =
          ev.type === "amp_overflow" || ev.type === "nonfinite"
            ? errInk
            : ev.type === "validation_best"
              ? okInk
              : ev.type === "resume"
                ? warnInk
                : blueInk;
        ctx.save();
        ctx.strokeStyle = withAlpha(color, ev.type === "checkpoint" ? 0.28 : 0.65);
        ctx.lineWidth = ev.type === "resume" || ev.type === "validation_best" ? 1.4 : 1;
        ctx.setLineDash(ev.type === "resume" ? [5, 3] : [2, 3]);
        ctx.beginPath();
        ctx.moveTo(px, padding.top);
        ctx.lineTo(px, h - padding.bottom);
        ctx.stroke();
        if (ev.type !== "checkpoint" && px - lastLabelX > 56) {
          const text = ev.label || t("ev_" + ev.type);
          ctx.font = '500 10.5px "Inter", "Ubuntu Sans", "Noto Sans CJK SC", sans-serif';
          const tw = ctx.measureText(text).width + 10;
          const bx = Math.min(w - padding.right - tw, Math.max(padding.left + 2, px + 3));
          ctx.fillStyle = withAlpha(color, 0.11);
          ctx.fillRect(bx, padding.top + 1, tw, 17);
          ctx.fillStyle = color;
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(text, bx + 5, padding.top + 9.5);
          lastLabelX = bx + tw;
        }
        ctx.restore();
      }
    }

    // 2. Highlighted jump step beacon
    if (this.highlightX != null && this.highlightX >= xmin && this.highlightX <= xmax) {
      const px = X(this.highlightX);
      ctx.save();
      ctx.strokeStyle = blueInk;
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(px, padding.top);
      ctx.lineTo(px, h - padding.bottom);
      ctx.stroke();
      ctx.restore();
    }

    // 3. Horizontal reference lines (e.g., grad_clip_norm or zero line)
    for (const ref of this.refs) {
      if (!Number.isFinite(ref.value) || (isLog && ref.value <= 0)) continue;
      const py = Y(ref.value);
      if (py < padding.top - 4 || py > h - padding.bottom + 4) continue;
      ctx.save();
      ctx.strokeStyle = ref.color || warnInk;
      ctx.lineWidth = 1.3;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(padding.left, py);
      ctx.lineTo(w - padding.right, py);
      ctx.stroke();
      if (ref.label) {
        ctx.font = '500 10.5px "Inter", "Ubuntu Sans", sans-serif';
        ctx.fillStyle = ref.color || warnInk;
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(ref.label, w - padding.right - 4, py - 3);
      }
      ctx.restore();
    }

    // 4. Shaded confidence bands (Multi-seed Mean ± Std)
    for (const band of this.bands) {
      if (band.points.length < 2) continue;
      ctx.save();
      ctx.fillStyle = withAlpha(band.color, 0.16);
      ctx.beginPath();
      let started = false;
      for (const [x, , hi] of band.points) {
        if (!Number.isFinite(hi) || (isLog && hi <= 0)) continue;
        if (!started) {
          ctx.moveTo(X(x), Y(hi));
          started = true;
        } else ctx.lineTo(X(x), Y(hi));
      }
      for (let i = band.points.length - 1; i >= 0; i--) {
        const [x, lo] = band.points[i];
        if (!Number.isFinite(lo) || (isLog && lo <= 0)) continue;
        ctx.lineTo(X(x), Y(lo));
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // 5. Data series (raw faint first, then crisp smoothed / validation lines on top)
    const ordered = [
      ...this.lines.filter((l) => l.faint),
      ...this.lines.filter((l) => !l.faint && !l.dash),
      ...this.lines.filter((l) => l.dash),
    ];
    for (const line of ordered) {
      ctx.save();
      ctx.beginPath();
      const stroke = line.faint ? withAlpha(line.color, theme === "dark" ? 0.2 : 0.16) : line.color;
      ctx.strokeStyle = stroke;
      ctx.fillStyle = line.color;
      ctx.lineWidth = line.trend ? 2.15 : line.dash ? 2.1 : line.faint ? 0.95 : 1.8;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.setLineDash(line.dash ? [6, 4] : []);
      let move = true;
      for (const [x, y] of line.points) {
        if (y == null || !Number.isFinite(y) || (isLog && y <= 0)) {
          move = true;
          continue;
        }
        const px = X(x),
          py = Y(y);
        if (move) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
        move = false;
      }
      ctx.stroke();

      // Sparse points (e.g. validation checkpoints) get crisp markers
      if (line.points.length <= 36 && !line.faint) {
        ctx.setLineDash([]);
        for (const [x, y] of line.points) {
          if (y == null || !Number.isFinite(y) || (isLog && y <= 0)) continue;
          const px = X(x),
            py = Y(y);
          ctx.beginPath();
          if (line.dash) {
            // Diamond marker for validation
            ctx.moveTo(px, py - 3.8);
            ctx.lineTo(px + 3.8, py);
            ctx.lineTo(px, py + 3.8);
            ctx.lineTo(px - 3.8, py);
            ctx.closePath();
            ctx.fill();
          } else {
            ctx.arc(px, py, 2.8, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.restore();
    }

    ctx.restore();
    if (this.hoverX != null) this.inspect(this.hoverX, false);
  }

  inspect(x, notify = true) {
    if (!this.domain) return;
    const lines = this.lines.filter((line) => line.points.length);
    if (!lines.length) return;
    const d = this.domain;
    const span = Math.max(1e-9, d.xmax - d.xmin);
    // Snap to closest point on primary non-faint line, or closest overall
    const primaryLines = lines.filter((l) => !l.faint);
    const anchorPool = primaryLines.length ? primaryLines : lines;
    const nearAnchors = anchorPool.map((line) => line.points[nearest(line.points, x)]);
    const at = nearAnchors.reduce((a, b) => (Math.abs(a[0] - x) <= Math.abs(b[0] - x) ? a : b))[0];
    const px = padding.left + ((at - d.xmin) * d.width) / span;
    if (px < padding.left - 4 || px > padding.left + d.width + 4) {
      this.hide(false);
      return;
    }
    this.hoverX = at;
    this.cross.style.left = px + "px";
    this.cross.hidden = false;

    // Show every visible series whose nearest point is within a sensible tolerance
    const tolerance = span * 0.04;
    this.tip.replaceChildren();
    this.tip.append(el("div", "plot-tip-head", `${t(this.axis)}  ${fmt(at)}`));
    const shown = [];
    for (const line of lines) {
      const pt = line.points[nearest(line.points, at)];
      if (!pt) continue;
      const dist = Math.abs(pt[0] - at);
      // Validation lines are sparse: show nearest validation point if within 12% span and annotate its step
      const maxTol = line.dash ? span * 0.14 : tolerance;
      if (dist > maxTol) continue;
      const row = el("div", "plot-tip-row");
      const name = el("span", "plot-tip-name");
      const dot = el("i", "plot-tip-dot");
      dot.style.background = line.displayColor || line.color;
      const suffix = dist > 0 && line.dash ? ` (@${fmt(pt[0], true)})` : "";
      name.append(dot, document.createTextNode(line.label + suffix));
      row.append(name, el("span", "plot-tip-val", fmt(pt[1])));
      this.tip.append(row);
      shown.push({ line, pt });
    }

    // If both a train trend/raw and validation point appear, show inline Val − Train delta
    const valHit = shown.find((s) => s.line.dash && Number.isFinite(s.pt[1]));
    const trainHit =
      shown.find((s) => s.line.trend && Number.isFinite(s.pt[1])) ||
      shown.find((s) => !s.line.dash && Number.isFinite(s.pt[1]));
    if (valHit && trainHit && shown.length <= 4) {
      const diff = valHit.pt[1] - trainHit.pt[1];
      const sign = diff > 0 ? "+" : "";
      this.tip.append(el("div", "plot-tip-delta", `Val − Train  ${sign}${fmt(diff, true)}`));
    }

    this.tip.hidden = false;
    const avail = this.area.clientWidth - this.tip.offsetWidth - 6;
    const preferRight = px + 14;
    const left = preferRight <= avail ? preferRight : Math.max(4, px - this.tip.offsetWidth - 14);
    this.tip.style.left = left + "px";
    this.tip.style.top = "8px";
    if (notify) broadcastHover(this, this.axis, at);
  }

  hide(notify = false) {
    this.cross.hidden = true;
    this.tip.hidden = true;
    this.hoverX = null;
    if (notify) broadcastHover(this, this.axis, null);
  }

  destroy() {
    activePlots.delete(this);
    this.unsubTheme();
    this.observer.disconnect();
    this.card.remove();
  }
}
