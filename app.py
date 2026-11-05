"""
NeuroCognitive AI - digital cognitive screening prototype (Streamlit).

Paper tests (MoCA, clock drawing) record only the final static drawing.
This app records HOW the task was performed: touch-initiation latency,
hesitation pauses, stroke velocity, path deviation and recall.

Run locally:   streamlit run app.py
Dependencies:  streamlit, plotly, numpy, pandas, scikit-learn  (see requirements.txt)

Research prototype for demonstration. Not a medical device and not a diagnostic tool.
"""

import html
import inspect
import math
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="NeuroCognitive AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#15324B"
MUTED = "#5A6F82"
LINE = "#D5DFE8"
SOFT = "#F3F6F9"
BLUE = "#1F5C99"
GREEN = "#1E8E5A"
AMBER = "#C98A12"
RED = "#C0392B"

TIERS = [
    {"name": "Low Risk", "color": GREEN, "fg": "#146C43", "bg": "#E6F4EC", "icon": "🟢"},
    {"name": "Borderline / Mild Decay", "color": AMBER, "fg": "#7A5200", "bg": "#FBF1DA", "icon": "🟡"},
    {"name": "High MCI / Vascular Risk Alert", "color": RED, "fg": "#9B2C22", "bg": "#FBE6E3", "icon": "🔴"},
]

LIVE_LABEL = "Select Custom / Live Test"
PRESET_A = "Preset A: Healthy Control (62yo) - Fast reaction (~280ms), 0 pauses"
PRESET_B = "Preset B: Early MCI / High Hesitation (69yo) - Normal final result, high sub-second hesitation (580ms, 4 pauses)"
PRESET_C = "Preset C: Vascular Cognitive Impairment (74yo) - High latency & motor-tremor deviation"

PAUSE_MS = 1000.0          # an inter-tap gap above this counts as a hesitation pause
Z_MIN, Z_MAX = -3.0, 6.0   # z-scores are clamped so single metrics cannot dominate the display

# Normative reference matrix (mean, SD) per age cohort.
# Latency and pause baselines follow the project specification; velocity-variability
# and path-deviation values are prototype reference values for this demo.
NORMS = {
    "45-55": {"latency_ms": (260.0, 35.0), "pauses": (0.5, 0.4), "vel_cv": (32.0, 8.0), "path_dev_mm": (2.5, 1.2)},
    "56-65": {"latency_ms": (280.0, 40.0), "pauses": (0.5, 0.4), "vel_cv": (35.0, 9.0), "path_dev_mm": (3.0, 1.5)},
    "66-85": {"latency_ms": (340.0, 50.0), "pauses": (0.5, 0.4), "vel_cv": (40.0, 10.0), "path_dev_mm": (3.8, 1.8)},
}

FEATURES = ["age", "education", "vascular", "latency_ms", "pauses", "vel_cv", "path_dev_mm", "verbal_ratio"]

# Demo patients. "raw" has the same structure the live task produces.
PRESETS = {
    PRESET_A: {
        "pid": "KZ-DEMO-A-0062", "age": 62, "edu": 15, "vasc": "No",
        "raw": {
            "latencies_ms": [272.0, 288.0, 280.0], "false_starts": 0,
            "intervals_ms": [640.0, 560.0, 690.0, 610.0, 650.0],
            "distances_mm": [30.0, 58.0, 41.0, 66.0, 36.0],
            "errors": 0, "path_dev_mm": 2.7, "straightness": 0.88,
            "digit_shown": "5826", "digit_entered": "5826", "digit_correct": 4, "digit_length": 4,
        },
    },
    PRESET_B: {
        "pid": "KZ-DEMO-B-0069", "age": 69, "edu": 16, "vasc": "No",
        "raw": {
            "latencies_ms": [566.0, 601.0, 573.0], "false_starts": 0,
            "intervals_ms": [1210.0, 1480.0, 690.0, 1350.0, 1620.0],
            "distances_mm": [30.0, 58.0, 41.0, 66.0, 36.0],
            "errors": 0, "path_dev_mm": 3.3, "straightness": 0.85,
            "digit_shown": "4297", "digit_entered": "4297", "digit_correct": 4, "digit_length": 4,
        },
    },
    PRESET_C: {
        "pid": "KZ-DEMO-C-0074", "age": 74, "edu": 9, "vasc": "Yes",
        "raw": {
            "latencies_ms": [455.0, 492.0, 468.0], "false_starts": 1,
            "intervals_ms": [980.0, 1260.0, 1090.0, 870.0, 1410.0],
            "distances_mm": [22.0, 70.0, 28.0, 74.0, 20.0],
            "errors": 1, "path_dev_mm": 8.6, "straightness": 0.58,
            "digit_shown": "7391", "digit_entered": "7394", "digit_correct": 3, "digit_length": 4,
        },
    },
}

# ---------------------------------------------------------------------------
# Client-side task (runs in the browser so timing is millisecond-accurate)
# ---------------------------------------------------------------------------
TASK_HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap");
:root{--ink:#15324B;--muted:#5A6F82;--line:#D5DFE8;--soft:#F3F6F9;--blue:#1F5C99;--green:#1E8E5A;--red:#C0392B;}
*{box-sizing:border-box;font-family:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
body{margin:0;padding:2px;color:var(--ink);background:transparent;}
.wrap{border:1px solid var(--line);border-radius:10px;background:#fff;padding:14px;}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap;}
.phase{font-weight:600;font-size:15px;}
.prog{display:flex;gap:6px;}
.dot{width:34px;height:6px;border-radius:3px;background:var(--line);}
.dot.on{background:var(--blue);} .dot.done{background:var(--green);}
.msg{color:var(--muted);font-size:14px;margin:0 0 10px;min-height:20px;line-height:1.4;}
.stage{position:relative;height:300px;border:1px solid var(--line);border-radius:8px;background:var(--soft);overflow:hidden;touch-action:manipulation;user-select:none;-webkit-user-select:none;}
.center{position:absolute;left:0;top:0;right:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:10px;text-align:center;}
.btn{background:var(--blue);color:#fff;border:0;border-radius:8px;padding:12px 22px;font-size:15px;font-weight:600;cursor:pointer;}
.btn.ghost{background:#fff;color:var(--blue);border:1px solid var(--blue);}
.btn:focus-visible,.target:focus-visible,.node:focus-visible,.key:focus-visible{outline:3px solid #8DB6E0;outline-offset:2px;}
.target{width:150px;height:150px;border-radius:50%;border:4px solid #fff;box-shadow:0 0 0 2px var(--line);font-size:17px;font-weight:600;color:#fff;cursor:pointer;touch-action:manipulation;}
.target.wait{background:#7A8FA3;} .target.go{background:var(--green);} .target.bad{background:var(--red);} .target.done{background:var(--blue);}
.node{position:absolute;transform:translate(-50%,-50%);width:54px;height:54px;border-radius:50%;border:2px solid var(--blue);background:#fff;color:var(--blue);font-size:20px;font-weight:600;cursor:pointer;touch-action:manipulation;}
.node.ok{background:var(--green);border-color:var(--green);color:#fff;}
.node.bad{background:var(--red);border-color:var(--red);color:#fff;}
.digits{font-size:48px;font-weight:600;letter-spacing:6px;color:var(--ink);}
.entry{min-height:44px;font-size:34px;font-weight:600;letter-spacing:8px;color:var(--ink);}
.pad{display:grid;grid-template-columns:repeat(3,64px);gap:8px;}
.key{height:40px;border-radius:8px;border:1px solid var(--line);background:#fff;font-size:18px;font-weight:600;color:var(--ink);cursor:pointer;touch-action:manipulation;}
.key.fn{font-size:13px;color:var(--muted);}
.done-title{font-size:20px;font-weight:600;} .done-sub{color:var(--muted);font-size:14px;line-height:1.5;}
</style></head>
<body>
<div class="wrap">
  <div class="top"><div class="phase" id="phase">Ready</div><div class="prog" id="prog"></div></div>
  <p class="msg" id="msg"></p>
  <div class="stage" id="stage"></div>
</div>
<script>
(function(){
"use strict";
var MM_PER_PX = 25.4 / 96, N_TRIALS = 3, N_TARGETS = 6, SPAN_LEN = 4, SPAN_SHOW_MS = 2500, MIN_VALID_RT = 100;
var POS = [[11,72],[27,24],[45,72],[62,26],[79,68],[90,22]];
var stage = document.getElementById("stage"), msgEl = document.getElementById("msg");
var phaseEl = document.getElementById("phase"), progEl = document.getElementById("prog");
var S = null;

function post(type, data){
  var m = {isStreamlitMessage: true, type: type};
  for (var k in data){ if (Object.prototype.hasOwnProperty.call(data, k)) { m[k] = data[k]; } }
  window.parent.postMessage(m, "*");
}
function setHeight(){
  var h = Math.ceil(document.querySelector(".wrap").getBoundingClientRect().height) + 12;
  post("streamlit:setFrameHeight", {height: h});
}
function sendValue(v){ post("streamlit:setComponentValue", {value: v, dataType: "json"}); }
function nowOf(e){
  if (e && typeof e.timeStamp === "number" && e.timeStamp > 0 && e.timeStamp < 1e10) { return e.timeStamp; }
  return performance.now();
}
function setProgress(idx){
  progEl.innerHTML = "";
  for (var i = 0; i < 3; i++){
    var d = document.createElement("span");
    d.className = "dot" + (i < idx ? " done" : (i === idx ? " on" : ""));
    progEl.appendChild(d);
  }
}
function newState(){
  return {phase:"intro", lat:[], early:0, goT:0, timer:null,
    trail:{next:1,t0:0,lastT:0,lastPt:null,intervals:[],dists:[],errors:0,samples:[],devs:[],strs:[],firstMs:null},
    digit:{seq:"",entered:""}};
}
function clearTimers(){ if (S && S.timer){ clearTimeout(S.timer); S.timer = null; } }
function mean(a){ var s = 0; for (var i = 0; i < a.length; i++){ s += a[i]; } return a.length ? s / a.length : 0; }
function r(x, p){ var f = Math.pow(10, p); return Math.round(x * f) / f; }

function showIntro(){
  clearTimers(); stage.onpointermove = null; S = newState(); setProgress(0);
  phaseEl.textContent = "Ready to begin";
  msgEl.textContent = "Three short tasks, about one minute in total. Use your usual pointing device or finger.";
  stage.innerHTML = '<div class="center"><button type="button" class="btn" id="startBtn">Start assessment</button></div>';
  document.getElementById("startBtn").addEventListener("click", startReaction);
  setHeight();
}

function startReaction(){
  clearTimers(); S.phase = "wait"; setProgress(0);
  phaseEl.textContent = "Task 1 of 3: Reaction";
  msgEl.textContent = "Trial " + (S.lat.length + 1) + " of " + N_TRIALS + ". Tap the circle as soon as it turns green. Do not tap before.";
  stage.innerHTML = '<div class="center"><button type="button" class="target wait" id="tgt">Wait</button></div>';
  var t = document.getElementById("tgt");
  t.addEventListener("pointerdown", onTarget);
  S.timer = setTimeout(function(){
    if (S.phase !== "wait") { return; }
    S.phase = "go"; t.className = "target go"; t.textContent = "Tap now"; S.goT = performance.now();
  }, 1300 + Math.random() * 2200);
  setHeight();
}

function onTarget(e){
  e.preventDefault();
  var t = document.getElementById("tgt");
  if (!t) { return; }
  if (S.phase === "wait"){
    clearTimers(); S.early++; S.phase = "pause";
    t.className = "target bad"; t.textContent = "Too early";
    S.timer = setTimeout(startReaction, 1000);
    return;
  }
  if (S.phase === "go"){
    var rt = nowOf(e) - S.goT;
    S.phase = "pause";
    if (rt < MIN_VALID_RT){
      S.early++; t.className = "target bad"; t.textContent = "Too early";
      S.timer = setTimeout(startReaction, 1000);
      return;
    }
    S.lat.push(r(rt, 1));
    t.className = "target done"; t.textContent = Math.round(rt) + " ms";
    S.timer = setTimeout(function(){
      if (S.lat.length >= N_TRIALS) { showTrailIntro(); } else { startReaction(); }
    }, 1000);
  }
}

function showTrailIntro(){
  S.phase = "trailIntro"; setProgress(1);
  phaseEl.textContent = "Task 2 of 3: Trail";
  msgEl.textContent = "Tap the numbers 1 to 6 in order, as quickly and accurately as you can.";
  stage.innerHTML = '<div class="center"><button type="button" class="btn" id="goTrail">Show numbers</button></div>';
  document.getElementById("goTrail").addEventListener("click", startTrail);
  setHeight();
}

function startTrail(){
  S.phase = "trail"; var T = S.trail; stage.innerHTML = "";
  for (var i = 0; i < N_TARGETS; i++){
    var b = document.createElement("button");
    b.type = "button"; b.className = "node"; b.textContent = String(i + 1); b.setAttribute("data-n", String(i + 1));
    b.style.left = POS[i][0] + "%"; b.style.top = POS[i][1] + "%";
    b.addEventListener("pointerdown", onNode);
    stage.appendChild(b);
  }
  stage.onpointermove = function(e){
    if (S.phase !== "trail") { return; }
    if (S.trail.samples.length < 3000) { S.trail.samples.push([e.clientX, e.clientY]); }
  };
  T.t0 = performance.now(); T.lastT = T.t0;
}

function segmentStats(T, chord, pt){
  var s = T.samples;
  if (s.length < 4 || chord < 30) { return; }
  var a = T.lastPt, dx = pt[0] - a[0], dy = pt[1] - a[1], len = Math.sqrt(dx * dx + dy * dy);
  var sum = 0, path = 0, prev = a;
  for (var i = 0; i < s.length; i++){
    sum += Math.abs(dx * (s[i][1] - a[1]) - dy * (s[i][0] - a[0])) / len;
    path += Math.sqrt(Math.pow(s[i][0] - prev[0], 2) + Math.pow(s[i][1] - prev[1], 2));
    prev = s[i];
  }
  path += Math.sqrt(Math.pow(pt[0] - prev[0], 2) + Math.pow(pt[1] - prev[1], 2));
  T.devs.push((sum / s.length) * MM_PER_PX);
  T.strs.push(Math.min(1, chord / Math.max(path, 1)));
}

function onNode(e){
  e.preventDefault();
  if (S.phase !== "trail") { return; }
  var T = S.trail, el = e.currentTarget, n = parseInt(el.getAttribute("data-n"), 10);
  var now = nowOf(e), pt = [e.clientX, e.clientY];
  if (n !== T.next){
    T.errors++; el.classList.add("bad");
    setTimeout(function(){ el.classList.remove("bad"); }, 250);
    return;
  }
  el.classList.add("ok");
  if (T.next === 1){
    T.firstMs = now - T.t0;
  } else {
    var chord = Math.sqrt(Math.pow(pt[0] - T.lastPt[0], 2) + Math.pow(pt[1] - T.lastPt[1], 2));
    T.intervals.push(r(now - T.lastT, 1));
    T.dists.push(r(Math.max(chord * MM_PER_PX, 1), 2));
    segmentStats(T, chord, pt);
  }
  T.lastT = now; T.lastPt = pt; T.samples = []; T.next++;
  if (T.next > N_TARGETS){
    S.phase = "trailDone"; stage.onpointermove = null;
    setTimeout(showDigitIntro, 500);
  }
}

function showDigitIntro(){
  S.phase = "digitIntro"; setProgress(2);
  phaseEl.textContent = "Task 3 of 3: Digit recall";
  msgEl.textContent = "Memorize " + SPAN_LEN + " digits shown for a few seconds, then enter them in the same order.";
  stage.innerHTML = '<div class="center"><button type="button" class="btn" id="goDigit">Show digits</button></div>';
  document.getElementById("goDigit").addEventListener("click", startDigits);
  setHeight();
}

function startDigits(){
  var seq = [];
  while (seq.length < SPAN_LEN){
    var d = Math.floor(Math.random() * 10);
    if (seq.length === 0 || seq[seq.length - 1] !== d) { seq.push(d); }
  }
  S.digit.seq = seq.join(""); S.digit.entered = ""; S.phase = "digitShow";
  msgEl.textContent = "Memorize the digits.";
  stage.innerHTML = '<div class="center"><div class="digits">' + seq.join(" ") + '</div></div>';
  S.timer = setTimeout(showKeypad, SPAN_SHOW_MS);
}

function showKeypad(){
  S.phase = "digitEnter";
  msgEl.textContent = "Enter the digits in the same order.";
  var keys = ["1","2","3","4","5","6","7","8","9","C","0","B"], h = "";
  for (var i = 0; i < keys.length; i++){
    var k = keys[i], label = k === "C" ? "Clear" : (k === "B" ? "Back" : k);
    h += '<button type="button" class="key' + (k === "C" || k === "B" ? " fn" : "") + '" data-k="' + k + '">' + label + '</button>';
  }
  stage.innerHTML = '<div class="center"><div class="entry" id="entry">&nbsp;</div><div class="pad" id="pad">' + h + '</div></div>';
  var pad = document.getElementById("pad");
  pad.addEventListener("pointerdown", function(e){
    var t = e.target;
    if (!t || !t.getAttribute) { return; }
    var k = t.getAttribute("data-k");
    if (!k || S.phase !== "digitEnter") { return; }
    e.preventDefault();
    if (k === "C") { S.digit.entered = ""; }
    else if (k === "B") { S.digit.entered = S.digit.entered.slice(0, -1); }
    else if (S.digit.entered.length < SPAN_LEN) { S.digit.entered += k; }
    var en = document.getElementById("entry");
    en.textContent = S.digit.entered.split("").join(" ") || "\u00a0";
    if (S.digit.entered.length === SPAN_LEN){
      S.phase = "digitDone";
      setTimeout(finish, 350);
    }
  });
  setHeight();
}

function finish(){
  S.phase = "done";
  var T = S.trail, correct = 0;
  for (var i = 0; i < SPAN_LEN; i++){
    if (S.digit.entered.charAt(i) === S.digit.seq.charAt(i)) { correct++; }
  }
  var payload = {
    complete: true,
    latencies_ms: S.lat,
    false_starts: S.early,
    trail_first_ms: T.firstMs === null ? null : r(T.firstMs, 1),
    intervals_ms: T.intervals,
    distances_mm: T.dists,
    errors: T.errors,
    path_dev_mm: T.devs.length ? r(mean(T.devs), 2) : null,
    straightness: T.strs.length ? r(mean(T.strs), 3) : null,
    digit_shown: S.digit.seq,
    digit_entered: S.digit.entered,
    digit_correct: correct,
    digit_length: SPAN_LEN,
    captured_at: Date.now()
  };
  setProgress(3);
  phaseEl.textContent = "Assessment complete";
  msgEl.textContent = "The capture was sent to the analysis engine below.";
  stage.innerHTML = '<div class="center"><div class="done-title">Capture complete</div>' +
    '<div class="done-sub">Mean latency ' + Math.round(mean(S.lat)) + ' ms. ' + T.intervals.length +
    ' trail intervals recorded, ' + T.errors + ' wrong taps, digits ' + correct + '/' + SPAN_LEN + '.</div>' +
    '<button type="button" class="btn ghost" id="again">Run again</button></div>';
  document.getElementById("again").addEventListener("click", function(){
    sendValue({complete: false, reset: true});
    showIntro();
  });
  sendValue(payload);
  setHeight();
}

window.addEventListener("message", function(ev){
  if (ev.data && ev.data.type === "streamlit:render") { setHeight(); }
});
post("streamlit:componentReady", {apiVersion: 1});
showIntro();
})();
</script></body></html>
'''


@st.cache_resource(show_spinner=False)
def get_task_component():
    """Write the task page to a temp folder and register it as a Streamlit component."""
    folder = Path(tempfile.gettempdir()) / "neurocognitive_ai_task_v1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(TASK_HTML, encoding="utf-8")
    return components.declare_component("neurocognitive_task", path=str(folder))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def esc(value):
    return html.escape(str(value))


def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def cohort_for_age(age):
    if age <= 55:
        return "45-55"
    if age <= 65:
        return "56-65"
    return "66-85"


def zscore(value, mean, sd):
    return float(np.clip((value - mean) / sd, Z_MIN, Z_MAX))


def tier_from_z(z):
    if z < 1.0:
        return 0
    if z < 2.0:
        return 1
    return 2


def tier_from_prob(pct):
    if pct < 20:
        return 0
    if pct < 50:
        return 1
    return 2


def badge(tier_idx, text=None):
    t = TIERS[tier_idx]
    label = text if text else t["name"]
    return (f'<span class="nc-badge" style="background:{t["bg"]};color:{t["fg"]};'
            f'border:1px solid {t["color"]}">{esc(label)}</span>')


def html_table(headers, rows, active=None):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cls = ' class="active"' if active and active[i] else ""
        body += f"<tr{cls}>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f'<table class="nc-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def show_plot(fig):
    """Render a Plotly figure across Streamlit versions."""
    config = {"displayModeBar": False}
    try:
        params = inspect.signature(st.plotly_chart).parameters
        if "width" in params:
            st.plotly_chart(fig, width="stretch", config=config, theme=None)
        else:
            st.plotly_chart(fig, use_container_width=True, config=config, theme=None)
    except Exception:
        st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Data validation for the live task payload
# ---------------------------------------------------------------------------
def _num_list(values, low, high, name):
    if not isinstance(values, (list, tuple)) or len(values) < 1:
        raise ValueError(f"{name} is missing")
    out = []
    for item in values:
        number = float(item)
        if not math.isfinite(number) or number < low or number > high:
            raise ValueError(f"{name} contains an implausible value")
        out.append(number)
    return out


def sanitize_raw(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload is not an object")
    latencies = _num_list(payload.get("latencies_ms"), 80, 5000, "latencies")
    intervals = _num_list(payload.get("intervals_ms"), 50, 120000, "trail intervals")
    distances = _num_list(payload.get("distances_mm"), 0.5, 2000, "trail distances")
    if len(intervals) != len(distances):
        raise ValueError("trail intervals and distances differ in length")
    dev = payload.get("path_dev_mm")
    dev = None if dev is None else float(dev)
    if dev is not None and (not math.isfinite(dev) or dev < 0 or dev > 300):
        dev = None
    straight = payload.get("straightness")
    straight = None if straight is None else float(np.clip(float(straight), 0.0, 1.0))
    digit_len = int(payload.get("digit_length", 4))
    digit_len = max(1, min(digit_len, 12))
    digit_ok = int(np.clip(int(payload.get("digit_correct", 0)), 0, digit_len))
    return {
        "latencies_ms": latencies,
        "false_starts": int(max(0, int(payload.get("false_starts", 0)))),
        "intervals_ms": intervals,
        "distances_mm": distances,
        "errors": int(max(0, int(payload.get("errors", 0)))),
        "path_dev_mm": dev,
        "straightness": straight,
        "digit_shown": str(payload.get("digit_shown", "")),
        "digit_entered": str(payload.get("digit_entered", "")),
        "digit_correct": digit_ok,
        "digit_length": digit_len,
    }


# ---------------------------------------------------------------------------
# Machine-learning model trained on the fly from synthetic normative vectors
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model():
    rng = np.random.default_rng(2026)
    n = 3200
    age = rng.uniform(45, 85, n)
    edu = rng.integers(4, 21, n).astype(float)

    mean_lat = np.where(age <= 55, 260.0, np.where(age <= 65, 280.0, 340.0))
    sd_lat = np.where(age <= 55, 35.0, np.where(age <= 65, 40.0, 50.0))
    mean_cv = np.where(age <= 55, 32.0, np.where(age <= 65, 35.0, 40.0))
    sd_cv = np.where(age <= 55, 8.0, np.where(age <= 65, 9.0, 10.0))
    mean_dev = np.where(age <= 55, 2.5, np.where(age <= 65, 3.0, 3.8))
    sd_dev = np.where(age <= 55, 1.2, np.where(age <= 65, 1.5, 1.8))

    impaired = (rng.random(n) < 0.45).astype(int)
    subtype = np.where(impaired == 1, rng.choice(3, size=n, p=[0.40, 0.25, 0.35]), -1)

    # (latency z shift, pause rate, velocity-CV z shift, path-deviation z shift, vascular prevalence)
    shifts = {
        -1: (0.0, 0.5, 0.0, 0.0, 0.25),   # normal ageing
        0: (2.2, 3.0, 0.8, 0.5, 0.40),    # frontal-subcortical executive pattern
        1: (0.8, 1.2, 0.3, 0.2, 0.30),    # amnestic-leaning pattern
        2: (2.6, 2.5, 2.2, 2.8, 0.85),    # vascular / motor pattern
    }
    verbal_probs = {
        -1: [0.80, 0.15, 0.04, 0.01, 0.00],
        0: [0.70, 0.20, 0.07, 0.03, 0.00],
        1: [0.15, 0.25, 0.35, 0.20, 0.05],
        2: [0.45, 0.30, 0.15, 0.07, 0.03],
    }

    lat = np.zeros(n)
    pauses = np.zeros(n)
    cv = np.zeros(n)
    dev = np.zeros(n)
    vasc = np.zeros(n)
    verbal = np.zeros(n)
    levels = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    for key, (lat_shift, pause_rate, cv_shift, dev_shift, vasc_p) in shifts.items():
        idx = np.where(subtype == key)[0]
        m = len(idx)
        if m == 0:
            continue
        lat[idx] = mean_lat[idx] + sd_lat[idx] * rng.normal(lat_shift, 1.0, m)
        pauses[idx] = rng.poisson(pause_rate, m)
        cv[idx] = mean_cv[idx] + sd_cv[idx] * rng.normal(cv_shift, 1.0, m)
        dev[idx] = mean_dev[idx] + sd_dev[idx] * rng.normal(dev_shift, 1.0, m)
        vasc[idx] = (rng.random(m) < vasc_p).astype(float)
        verbal[idx] = rng.choice(levels, size=m, p=verbal_probs[key])

    data = pd.DataFrame({
        "age": age,
        "education": edu,
        "vascular": vasc,
        "latency_ms": np.clip(lat, 150, None),
        "pauses": pauses,
        "vel_cv": np.clip(cv, 5, None),
        "path_dev_mm": np.clip(dev, 0.3, None),
        "verbal_ratio": verbal,
    })[FEATURES]

    model = RandomForestClassifier(
        n_estimators=160, max_depth=7, min_samples_leaf=6, random_state=7, n_jobs=1
    )
    model.fit(data, impaired)
    return model, n


def predict_risk(model, features, z_comp):
    """Return Early-MCI risk probability in percent. Falls back to a logistic curve if the model is unavailable."""
    try:
        if model is None:
            raise RuntimeError("model unavailable")
        frame = pd.DataFrame([features], columns=FEATURES)
        prob = float(model.predict_proba(frame)[0][1])
    except Exception:
        prob = 1.0 / (1.0 + math.exp(-(1.1 * z_comp - 1.6)))
    return float(np.clip(prob * 100.0, 1.0, 99.0))


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------
def score_from_z(z, scale, base=85.0):
    return float(np.clip(base - scale * z, 5.0, 100.0))


def analyze(raw, age, edu, vasc_yes, model):
    cohort = cohort_for_age(age)
    norms = NORMS[cohort]

    latencies = np.array(raw["latencies_ms"], dtype=float)
    intervals = np.array(raw["intervals_ms"], dtype=float)
    distances = np.array(raw["distances_mm"], dtype=float)
    velocities = distances / (intervals / 1000.0)

    lat = float(latencies.mean())
    vel_mean = float(velocities.mean())
    vel_sd = float(velocities.std(ddof=1)) if len(velocities) > 1 else 0.0
    vel_cv = 100.0 * vel_sd / vel_mean if vel_mean > 0 else 0.0

    pause_idx = [i for i, t in enumerate(intervals) if t > PAUSE_MS]
    n_pauses = len(pause_idx)
    trail_time_s = float(intervals.sum() / 1000.0)
    excess_s = float(sum(intervals[i] - PAUSE_MS for i in pause_idx) / 1000.0)

    dev = raw.get("path_dev_mm")
    dev_captured = dev is not None
    dev_value = float(dev) if dev_captured else norms["path_dev_mm"][0]

    straight = raw.get("straightness")
    errors = int(raw.get("errors", 0))
    verbal_ratio = raw["digit_correct"] / float(raw["digit_length"])

    z_lat = zscore(lat, *norms["latency_ms"])
    z_pause = zscore(float(n_pauses), *norms["pauses"])
    z_cv = zscore(vel_cv, *norms["vel_cv"])
    z_dev = zscore(dev_value, *norms["path_dev_mm"])
    z_motor = (z_cv + z_dev) / 2.0
    z_comp = (z_lat + z_pause) / 2.0
    decoupling = z_lat - z_motor

    if straight is None:
        visuo_base = 85.0
    else:
        visuo_base = 30.0 + 70.0 * float(straight)
    domains = {
        "Processing Speed": score_from_z(z_lat, 12.0),
        "Visuospatial Control": float(np.clip(visuo_base - 15.0 * errors, 5.0, 100.0)),
        "Verbal Recall": float(np.clip(100.0 * verbal_ratio, 5.0, 100.0)),
        "Executive Function": float(np.clip(85.0 - 6.0 * z_pause - 5.0 * errors, 5.0, 100.0)),
        "Motor Smoothness": score_from_z(z_motor, 12.0),
    }

    features = {
        "age": float(age), "education": float(edu), "vascular": 1.0 if vasc_yes else 0.0,
        "latency_ms": lat, "pauses": float(n_pauses), "vel_cv": vel_cv,
        "path_dev_mm": dev_value, "verbal_ratio": verbal_ratio,
    }
    risk = predict_risk(model, features, z_comp)

    tier_z = tier_from_z(z_comp)
    tier_ml = tier_from_prob(risk)
    return {
        "cohort": cohort, "norms": norms, "lat": lat, "latencies": latencies,
        "intervals": intervals, "distances": distances, "velocities": velocities,
        "vel_mean": vel_mean, "vel_sd": vel_sd, "vel_cv": vel_cv,
        "pause_idx": pause_idx, "n_pauses": n_pauses, "trail_time_s": trail_time_s,
        "excess_s": excess_s, "dev": dev_value, "dev_captured": dev_captured,
        "straight": straight, "errors": errors, "false_starts": int(raw.get("false_starts", 0)),
        "verbal_ratio": verbal_ratio, "digit_correct": raw["digit_correct"],
        "digit_length": raw["digit_length"],
        "z_lat": z_lat, "z_pause": z_pause, "z_cv": z_cv, "z_dev": z_dev,
        "z_motor": z_motor, "z_comp": z_comp, "decoupling": decoupling,
        "domains": domains, "risk": risk, "tier_z": tier_z, "tier_ml": tier_ml,
        "tier_final": max(tier_z, tier_ml), "vasc_yes": vasc_yes,
    }


# ---------------------------------------------------------------------------
# Clinical report text
# ---------------------------------------------------------------------------
def pause_phase(i, n_intervals):
    if i == 0:
        return "early-task delay"
    if i == n_intervals - 1:
        return "late-task delay"
    return "mid-task delay"


def interpretation_points(A):
    zl, zp, zm, dec = A["z_lat"], A["z_pause"], A["z_motor"], A["decoupling"]
    ver = A["verbal_ratio"]
    points = []

    if zl >= 1 and zp >= 1 and ver >= 0.75 and dec >= 2:
        points.append(
            f"**Executive-pattern slowing.** High initiation latency ({zl:+.1f} SD) combined with "
            f"{A['n_pauses']} hesitation pauses, while movement execution stays near normal "
            f"(motor z {zm:+.1f}) and digit recall is preserved ({A['digit_correct']}/{A['digit_length']}), "
            "suggests early frontal-subcortical executive dysfunction rather than primary hippocampal memory loss. "
            "The delay sits in planning and initiating, not in the hand."
        )
    if zm >= 2 and zl >= 1:
        extra = " The reported vascular risk factor is consistent with this pattern." if A["vasc_yes"] else ""
        points.append(
            f"**Coupled cognitive-motor slowing.** Delayed initiation ({zl:+.1f} SD) together with high stroke-velocity "
            f"variability and path deviation (motor z {zm:+.1f}) points toward a subcortical small-vessel "
            f"(vascular) pattern, where both thinking speed and motor smoothness decline.{extra}"
        )
    if ver < 0.6:
        points.append(
            f"**Recall component.** Digit recall was {A['digit_correct']}/{A['digit_length']}. "
            "A working-memory or amnestic contribution cannot be excluded; delayed word recall would help separate it."
        )
    if not points and A["tier_final"] >= 1:
        points.append(
            "**Mild process-level slowing without a dominant pattern.** This can reflect early cognitive change, "
            "fatigue, anxiety or device and input factors. A repeat session is more informative than a single result."
        )
    if not points:
        points.append(
            "**No process-level markers of concern.** Initiation speed, pacing, movement variability and recall are all "
            "within the age-adjusted control range."
        )
    if A["vasc_yes"] and A["tier_final"] >= 1 and zm < 2:
        points.append(
            "Hypertension or vascular risk is reported; blood-pressure control and vascular risk review are relevant "
            "to any follow-up."
        )
    return points


def next_step(tier):
    if tier == 2:
        return ("Refer for a formal neuropsychological assessment (for example MoCA and Trail Making Test A/B) "
                "and clinical review. This screen does not establish a diagnosis.")
    if tier == 1:
        return "Repeat the screen in 3 to 6 months under similar conditions and consider a brief paper MoCA for comparison."
    return "No action needed beyond routine follow-up. Re-screen yearly or sooner if symptoms appear."


def build_audit(A, ctx):
    n = len(A["intervals"])
    lo, hi = float(A["latencies"].min()), float(A["latencies"].max())
    lat_mean, lat_sd = A["norms"]["latency_ms"]
    p_mean, p_sd = A["norms"]["pauses"]
    lines = []
    lines.append(
        f"**Patient {ctx['pid']}**, {ctx['age']} years, cohort {A['cohort']}, {ctx['edu']} years of education, "
        f"vascular risk: {'Yes' if A['vasc_yes'] else 'No'}."
    )
    lines.append("")
    lines.append("#### 1. What happened during execution")
    lines.append(
        f"- **Initiation latency:** mean {A['lat']:.0f} ms over {len(A['latencies'])} trials (range {lo:.0f} to {hi:.0f} ms) "
        f"against a cohort baseline of {lat_mean:.0f} ± {lat_sd:.0f} ms, which is {A['z_lat']:+.1f} SD."
    )
    if A["n_pauses"] == 0:
        lines.append(
            f"- **Hesitation pauses:** none. Every step of the trail was completed within {PAUSE_MS / 1000:.1f} s of the previous one."
        )
    else:
        lines.append(
            f"- **Hesitation pauses:** {A['n_pauses']} gaps longer than {PAUSE_MS / 1000:.1f} s "
            f"(cohort baseline {p_mean:.1f} ± {p_sd:.1f}, {A['z_pause']:+.1f} SD):"
        )
        for k, i in enumerate(A["pause_idx"], start=1):
            lines.append(
                f"    - Pause {k}: between targets {i + 1} and {i + 2}, {A['intervals'][i] / 1000:.2f} s "
                f"({pause_phase(i, n)})."
            )
    share = 100.0 * A["excess_s"] / A["trail_time_s"] if A["trail_time_s"] > 0 else 0.0
    lines.append(
        f"- **Hesitation time:** {A['excess_s']:.2f} s above the pause threshold out of {A['trail_time_s']:.2f} s of trail time "
        f"({share:.0f}%). This is thinking time that a finished drawing cannot show."
    )
    lines.append(
        f"- **Stroke velocity:** mean {A['vel_mean']:.1f} mm/s, deviation ±{A['vel_sd']:.1f} mm/s "
        f"(variability {A['vel_cv']:.0f}%, {A['z_cv']:+.1f} SD)."
    )
    if A["dev_captured"]:
        lines.append(
            f"- **Path deviation (tremor proxy):** {A['dev']:.1f} mm from the straight line between targets "
            f"({A['z_dev']:+.1f} SD)."
        )
    else:
        lines.append(
            "- **Path deviation:** not captured on this device (a touch screen records no movement between taps), "
            "so it was held at the cohort mean."
        )
    lines.append(
        f"- **Motor-cognitive decoupling:** initiation z minus motor z = {A['decoupling']:+.1f}. "
        + ("Planning is slowed out of proportion to movement." if A["decoupling"] >= 2
           else "Planning and movement are slowed to a similar degree." if A["z_lat"] >= 1
           else "No separation between planning and movement.")
    )
    extras = f"{A['errors']} wrong tap(s), {A['false_starts']} anticipatory tap(s), digit recall {A['digit_correct']}/{A['digit_length']}."
    lines.append(f"- **Accuracy:** {extras}")
    lines.append("")
    lines.append("#### 2. Why this score was generated")
    drivers = sorted(
        [("initiation latency", A["z_lat"]), ("hesitation pauses", A["z_pause"]),
         ("stroke-velocity variability", A["z_cv"]), ("path deviation", A["z_dev"])],
        key=lambda item: item[1], reverse=True,
    )
    top = [f"{name} ({z:+.1f} SD)" for name, z in drivers[:3] if z >= 1.0]
    if top:
        lines.append(
            f"The Z-score engine places this session at {A['z_comp']:+.1f} SD overall (latency and pauses combined), "
            f"driven mainly by {', '.join(top)}. The Random Forest model returns an Early MCI risk of {A['risk']:.0f}%."
        )
    else:
        lines.append(
            f"No metric reached 1 SD above the age-matched mean. Overall deviation is {A['z_comp']:+.1f} SD and the "
            f"Random Forest model returns an Early MCI risk of {A['risk']:.0f}%."
        )
    if A["tier_z"] != A["tier_ml"]:
        lines.append(
            f"The two engines land in different bands ({TIERS[A['tier_z']]['name']} by Z-score, "
            f"{TIERS[A['tier_ml']]['name']} by model), so the higher band is shown as the alert level."
        )
    lines.append("")
    lines.append("#### 3. Differential clinical interpretation")
    for point in interpretation_points(A):
        lines.append(f"- {point}")
    lines.append("")
    lines.append("#### 4. Suggested next step")
    lines.append(next_step(A["tier_final"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FONT = dict(family="IBM Plex Sans, -apple-system, Segoe UI, Roboto, Arial, sans-serif", color=INK, size=13)


def radar_figure(A):
    cats = list(A["domains"].keys())
    vals = [A["domains"][c] for c in cats]
    color = TIERS[A["tier_final"]]["color"]
    theta = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[85] * (len(cats) + 1), theta=theta, mode="lines", name="Normal control mean",
        line=dict(color=MUTED, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[70] * (len(cats) + 1), theta=theta, mode="lines", name="Lower limit of normal band",
        line=dict(color=GREEN, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=theta, mode="lines+markers", name="Patient",
        fill="toself", fillcolor=hex_rgba(color, 0.22),
        line=dict(color=color, width=3), marker=dict(size=7),
    ))
    fig.update_layout(
        font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], gridcolor=LINE, linecolor=LINE),
            angularaxis=dict(gridcolor=LINE, linecolor=LINE),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=50, r=50, t=20, b=40), height=420,
    )
    return fig


def zbar_figure(A):
    rows = [
        ("Initiation latency", A["z_lat"], f"{A['lat']:.0f} ms"),
        ("Hesitation pauses", A["z_pause"], f"{A['n_pauses']} pauses"),
        ("Velocity variability", A["z_cv"], f"{A['vel_cv']:.0f}% CV"),
        ("Path deviation", A["z_dev"], f"{A['dev']:.1f} mm"),
    ]
    labels = [r[0] for r in rows]
    zs = [r[1] for r in rows]
    colors = [TIERS[tier_from_z(z)]["color"] for z in zs]
    texts = [f"{r[2]}  ({r[1]:+.1f} SD)" for r in rows]
    fig = go.Figure(go.Bar(
        x=zs, y=labels, orientation="h", marker_color=colors, text=texts, textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:+.2f} SD<extra></extra>",
    ))
    fig.add_vrect(x0=-1, x1=1, fillcolor=GREEN, opacity=0.14, line_width=0, layer="below")
    fig.add_vrect(x0=1, x1=2, fillcolor=AMBER, opacity=0.16, line_width=0, layer="below")
    fig.add_vrect(x0=2, x1=7.5, fillcolor=RED, opacity=0.08, line_width=0, layer="below")
    for x, text in [(0, "Normal control band"), (1.5, "1 to 2 SD"), (4.7, "2 SD or more: alert")]:
        fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=text, showarrow=False,
                           font=dict(size=11, color=MUTED))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(
        font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", showlegend=False,
        xaxis=dict(title="Standard deviations from the age-adjusted normal control mean",
                   range=[-3.5, 9.5], zeroline=False, gridcolor=LINE),
        yaxis=dict(autorange="reversed", gridcolor=LINE),
        margin=dict(l=10, r=10, t=40, b=50), height=420,
    )
    return fig


def timeline_figure(A):
    n = len(A["intervals"])
    secs = [t / 1000.0 for t in A["intervals"]]
    labels = [f"{i + 1} to {i + 2}" for i in range(n)]
    colors = [RED if t * 1000.0 > PAUSE_MS else BLUE for t in secs]
    fig = go.Figure(go.Bar(
        x=labels, y=secs, marker_color=colors, text=[f"{t:.2f} s" for t in secs], textposition="outside",
        cliponaxis=False, hovertemplate="Targets %{x}: %{y:.2f} s<extra></extra>",
    ))
    fig.add_hline(y=PAUSE_MS / 1000.0, line_dash="dash", line_color=RED,
                  annotation_text="Hesitation threshold (1.0 s)", annotation_position="top left",
                  annotation_font=dict(size=11, color=RED))
    ymax = max(secs + [PAUSE_MS / 1000.0]) * 1.3
    fig.update_layout(
        font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", showlegend=False,
        xaxis=dict(title="Trail step (target to target)"),
        yaxis=dict(title="Time between taps (s)", range=[0, ymax], gridcolor=LINE),
        margin=dict(l=10, r=10, t=20, b=50), height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp{font-family:'IBM Plex Sans',-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#F3F6F9;color:#15324B;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1180px;}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #D5DFE8;}
h1,h2,h3,h4{color:#15324B;letter-spacing:-0.01em;}
.nc-hero{background:#FFFFFF;border:1px solid #D5DFE8;border-radius:10px;padding:22px 26px;margin-bottom:18px;}
.nc-hero h1{margin:0 0 4px 0;font-size:2.1rem;font-weight:600;padding:0;}
.nc-hero p{margin:0;color:#5A6F82;font-size:1.02rem;line-height:1.5;max-width:760px;}
.nc-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
.nc-chip{background:#F3F6F9;border:1px solid #D5DFE8;border-radius:6px;padding:4px 10px;font-size:0.86rem;color:#15324B;}
.nc-step{margin:30px 0 12px 0;}
.nc-step .tag{display:inline-block;background:#1F5C99;color:#fff;border-radius:6px;padding:2px 10px;font-size:0.84rem;font-weight:600;margin-bottom:6px;}
.nc-step h2{margin:0;font-size:1.5rem;font-weight:600;padding:0;}
.nc-step p{margin:4px 0 0 0;color:#5A6F82;font-size:0.97rem;line-height:1.5;max-width:820px;}
.nc-card{background:#FFFFFF;border:1px solid #D5DFE8;border-radius:8px;padding:16px 18px;height:100%;}
.nc-card h4{margin:0 0 8px 0;font-size:1.02rem;font-weight:600;}
.nc-card p,.nc-card li{font-size:0.94rem;line-height:1.55;color:#15324B;margin:0 0 4px 0;}
.nc-banner{border-radius:8px;border:1px solid #D5DFE8;border-left-width:7px;padding:14px 18px;margin:14px 0 6px 0;}
.nc-banner .t{font-size:1.15rem;font-weight:600;margin-bottom:3px;}
.nc-banner .d{font-size:0.95rem;line-height:1.5;}
.nc-bar{height:12px;background:#E3EAF0;border-radius:6px;overflow:hidden;margin:8px 0 4px 0;}
.nc-bar > div{height:100%;border-radius:6px;}
.nc-table{width:100%;border-collapse:collapse;background:#FFFFFF;border:1px solid #D5DFE8;border-radius:8px;overflow:hidden;font-size:0.92rem;}
.nc-table th{background:#EAF0F5;color:#15324B;text-align:left;font-weight:600;padding:9px 12px;border-bottom:1px solid #D5DFE8;}
.nc-table td{padding:9px 12px;border-bottom:1px solid #E6ECF1;color:#15324B;}
.nc-table tr:last-child td{border-bottom:0;}
.nc-table tr.active td{background:#EDF4FB;font-weight:600;}
.nc-badge{display:inline-block;border-radius:5px;padding:2px 8px;font-size:0.84rem;font-weight:600;white-space:nowrap;}
.nc-note{color:#5A6F82;font-size:0.88rem;line-height:1.5;margin-top:8px;}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #D5DFE8;border-radius:8px;padding:14px 16px;}
[data-testid="stMetricLabel"]{color:#5A6F82;}
[data-testid="stVerticalBlockBorderWrapper"]{background:#FFFFFF;}
.nc-foot{margin-top:34px;padding-top:14px;border-top:1px solid #D5DFE8;color:#5A6F82;font-size:0.86rem;line-height:1.55;}
</style>
"""


def step_header(number, title, description):
    st.markdown(
        f'<div class="nc-step"><span class="tag">Step {number}</span><h2>{esc(title)}</h2>'
        f'<p>{esc(description)}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def apply_preset():
    preset = PRESETS.get(st.session_state.get("preset_choice"))
    if preset:
        st.session_state["pid"] = preset["pid"]
        st.session_state["age"] = preset["age"]
        st.session_state["edu"] = preset["edu"]
        st.session_state["vasc"] = preset["vasc"]


def load_preset_b():
    st.session_state["preset_choice"] = PRESET_B
    apply_preset()


def render_sidebar():
    defaults = {"pid": "KZ-2026-9042", "age": 68, "edu": 12, "vasc": "No", "preset_choice": LIVE_LABEL}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.markdown("### ⚙️ Patient Clinical Metadata")
        st.text_input("Patient ID", key="pid")
        st.slider("Age (years)", 45, 85, key="age")
        st.slider("Years of formal education", 4, 20, key="edu")
        st.radio("Vascular risk / hypertension", ["No", "Yes"], horizontal=True, key="vasc")
        st.selectbox(
            "Demo patient preset",
            [LIVE_LABEL, PRESET_A, PRESET_B, PRESET_C],
            key="preset_choice",
            on_change=apply_preset,
        )
        st.caption(f"Age-matched normative cohort: {cohort_for_age(st.session_state['age'])}")
        st.caption("Presets load a recorded session. Choose the live option to analyse your own run of the task.")

    return {
        "pid": st.session_state["pid"].strip() or "Unnamed patient",
        "age": int(st.session_state["age"]),
        "edu": int(st.session_state["edu"]),
        "vasc_yes": st.session_state["vasc"] == "Yes",
        "preset": st.session_state["preset_choice"],
    }


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def render_hero(ctx):
    cohort = cohort_for_age(ctx["age"])
    st.markdown(
        '<div class="nc-hero"><h1>NeuroCognitive AI</h1>'
        "<p>A paper clock-drawing test only sees the finished drawing. This screen records how the task was "
        "performed: how fast the person starts, where they hesitate, how steadily they move.</p>"
        '<div class="nc-chips">'
        f'<span class="nc-chip">Patient {esc(ctx["pid"])}</span>'
        f'<span class="nc-chip">Age {ctx["age"]}</span>'
        f'<span class="nc-chip">Cohort {cohort}</span>'
        f'<span class="nc-chip">Education {ctx["edu"]} years</span>'
        f'<span class="nc-chip">Vascular risk: {"Yes" if ctx["vasc_yes"] else "No"}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_step1(ctx):
    step_header(
        1, "Dynamic micro-task",
        "Three short tasks record initiation latency in milliseconds, hesitation pauses above 1.0 second, "
        "stroke velocity in mm/s and path deviation, all timed in the browser.",
    )
    if ctx["preset"] != LIVE_LABEL:
        st.info("A demo preset is loaded, so the dashboard shows its recorded session. "
                "Choose “Select Custom / Live Test” in the sidebar to analyse your own run of the task.")

    try:
        component = get_task_component()
        payload = component(key="nc_task", default=None)
    except Exception as exc:
        payload = None
        st.warning("The interactive task could not load in this browser, so the demo presets are the way to "
                   f"present. Details: {exc}")

    if isinstance(payload, dict):
        if payload.get("reset"):
            st.session_state.pop("live_raw", None)
        elif payload.get("complete"):
            try:
                st.session_state["live_raw"] = sanitize_raw(payload)
            except Exception as exc:
                st.warning(f"That capture was discarded because it looked invalid ({exc}). Please run the task again.")
                st.session_state.pop("live_raw", None)


def resolve_raw(ctx):
    if ctx["preset"] in PRESETS:
        return PRESETS[ctx["preset"]]["raw"]
    return st.session_state.get("live_raw")


def render_capture_log(A):
    rows = []
    for i in range(len(A["intervals"])):
        paused = A["intervals"][i] > PAUSE_MS
        rows.append({
            "Trail step": f"{i + 1} to {i + 2}",
            "Interval (ms)": round(float(A["intervals"][i])),
            "Distance (mm)": round(float(A["distances"][i]), 1),
            "Velocity (mm/s)": round(float(A["velocities"][i]), 1),
            "Hesitation (over 1.0 s)": "Yes" if paused else "No",
        })
    frame = pd.DataFrame(rows)
    with st.expander("Raw capture log"):
        st.caption("Reaction trials (ms): " + ", ".join(f"{v:.0f}" for v in A["latencies"]))
        try:
            st.dataframe(frame, hide_index=True)
        except Exception:
            st.table(frame)


def render_step2(A, ctx, model, n_train):
    step_header(
        2, "Normative comparison and Z-score engine",
        "Each metric is compared with people of the same age group. A Random Forest trained on the fly adds an "
        "Early MCI risk probability.",
    )
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("**Normative baseline matrix**")
        frame = pd.DataFrame([
            {
                "cohort": cohort,
                "lat": f"{v['latency_ms'][0]:.0f} ± {v['latency_ms'][1]:.0f}",
                "pau": f"{v['pauses'][0]:.1f} ± {v['pauses'][1]:.1f}",
                "cv": f"{v['vel_cv'][0]:.0f} ± {v['vel_cv'][1]:.0f}",
                "dev": f"{v['path_dev_mm'][0]:.1f} ± {v['path_dev_mm'][1]:.1f}",
            }
            for cohort, v in NORMS.items()
        ])
        rows = [[r.cohort, r.lat, r.pau, r.cv, r.dev] for r in frame.itertuples(index=False)]
        active = [r[0] == A["cohort"] for r in rows]
        st.markdown(
            html_table(
                ["Age cohort", "Initiation latency (ms)", "Hesitation pauses", "Velocity variability (CV %)",
                 "Path deviation (mm)"],
                rows, active,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nc-note">Values are mean ± SD. The highlighted row is the active cohort. These are '
            "prototype reference values for demonstration and need validation against a clinical cohort "
            "before any clinical use.</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Patient Z-scores against the active cohort**")
        norms = A["norms"]
        z_rows = [
            ("Initiation latency", f"{A['lat']:.0f} ms", norms["latency_ms"], A["z_lat"], "{:.0f}"),
            ("Hesitation pauses", f"{A['n_pauses']}", norms["pauses"], A["z_pause"], "{:.1f}"),
            ("Velocity variability", f"{A['vel_cv']:.0f}%", norms["vel_cv"], A["z_cv"], "{:.0f}"),
            ("Path deviation", f"{A['dev']:.1f} mm" + ("" if A["dev_captured"] else " (n/a)"),
             norms["path_dev_mm"], A["z_dev"], "{:.1f}"),
        ]
        rows = []
        for name, value, (mean, sd), z, fmt in z_rows:
            tier = tier_from_z(z)
            short = {0: "Low", 1: "Borderline", 2: "Alert"}[tier]
            rows.append([name, value, f"{fmt.format(mean)} ± {fmt.format(sd)}", f"{z:+.1f}", badge(tier, short)])
        st.markdown(html_table(["Metric", "Patient", "Cohort", "Z", "Band"], rows), unsafe_allow_html=True)
        st.markdown(
            '<div class="nc-note">Z below 1.0 SD is low risk, 1.0 to 2.0 SD is borderline, and 2.0 SD or more raises '
            "an alert. Z-scores are limited to the range -3 to +6.</div>",
            unsafe_allow_html=True,
        )

    final = TIERS[A["tier_final"]]
    messages = {
        0: "Process metrics sit inside the age-adjusted control range.",
        1: "Some process metrics show mild decay compared with the age-matched control group.",
        2: "Process metrics deviate strongly from age-matched controls. Formal assessment is recommended.",
    }
    st.markdown(
        f'<div class="nc-banner" style="background:{final["bg"]};border-left-color:{final["color"]}">'
        f'<div class="t" style="color:{final["fg"]}">{final["icon"]} {esc(final["name"])}</div>'
        f'<div class="d">{esc(messages[A["tier_final"]])} Overall deviation {A["z_comp"]:+.1f} SD '
        f'({esc(TIERS[A["tier_z"]]["name"])}), model risk {A["risk"]:.0f}% ({esc(TIERS[A["tier_ml"]]["name"])}). '
        "The alert level is the higher of the two.</div></div>",
        unsafe_allow_html=True,
    )
    ml_color = TIERS[A["tier_ml"]]["color"]
    st.markdown(
        f"**Early MCI risk probability: {A['risk']:.0f}%**"
        f'<div class="nc-bar"><div style="width:{A["risk"]:.0f}%;background:{ml_color}"></div></div>'
        f'<div class="nc-note">Random Forest (160 trees) trained at load time on {n_train:,} synthetic normative and '
        "impaired vectors generated from the baseline matrix. It illustrates the approach and has not been "
        "validated on patient data.</div>",
        unsafe_allow_html=True,
    )


def render_step3(A):
    step_header(
        3, "Clinical diagnostic dashboard",
        "Key results first, then performance across five clinical domains and against the age-adjusted control band.",
    )
    z_tier = TIERS[A["tier_z"]]["name"]
    lat_mean = A["norms"]["latency_ms"][0]
    p_mean = A["norms"]["pauses"][0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{A['risk']:.0f}%", delta=TIERS[A["tier_ml"]]["name"], delta_color="off")
    c2.metric("Mean Latency", f"{A['lat']:.0f} ms", delta=f"{A['lat'] - lat_mean:+.0f} ms vs cohort", delta_color="inverse")
    c3.metric("Hesitation Pauses", f"{A['n_pauses']}", delta=f"{A['n_pauses'] - p_mean:+.1f} vs cohort", delta_color="inverse")
    c4.metric("Z-Score Deviation", f"{A['z_comp']:+.1f} SD", delta=z_tier, delta_color="off")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Performance across five clinical domains**")
        show_plot(radar_figure(A))
        st.caption("Scores run from 0 to 100, higher is better. Verbal Recall is a digit-span proxy.")
    with right:
        st.markdown("**Patient against the age-adjusted normal control band**")
        show_plot(zbar_figure(A))
        st.caption("Bars to the right of zero are worse than the age-matched mean.")


def render_step4(A, ctx):
    step_header(
        4, "Automated clinical process report",
        "What happened during the process, and why the score was generated.",
    )
    n_ok = A["errors"] == 0 and A["verbal_ratio"] >= 0.75
    paper_text = ("The trail was completed in the correct order with no errors and recall was intact. "
                  "On paper this session reads as normal." if n_ok
                  else f"The trail was completed with {A['errors']} wrong tap(s) and digit recall "
                       f"{A['digit_correct']}/{A['digit_length']}. A paper test would show these end results only.")
    proc_text = (f"{A['n_pauses']} hesitation pause(s), mean initiation latency {A['lat']:.0f} ms "
                 f"({A['z_lat']:+.1f} SD), movement variability {A['vel_cv']:.0f}%.")
    c1, c2 = st.columns(2, gap="large")
    c1.markdown(f'<div class="nc-card"><h4>What a paper test records</h4><p>{esc(paper_text)}</p></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="nc-card"><h4>What the process recorded</h4><p>{esc(proc_text)}</p></div>',
                unsafe_allow_html=True)

    st.markdown("**Process timeline: time between taps, hesitations in red**")
    show_plot(timeline_figure(A))

    with st.container(border=True):
        st.markdown("### 📋 AI Behavioral Execution Audit (Process-Level Breakdown)")
        st.markdown(build_audit(A, ctx))


def safe_section(label, func, *args):
    try:
        func(*args)
    except Exception as exc:
        st.error(f"{label} could not be displayed ({type(exc).__name__}: {exc}). "
                 "The rest of the dashboard is unaffected. Try a demo preset from the sidebar.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    ctx = render_sidebar()
    render_hero(ctx)

    safe_section("Step 1", render_step1, ctx)

    raw = resolve_raw(ctx)
    if raw is None:
        st.info("Complete the three tasks above and the analysis appears here. "
                "To preview the dashboard right away, load a demo patient.")
        st.button("Load Preset B demo (early MCI)", on_click=load_preset_b)
        return

    try:
        model, n_train = train_model()
    except Exception:
        model, n_train = None, 0

    try:
        A = analyze(raw, ctx["age"], ctx["edu"], ctx["vasc_yes"], model)
    except Exception as exc:
        st.error(f"The session data could not be analysed ({type(exc).__name__}: {exc}). "
                 "Run the task again or choose a demo preset.")
        st.button("Load Preset B demo (early MCI)", on_click=load_preset_b, key="fallback_preset")
        return

    safe_section("Capture log", render_capture_log, A)
    safe_section("Step 2", render_step2, A, ctx, model, n_train)
    safe_section("Step 3", render_step3, A)
    safe_section("Step 4", render_step4, A, ctx)

    st.markdown(
        '<div class="nc-foot">NeuroCognitive AI is a research prototype for demonstration. It is a screening aid, '
        "not a diagnostic device, and its output must be interpreted by a qualified clinician. Normative values "
        "are illustrative and the machine-learning model is trained on synthetic data.</div>",
        unsafe_allow_html=True,
    )


try:
    main()
except Exception as fatal:  # last-resort guard so a live presentation never shows a raw traceback
    st.error(f"Something went wrong ({type(fatal).__name__}: {fatal}). Reload the page and pick a demo preset.")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
