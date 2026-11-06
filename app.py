"""
NeuroCognitive AI - digital cognitive screening prototype (Streamlit).

Paper tests (MoCA, clock drawing) record only the final static drawing.
This app records HOW the task was performed: touch-initiation latency,
hesitation pauses, stroke velocity, path deviation and recall.

Run locally:   streamlit run app.py
Dependencies:  streamlit, plotly, numpy, pandas, scikit-learn, scipy  (see requirements.txt)

Two capture channels feed one report:
  - touch task (browser-timed): initiation latency, hesitation pauses, stroke velocity, path deviation
  - voice (microphone or WAV): sustained-vowel phonation (jitter, shimmer, HNR) and reading-passage
    speech timing (articulation rate, hesitation pause ratio), combined into a Vocal Biomarker Z-Score

Research prototype for demonstration. Not a medical device and not a diagnostic tool.
"""

import html
import inspect
import io
import math
import tempfile
import traceback
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from scipy import signal as sps
from scipy.io import wavfile
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
# Voice biomarker pipeline (numpy and scipy only, no external speech toolkit)
# ---------------------------------------------------------------------------
PASSAGE = (
    "When the sun rises over the hills, the village slowly wakes. "
    "Farmers carry baskets to the market, and children walk to school along the river. "
    "By noon, the streets are full of voices, and the bakery smells of warm bread."
)
PASSAGE_SYLLABLES = 54      # counted by hand: 15 + 21 + 18
PAUSE_MIN_S = 0.5           # a silent gap longer than this inside the reading counts as a hesitation pause
MAX_AUDIO_S = 60.0

# Age-matched reference values (mean, SD). Prototype values for demonstration; validate before clinical use.
VOICE_NORMS = {
    "45-55": {"jitter": (0.45, 0.20), "shimmer": (2.2, 0.9), "hnr": (22.0, 3.0),
              "art_rate": (5.3, 0.6), "pause_ratio": (0.07, 0.05)},
    "56-65": {"jitter": (0.55, 0.25), "shimmer": (2.6, 1.0), "hnr": (21.0, 3.2),
              "art_rate": (5.1, 0.6), "pause_ratio": (0.09, 0.05)},
    "66-85": {"jitter": (0.70, 0.30), "shimmer": (3.1, 1.2), "hnr": (19.5, 3.5),
              "art_rate": (4.7, 0.6), "pause_ratio": (0.12, 0.06)},
}


def _read_wav_stdlib(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        channels, width, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(float) / 8388608.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128.0) / 128.0
    else:
        raise ValueError("Unsupported WAV sample width")
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, fs


def load_audio(data):
    """Decode WAV bytes to a mono float array in [-1, 1]."""
    if not data:
        raise ValueError("No audio was received.")
    try:
        fs, x = wavfile.read(io.BytesIO(data))
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.dtype == np.uint8:
            x = (x.astype(float) - 128.0) / 128.0
        elif np.issubdtype(x.dtype, np.integer):
            x = x.astype(float) / float(2 ** (8 * x.dtype.itemsize - 1))
        else:
            x = x.astype(float)
    except Exception:
        try:
            x, fs = _read_wav_stdlib(data)
        except Exception:
            raise ValueError("Could not read this audio. Please record again or upload a WAV file.")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    fs = int(fs)
    if fs < 8000:
        raise ValueError("The sample rate is too low for voice analysis (8 kHz minimum).")
    duration = len(x) / fs
    if duration < 1.0:
        raise ValueError("The recording is shorter than 1 second.")
    if duration > MAX_AUDIO_S:
        raise ValueError("The recording is longer than 60 seconds.")
    x = sps.sosfiltfilt(sps.butter(2, 40, "high", fs=fs, output="sos"), x - np.mean(x))
    return x, fs


def _frames(x, n, hop):
    if len(x) < n:
        raise ValueError("The recording is too short to analyse.")
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop]


def _autocorr_frames(x, fs, win_s, hop_s):
    """Window-corrected normalised autocorrelation of every frame (Boersma 1993 style)."""
    n = int(win_s * fs)
    hop = max(1, int(hop_s * fs))
    window = np.hanning(n)
    frames = _frames(x, n, hop) * window
    frames = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2
    ac = np.fft.irfft(power, nfft, axis=1)[:, :n]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    wpow = np.abs(np.fft.rfft(window, nfft)) ** 2
    wac = np.fft.irfft(wpow, nfft)[:n]
    wac = wac / wac[0]
    r = ac / np.maximum(wac, 1e-3)
    times = (np.arange(len(frames)) * hop + n / 2.0) / fs
    return r, times


def _parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = y0 - 2.0 * y1 + y2
    if abs(d) < 1e-15:
        return float(i)
    return i + 0.5 * (y0 - y2) / d


def _pitch_and_strength(r, fs, fmin=70.0, fmax=400.0):
    """Per-frame F0 (Hz) and periodicity strength (0..1) from the autocorrelation."""
    lo = int(fs / fmax)
    hi = min(int(fs / fmin), r.shape[1] - 2)
    f0 = np.full(len(r), np.nan)
    strength = np.zeros(len(r))
    for i in range(len(r)):
        s = r[i, lo:hi + 1]
        pk, _ = sps.find_peaks(s)
        if pk.size == 0:
            continue
        vals = s[pk]
        if vals.max() <= 0:
            continue
        strong = pk[vals >= 0.9 * vals.max()]
        p = int(strong[0])
        lag = _parabolic(s, p) + lo
        if 0 < p < len(s) - 1:
            y0, y1, y2 = s[p - 1], s[p], s[p + 1]
            d = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / d if abs(d) > 1e-15 else 0.0
            value = y1 - 0.25 * (y0 - y2) * delta
        else:
            value = s[p]
        f0[i] = fs / lag
        strength[i] = float(np.clip(value, 0.0, 0.9999))
    return f0, strength


def _longest_run(mask, max_gap=3):
    m = mask.copy()
    idx = np.where(m)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            m[a:b] = True
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(m, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _runs(mask):
    """Return (start, end_exclusive) for each run of True."""
    out = []
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def spectrogram_data(x, fs, hop_s):
    """Pre-emphasised wideband spectrogram up to 5 kHz, in dB."""
    target = 16000
    if fs != target:
        g = math.gcd(int(fs), target)
        x16 = sps.resample_poly(x, target // g, int(fs) // g)
    else:
        x16 = x
    x16 = sps.lfilter([1.0, -0.97], [1.0], x16)
    hop = int(hop_s * target)
    f, t, S = sps.spectrogram(x16, fs=target, window="hann", nperseg=400, noverlap=400 - hop,
                              nfft=512, scaling="spectrum", mode="psd")
    keep = f <= 5000
    db = 10.0 * np.log10(S[keep] + 1e-12)
    return {"f": f[keep].astype(np.float32), "t": t.astype(np.float32), "db": db.astype(np.float32)}


def _quality(x, fs, clip_limit=0.02, snr_min=15.0, snr=None):
    notes = []
    clip_pct = 100.0 * float(np.mean(np.abs(x) >= 0.99 * max(1e-9, np.max(np.abs(x))))) if np.max(np.abs(x)) >= 0.98 else 0.0
    ok = True
    if clip_pct > 100 * clip_limit:
        ok = False
        notes.append(f"Clipping on {clip_pct:.1f}% of samples. Move back from the microphone.")
    if snr is not None and snr < snr_min:
        ok = False
        notes.append(f"Background noise is high (about {snr:.0f} dB signal-to-noise). Record somewhere quieter.")
    return {"ok": ok, "notes": notes, "snr_db": snr, "clip_pct": clip_pct}


def analyze_vowel(data):
    """Sustained vowel: jitter (local, %), shimmer (local, %) and HNR (dB)."""
    x, fs = load_audio(data)
    if fs < 11025:
        raise ValueError("Sample rate too low for jitter analysis (11 kHz minimum).")
    hop_s = 0.01
    r, times = _autocorr_frames(x, fs, 0.06, hop_s)
    f0, strength = _pitch_and_strength(r, fs)
    voiced = np.isfinite(f0) & (strength >= 0.6)
    s, e = _longest_run(voiced, 3)
    run_s = (e - s) * hop_s
    if run_s < 1.5:
        raise ValueError(f"Only {run_s:.1f} s of steady voicing was found. Hold 'Ahhh' steadily for about 5 seconds.")
    margin = 0.4 if run_s >= 3.0 else 0.2
    t0, t1 = times[s] + margin, times[e - 1] - margin
    if t1 - t0 > 3.0:
        mid = 0.5 * (t0 + t1)
        t0, t1 = mid - 1.5, mid + 1.5
    in_win = (times >= t0) & (times <= t1) & voiced
    if in_win.sum() < 10:
        raise ValueError("The steady part of the vowel was too short to analyse.")
    f0_med = float(np.median(f0[in_win]))
    f0_sd = float(np.std(f0[in_win]))

    seg = x[int(t0 * fs):int(t1 * fs)]
    raw = sps.sosfiltfilt(sps.butter(2, 40, "high", fs=fs, output="sos"), seg)
    cut = min(float(np.clip(2.5 * f0_med, 250.0, 1000.0)), 0.45 * fs)
    y = sps.sosfiltfilt(sps.butter(4, cut, "low", fs=fs, output="sos"), raw)
    if abs(np.percentile(y, 1)) > abs(np.percentile(y, 99)):
        y, raw = -y, -raw
    t_samples = fs / f0_med
    pk, _ = sps.find_peaks(y, distance=max(2, int(0.7 * t_samples)))
    if len(pk):
        pk = pk[y[pk] >= 0.4 * np.median(y[pk])]
    if len(pk) < 60:
        raise ValueError("Too few vocal cycles were detected. Record a louder, steadier 'Ahhh'.")
    pos = np.array([_parabolic(y, int(p)) for p in pk])
    periods = np.diff(pos) / fs
    med = float(np.median(periods))
    good = (periods > 0.75 * med) & (periods < 1.3 * med)
    pair_ok = good[:-1] & good[1:]
    if pair_ok.sum() < 40:
        raise ValueError("The pitch was too unstable to measure jitter. Try a steadier 'Ahhh'.")
    d_t = np.abs(np.diff(periods))
    jitter = 100.0 * float(d_t[pair_ok].mean() / periods[good].mean())

    amps = np.array([np.ptp(raw[pk[i]:pk[i + 1]]) for i in range(len(pk) - 1)])
    a1, a2 = amps[:-1], amps[1:]
    ratio_ok = (np.maximum(a1, a2) / np.maximum(np.minimum(a1, a2), 1e-12)) < 1.6
    amp_pairs = pair_ok & ratio_ok
    shimmer = 100.0 * float(np.abs(np.diff(amps))[amp_pairs].mean() / amps[good].mean())

    st_r = strength[in_win]
    hnr_vals = 10.0 * np.log10(np.clip(st_r, 1e-3, 0.9999) / (1.0 - np.clip(st_r, 1e-3, 0.9999)))
    hnr = float(np.mean(hnr_vals))

    fr = _frames(x, int(0.025 * fs), int(0.01 * fs))
    rms = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9
    snr = None
    if np.percentile(rms, 10) < 0.3 * np.median(rms):
        snr = float(20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) / np.percentile(rms, 10)))
    quality = _quality(x, fs, snr=snr, snr_min=15.0)

    keep = np.where(voiced & (times >= times[s]) & (times <= times[e - 1]))[0]
    return {
        "kind": "vowel", "fs": fs, "duration_s": len(x) / fs, "window": (float(t0), float(t1)),
        "f0_hz": f0_med, "f0_sd_hz": f0_sd, "jitter": jitter, "shimmer": shimmer, "hnr": hnr,
        "n_cycles": int(pair_ok.sum()), "quality": quality,
        "f0_track": (times[keep].astype(np.float32), f0[keep].astype(np.float32)),
        "spec": spectrogram_data(x, fs, 0.01),
    }


def analyze_reading(data):
    """Reading passage: articulation rate (syllables/s) and hesitation pause ratio (pauses over 0.5 s)."""
    x, fs = load_audio(data)
    if fs > 16000:
        g = math.gcd(int(fs), 16000)
        x = sps.resample_poly(x, 16000 // g, int(fs) // g)
        fs_in = fs
        fs = 16000
    else:
        fs_in = fs
    edge = int(0.08 * fs)          # drop recorder click and filter edge effects at both ends
    x = x[edge:-edge]
    hop_s = 0.01
    hop = int(hop_s * fs)
    n = int(0.03 * fs)
    fr = _frames(x, n, hop)
    rms = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9
    env_t = (np.arange(len(fr)) * hop + n / 2.0) / fs
    env = 20.0 * np.log10(rms)
    env_s = np.convolve(env, np.ones(3) / 3.0, mode="same")

    p95, p5 = float(np.percentile(env_s, 95)), float(np.percentile(env_s, 5))
    if p95 < -50.0:
        raise ValueError("The recording is almost silent. Check the microphone and read the passage aloud.")
    if p95 - p5 < 15.0:
        raise ValueError("No clear speech was found above the background. Read the passage aloud in a quieter place.")
    act_thr = max(p95 - 30.0, p5 + 8.0)
    active = env_s > act_thr
    runs = _runs(active)
    if not runs:
        raise ValueError("No speech was detected. Check the microphone and read the passage aloud.")
    first, last = runs[0][0], runs[-1][1]
    span = (last - first) * hop_s
    if span < 4.0:
        raise ValueError(f"Only {span:.1f} s of speech was detected. Please read the whole passage.")

    pauses = []
    for (_, e0), (s1, _) in zip(runs[:-1], runs[1:]):
        gap = (s1 - e0) * hop_s
        if gap > PAUSE_MIN_S:
            pauses.append((float(env_t[e0]), float(gap)))
    pause_total = float(sum(g for _, g in pauses))
    art_time = span - pause_total
    if art_time < 2.0:
        raise ValueError("Most of the recording was silence. Please read the passage again.")

    r, r_times = _autocorr_frames(x, fs, 0.04, hop_s)
    _, strength = _pitch_and_strength(r, fs)
    voiced_env = np.interp(env_t, r_times, strength) >= 0.45
    peak_thr = max(p95 - 25.0, act_thr)
    cand, _ = sps.find_peaks(env_s, height=peak_thr, prominence=2.0, distance=6)
    nuclei = [int(i) for i in cand if first <= i < last and voiced_env[max(0, i - 1):i + 2].any()]
    n_detected = len(nuclei)
    completeness = n_detected / float(PASSAGE_SYLLABLES)
    if 0.75 <= completeness <= 1.3:
        rate = PASSAGE_SYLLABLES / art_time
        basis = "passage syllable count, confirmed by the detected syllables"
    else:
        rate = n_detected / art_time
        basis = "detected syllable nuclei"
    ratio = pause_total / span

    inactive = ~active
    snr = None
    if inactive.sum() >= 10:
        snr = float(np.mean(env[active]) - np.median(env[inactive]))
    quality = _quality(x, fs, snr=snr, snr_min=12.0)
    if not 0.6 <= completeness <= 1.4:
        quality["notes"].append(
            f"Detected {n_detected} syllables against {PASSAGE_SYLLABLES} expected, so the passage may not have been "
            "read in full or the room was noisy.")

    return {
        "kind": "reading", "fs": fs_in, "duration_s": len(x) / fs, "span_s": float(span),
        "art_time_s": float(art_time), "art_rate": float(rate), "rate_basis": basis,
        "pause_ratio": float(ratio), "pause_total_s": pause_total, "pauses": pauses,
        "n_pauses": len(pauses), "n_syllables": n_detected, "completeness": float(completeness),
        "nuclei_t": [float(env_t[i]) for i in nuclei], "speech_start_s": float(env_t[first]),
        "speech_end_s": float(env_t[min(last, len(env_t) - 1)]), "quality": quality,
        "spec": spectrogram_data(x, fs, 0.015),
    }


# ---------------------------------------------------------------------------
# Synthetic audio for the demo presets (processed by the same pipeline as live audio)
# ---------------------------------------------------------------------------
def _resonator(x, fs, f, bw):
    r = math.exp(-math.pi * bw / fs)
    th = 2.0 * math.pi * f / fs
    a = [1.0, -2.0 * r * math.cos(th), r * r]
    return sps.lfilter([sum(a)], a, x)


def synth_vowel(fs=44100, dur=5.0, f0=120.0, jitter=0.4, shimmer=2.0, hnr=22.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    lead, tail = 0.35, 0.3
    sj, ss = jitter / 100.0 / 1.128, shimmer / 100.0 / 1.128
    times, amps = [], []
    t = lead
    while t < dur - tail:
        period = (1.0 / f0) * float(np.clip(1.0 + sj * rng.standard_normal(), 0.7, 1.3))
        times.append(t)
        amps.append(max(0.3, 1.0 + ss * rng.standard_normal()))
        t += period
    pos = np.array(times) * fs
    idx = np.floor(pos).astype(int)
    frac = pos - idx
    imp = np.zeros(n + 2)
    np.add.at(imp, idx, np.array(amps) * (1.0 - frac))
    np.add.at(imp, idx + 1, np.array(amps) * frac)
    sig = sps.lfilter([1.0], [1.0, -0.95], imp)
    for f, bw in [(750, 90), (1200, 110), (2600, 170), (3500, 250)]:
        sig = _resonator(sig, fs, f, bw)
    sig = sig[:n]
    env = np.zeros(n)
    i0, i1 = int(lead * fs), int((dur - tail) * fs)
    env[i0:i1] = 1.0
    fade = int(0.04 * fs)
    env[i0:i0 + fade] = np.linspace(0, 1, fade)
    env[i1 - fade:i1] = np.linspace(1, 0, fade)
    sig = sig * env
    power = float(np.mean(sig[i0:i1] ** 2))
    noise = rng.standard_normal(n) * math.sqrt(power / 10 ** (hnr / 10.0)) * env
    out = sig + noise
    out = out / np.max(np.abs(out)) * 0.6
    out = out + rng.standard_normal(n) * 0.0012
    return out


def synth_reading(fs=16000, rate=5.0, gaps=None, f0=120.0, seed=2, n_syll=PASSAGE_SYLLABLES):
    rng = np.random.default_rng(seed)
    gaps = dict(gaps or {})
    slot = 1.0 / rate
    vowels = [(730, 1090), (530, 1840), (270, 2290), (570, 840), (300, 870)]
    pieces = [np.zeros(int(0.5 * fs))]
    for i in range(n_syll):
        dur = 0.8 * slot
        m = int(dur * fs)
        t = np.arange(m) / fs
        f1, f2 = vowels[int(rng.integers(len(vowels)))]
        f1 *= 1.0 + 0.06 * rng.standard_normal()
        f2 *= 1.0 + 0.06 * rng.standard_normal()
        f0_i = f0 * (1.05 - 0.0015 * i + 0.03 * rng.standard_normal())
        f0t = f0_i * (1.0 - 0.06 * t / dur)
        phase = 2.0 * np.pi * np.cumsum(f0t) / fs
        k = np.arange(1, int(min(4800.0 / f0_i, 40)) + 1)[:, None]
        fk = k * f0_i
        gain = np.exp(-0.5 * ((fk - f1) / 180.0) ** 2) + 0.7 * np.exp(-0.5 * ((fk - f2) / 250.0) ** 2) + 0.04
        burst = (gain / k ** 0.8 * np.sin(k * phase[None, :])).sum(axis=0)
        burst = burst / max(np.sqrt(np.mean(burst ** 2)), 1e-9)
        burst = burst * np.sin(np.pi * t / dur) ** 2
        pieces.append(burst)
        pieces.append(np.zeros(int((slot - dur) * fs)))
        if i in gaps:
            pieces.append(np.zeros(int(gaps[i] * fs)))
    pieces.append(np.zeros(int(0.5 * fs)))
    x = np.concatenate(pieces)
    x = x / np.max(np.abs(x)) * 0.5
    return x + rng.standard_normal(len(x)) * 0.0012


def to_wav_bytes(x, fs):
    buf = io.BytesIO()
    wavfile.write(buf, int(fs), (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()

# Demo voice profiles. Audio is synthesised from these settings and then analysed by exactly the
# same code as a live recording.
PRESET_VOICE = {
    PRESET_A: {
        "vowel": dict(f0=125.0, jitter=0.35, shimmer=2.0, hnr=23.0, seed=11),
        "reading": dict(rate=5.6, gaps={14: 0.2, 35: 0.2}, f0=122.0, seed=12),
    },
    PRESET_B: {
        "vowel": dict(f0=118.0, jitter=0.45, shimmer=2.5, hnr=22.0, seed=21),
        "reading": dict(rate=4.0, gaps={6: 0.95, 14: 0.2, 19: 1.25, 28: 0.85, 35: 0.2, 41: 1.35}, f0=116.0, seed=22),
    },
    PRESET_C: {
        "vowel": dict(f0=108.0, jitter=1.2, shimmer=4.8, hnr=15.0, seed=31),
        "reading": dict(rate=3.8, gaps={14: 0.8, 30: 0.85, 35: 0.2, 42: 0.8}, f0=106.0, seed=32),
    },
}


@st.cache_data(show_spinner=False, max_entries=6)
def cached_vowel(data):
    return analyze_vowel(data)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_reading(data):
    return analyze_reading(data)


@st.cache_data(show_spinner=False)
def preset_voice_audio(label):
    p = PRESET_VOICE[label]
    return (to_wav_bytes(synth_vowel(fs=44100, **p["vowel"]), 44100),
            to_wav_bytes(synth_reading(fs=16000, **p["reading"]), 16000))


def run_voice(vowel_bytes, reading_bytes):
    """Analyse whichever recordings exist. Problems become readable messages, never exceptions."""
    result = {"vowel": None, "reading": None, "errors": {}}
    for key, data, func in (("vowel", vowel_bytes, cached_vowel), ("reading", reading_bytes, cached_reading)):
        if not data:
            continue
        try:
            result[key] = func(data)
        except ValueError as exc:
            result["errors"][key] = str(exc)
        except Exception as exc:
            result["errors"][key] = f"The audio could not be analysed ({type(exc).__name__})."
    return result


def zscore_low_bad(value, mean, sd):
    """Z-score where a LOWER value is worse (articulation rate, HNR)."""
    return float(np.clip((mean - value) / sd, Z_MIN, Z_MAX))


def voice_zscores(V, age):
    norms = VOICE_NORMS[cohort_for_age(age)]
    vz = {"norms": norms, "z_phon": None, "z_ling": None, "z_voc": None, "tier_voc": None,
          "phon_used": False, "ling_used": False}
    vow, rd = V.get("vowel"), V.get("reading")
    if vow:
        vz["z_jit"] = zscore(vow["jitter"], *norms["jitter"])
        vz["z_shim"] = zscore(vow["shimmer"], *norms["shimmer"])
        vz["z_hnr"] = zscore_low_bad(vow["hnr"], *norms["hnr"])
        vz["z_phon"] = float(np.mean([vz["z_jit"], vz["z_shim"], vz["z_hnr"]]))
        vz["phon_used"] = bool(vow["quality"]["ok"])
    if rd:
        vz["z_rate"] = zscore_low_bad(rd["art_rate"], *norms["art_rate"])
        vz["z_vpause"] = zscore(rd["pause_ratio"], *norms["pause_ratio"])
        vz["z_ling"] = float(np.mean([vz["z_rate"], vz["z_vpause"]]))
        vz["ling_used"] = bool(rd["quality"]["ok"])
    used = [z for z, flag in ((vz["z_phon"], vz["phon_used"]), (vz["z_ling"], vz["ling_used"]))
            if z is not None and flag]
    if used:
        vz["z_voc"] = float(np.mean(used))
        vz["tier_voc"] = tier_from_z(vz["z_voc"])
    return vz


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


def analyze(raw, age, edu, vasc_yes, model, voice=None):
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
    z_voc, z_multi, tier_multi = None, None, 0
    if voice and voice["vz"]["z_voc"] is not None:
        z_voc = float(voice["vz"]["z_voc"])
        z_multi = (z_comp + z_voc) / 2.0
        tier_multi = tier_from_z(z_multi)
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
        "tier_final": max(tier_z, tier_ml, tier_multi), "vasc_yes": vasc_yes,
        "voice": voice, "z_voc": z_voc, "z_multi": z_multi, "tier_multi": tier_multi,
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
    vz = A["voice"]["vz"] if A["voice"] else None
    if vz:
        zph = vz["z_phon"] if vz["phon_used"] else None
        zli = vz["z_ling"] if vz["ling_used"] else None
        if zli is not None and zli >= 1 and (zph is None or zph < 1) and zl >= 1:
            points.append(
                f"**Cross-modal hesitation.** Slowed, pause-laden speech ({zli:+.1f} SD) appears alongside slowed touch "
                "initiation, while voice quality itself is preserved. Slowing that shows up in both channels without a "
                "phonation problem points to planning-level slowing rather than a laryngeal or speech-motor cause."
            )
        if zph is not None and zph >= 1.5 and zm >= 1.5:
            points.append(
                f"**Shared motor-control signature.** Unstable phonation ({zph:+.1f} SD) and hand path deviation "
                f"(motor z {zm:+.1f}) occur together, which suggests one underlying motor-control disturbance, for example "
                "subcortical or vascular, rather than two unrelated problems."
            )
        if zph is not None and zph >= 2 and zm < 1 and zl < 1:
            points.append(
                f"**Isolated voice abnormality.** Phonation is abnormal ({zph:+.1f} SD) while touch performance is normal. "
                "Laryngeal, respiratory or smoking-related causes should be considered before attributing this to cognition."
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
    voice = A["voice"]
    if voice:
        vv, vzs = voice["V"], voice["vz"]
        vow, rd = vv.get("vowel"), vv.get("reading")
        if vow:
            lines.append(
                f"- **Sustained vowel (phonation):** jitter {vow['jitter']:.2f}% ({vzs['z_jit']:+.1f} SD), "
                f"shimmer {vow['shimmer']:.2f}% ({vzs['z_shim']:+.1f} SD), HNR {vow['hnr']:.1f} dB "
                f"({vzs['z_hnr']:+.1f} SD), median F0 {vow['f0_hz']:.0f} Hz."
                + ("" if vzs["phon_used"] else " Excluded from scoring because of recording quality.")
            )
        if rd:
            lines.append(
                f"- **Reading passage (speech timing):** articulation rate {rd['art_rate']:.1f} syllables/s "
                f"({vzs['z_rate']:+.1f} SD) and a hesitation pause ratio of {100 * rd['pause_ratio']:.0f}% "
                f"({vzs['z_vpause']:+.1f} SD), with {rd['n_pauses']} silent gap(s) longer than {PAUSE_MIN_S:.1f} s."
                + ("" if vzs["ling_used"] else " Excluded from scoring because of recording quality.")
            )
            for k, (start, gap) in enumerate(rd["pauses"][:6], start=1):
                lines.append(f"    - Speech pause {k}: {gap:.2f} s of silence starting {start:.1f} s into the reading.")
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
    if A["z_voc"] is not None:
        lines.append(
            f"The Vocal Biomarker Z-Score is {A['z_voc']:+.1f} SD (the average of the phonation and speech-timing "
            f"branches that passed the quality check). Averaged with the touch composite it gives a multimodal "
            f"deviation of {A['z_multi']:+.1f} SD."
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


SPEC_SCALE = [[0.0, "#FFFFFF"], [0.35, "#BFD5EA"], [0.7, "#3F78B5"], [1.0, "#0F2A44"]]


def _band_figure(rows, height):
    """Horizontal z-score bars over the green / amber / red risk bands. rows = (label, z, text)."""
    labels = [r[0] for r in rows]
    zs = [r[1] for r in rows]
    colors = [TIERS[tier_from_z(z)]["color"] for z in zs]
    fig = go.Figure(go.Bar(
        x=zs, y=labels, orientation="h", marker_color=colors, text=[r[2] for r in rows],
        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:+.2f} SD<extra></extra>",
    ))
    fig.add_vrect(x0=-1, x1=1, fillcolor=GREEN, opacity=0.14, line_width=0, layer="below")
    fig.add_vrect(x0=1, x1=2, fillcolor=AMBER, opacity=0.16, line_width=0, layer="below")
    fig.add_vrect(x0=2, x1=9.5, fillcolor=RED, opacity=0.08, line_width=0, layer="below")
    for x, text in [(0, "Normal control band"), (1.5, "1 to 2 SD"), (4.7, "2 SD or more: alert")]:
        fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=text, showarrow=False,
                           font=dict(size=11, color=MUTED))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(
        font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", showlegend=False,
        xaxis=dict(title="Standard deviations from the age-adjusted normal control mean",
                   range=[-3.5, 9.5], zeroline=False, gridcolor=LINE),
        yaxis=dict(autorange="reversed", gridcolor=LINE, automargin=True),
        margin=dict(l=10, r=10, t=40, b=50), height=height,
    )
    return fig


def zbar_figure(A):
    rows = [
        ("Initiation latency", A["z_lat"], f"{A['lat']:.0f} ms  ({A['z_lat']:+.1f} SD)"),
        ("Hesitation pauses", A["z_pause"], f"{A['n_pauses']} pauses  ({A['z_pause']:+.1f} SD)"),
        ("Velocity variability", A["z_cv"], f"{A['vel_cv']:.0f}% CV  ({A['z_cv']:+.1f} SD)"),
        ("Path deviation", A["z_dev"], f"{A['dev']:.1f} mm  ({A['z_dev']:+.1f} SD)"),
    ]
    return _band_figure(rows, 420)


def multimodal_figure(A):
    rows = [
        ("Touch: initiation latency", A["z_lat"], f"{A['lat']:.0f} ms  ({A['z_lat']:+.1f} SD)"),
        ("Touch: hesitation pauses", A["z_pause"], f"{A['n_pauses']} pauses  ({A['z_pause']:+.1f} SD)"),
    ]
    if A["voice"]:
        vz = A["voice"]["vz"]
        if vz["z_phon"] is not None:
            tag = "" if vz["phon_used"] else " (excluded)"
            rows.append((f"Voice: phonation{tag}", vz["z_phon"], f"{vz['z_phon']:+.1f} SD"))
        if vz["z_ling"] is not None:
            tag = "" if vz["ling_used"] else " (excluded)"
            rows.append((f"Voice: speech timing{tag}", vz["z_ling"], f"{vz['z_ling']:+.1f} SD"))
    if A["z_voc"] is not None:
        rows.append(("Vocal Biomarker Z-Score", A["z_voc"], f"{A['z_voc']:+.1f} SD"))
        rows.append(("Multimodal Z (touch + voice)", A["z_multi"], f"{A['z_multi']:+.1f} SD"))
    return _band_figure(rows, 90 + 50 * len(rows))


def latency_figure(A):
    mean, sd = A["norms"]["latency_ms"]
    trials = [float(v) for v in A["latencies"]]
    labels = [f"Trial {i + 1}" for i in range(len(trials))]
    colors = [TIERS[tier_from_z((v - mean) / sd)]["color"] for v in trials]
    top = max(max(trials) * 1.25, mean + 2.6 * sd)
    fig = go.Figure(go.Bar(
        x=labels, y=trials, marker_color=colors, text=[f"{v:.0f} ms" for v in trials], textposition="outside",
        cliponaxis=False, hovertemplate="%{x}: %{y:.0f} ms<extra></extra>",
    ))
    fig.add_hrect(y0=mean - sd, y1=mean + sd, fillcolor=GREEN, opacity=0.15, line_width=0, layer="below")
    fig.add_hrect(y0=mean + sd, y1=mean + 2 * sd, fillcolor=AMBER, opacity=0.18, line_width=0, layer="below")
    fig.add_hrect(y0=mean + 2 * sd, y1=top * 2, fillcolor=RED, opacity=0.07, line_width=0, layer="below")
    fig.add_hline(y=mean, line_dash="dot", line_color=INK, line_width=1,
                  annotation_text=f"Cohort mean {mean:.0f} ms", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=INK))
    fig.update_layout(
        font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", showlegend=False,
        yaxis=dict(title="Initiation latency (ms)", range=[0, top], gridcolor=LINE),
        xaxis=dict(title="Reaction trial"),
        margin=dict(l=10, r=10, t=20, b=50), height=420,
    )
    return fig


def spectrogram_figure(V):
    items = []
    if V.get("vowel"):
        items.append(("vowel", "Sustained vowel 'Ahhh'"))
    if V.get("reading"):
        items.append(("reading", "Reading passage"))
    if not items:
        return None
    fig = make_subplots(rows=len(items), cols=1, subplot_titles=[t for _, t in items], vertical_spacing=0.18)
    for row, (key, _) in enumerate(items, start=1):
        data = V[key]
        spec = data["spec"]
        top = float(np.max(spec["db"]))
        fig.add_trace(go.Heatmap(
            x=spec["t"], y=spec["f"], z=spec["db"], zmin=top - 60, zmax=top, colorscale=SPEC_SCALE,
            showscale=(row == 1), colorbar=dict(title="dB", thickness=10, len=0.8),
            hovertemplate="%{x:.2f} s, %{y:.0f} Hz, %{z:.0f} dB<extra></extra>",
        ), row=row, col=1)
        if key == "vowel":
            f_t, f_hz = data["f0_track"]
            fig.add_trace(go.Scatter(x=f_t, y=f_hz, mode="lines", line=dict(color=AMBER, width=2),
                                     showlegend=False, hovertemplate="F0 %{y:.0f} Hz<extra></extra>"),
                          row=row, col=1)
            fig.add_vrect(x0=data["window"][0], x1=data["window"][1], line_width=1.5, line_dash="dash",
                          line_color=AMBER, fillcolor="rgba(0,0,0,0)", row=row, col=1)
        else:
            for start, gap in data["pauses"]:
                fig.add_vrect(x0=start, x1=start + gap, fillcolor=RED, opacity=0.28, line_width=0,
                              annotation_text=f"{gap:.1f} s", annotation_position="top left",
                              annotation_font=dict(size=10, color=RED), row=row, col=1)
            nuclei = data["nuclei_t"]
            fig.add_trace(go.Scatter(x=nuclei, y=[4750] * len(nuclei), mode="markers",
                                     marker=dict(symbol="triangle-down", size=5, color=INK), showlegend=False,
                                     hovertemplate="Syllable at %{x:.2f} s<extra></extra>"), row=row, col=1)
        fig.update_xaxes(title_text="Time (s)", row=row, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", range=[0, 5000], row=row, col=1)
    fig.update_layout(font=FONT, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      margin=dict(l=10, r=10, t=40, b=40), height=250 * len(items) + 90)
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
        st.info("A demo preset is loaded, so the dashboard shows its recorded touch and voice session. "
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


def capture_audio(label, key, rate):
    """Microphone recorder with a WAV-upload fallback. Returns WAV bytes or None."""
    data = None
    if hasattr(st, "audio_input"):
        try:
            recorded = st.audio_input(label, sample_rate=rate, key=key)
        except TypeError:
            recorded = st.audio_input(label, key=key)
        if recorded is not None:
            data = recorded.getvalue()
    else:
        st.caption("Microphone recording needs Streamlit 1.39 or newer. Upload a WAV file instead.")
    with st.expander("Upload a WAV file instead"):
        uploaded = st.file_uploader("WAV file", type=["wav"], key=key + "_upload", label_visibility="collapsed")
        if uploaded is not None:
            data = uploaded.getvalue()
    return data


def _voice_band(z, used):
    if not used:
        return ('<span class="nc-badge" style="background:#EEF2F5;color:#5A6F82;border:1px solid #D5DFE8">'
                "Excluded</span>")
    tier = tier_from_z(z)
    return badge(tier, {0: "Low", 1: "Borderline", 2: "Alert"}[tier])


def render_voice_results(V, vz):
    norms = vz["norms"]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Acoustic Phonation Analyzer**")
        vow = V.get("vowel")
        if vow:
            used = vz["phon_used"]
            rows = [
                ["Jitter (local)", f"{vow['jitter']:.2f}%", f"{norms['jitter'][0]:.2f} ± {norms['jitter'][1]:.2f}",
                 f"{vz['z_jit']:+.1f}", _voice_band(vz["z_jit"], used)],
                ["Shimmer (local)", f"{vow['shimmer']:.2f}%", f"{norms['shimmer'][0]:.1f} ± {norms['shimmer'][1]:.1f}",
                 f"{vz['z_shim']:+.1f}", _voice_band(vz["z_shim"], used)],
                ["Harmonic-to-noise ratio", f"{vow['hnr']:.1f} dB", f"{norms['hnr'][0]:.1f} ± {norms['hnr'][1]:.1f}",
                 f"{vz['z_hnr']:+.1f}", _voice_band(vz["z_hnr"], used)],
            ]
            st.markdown(html_table(["Metric", "Patient", "Cohort", "Z", "Band"], rows), unsafe_allow_html=True)
            st.markdown(
                f'<div class="nc-note">Median pitch {vow["f0_hz"]:.0f} Hz, {vow["n_cycles"]} vocal cycles analysed '
                f'from the steady middle of the vowel. Cut-offs often cited for jitter and shimmer are about 1.04% '
                f'and 3.81%.</div>', unsafe_allow_html=True)
            for note in vow["quality"]["notes"]:
                st.warning(note)
        else:
            st.caption("No usable vowel recording yet.")
    with right:
        st.markdown("**Linguistic Temporal Analyzer**")
        rd = V.get("reading")
        if rd:
            used = vz["ling_used"]
            pm, ps = norms["pause_ratio"]
            rows = [
                ["Articulation rate", f"{rd['art_rate']:.1f} syll/s",
                 f"{norms['art_rate'][0]:.1f} ± {norms['art_rate'][1]:.1f}", f"{vz['z_rate']:+.1f}",
                 _voice_band(vz["z_rate"], used)],
                ["Hesitation pause ratio", f"{100 * rd['pause_ratio']:.0f}%", f"{100 * pm:.0f} ± {100 * ps:.0f}%",
                 f"{vz['z_vpause']:+.1f}", _voice_band(vz["z_vpause"], used)],
            ]
            st.markdown(html_table(["Metric", "Patient", "Cohort", "Z", "Band"], rows), unsafe_allow_html=True)
            st.markdown(
                f'<div class="nc-note">{rd["n_pauses"]} silent gap(s) over {PAUSE_MIN_S:.1f} s, {rd["pause_total_s"]:.1f} s '
                f'in total. {rd["n_syllables"]} of {PASSAGE_SYLLABLES} syllables detected; the rate uses the '
                f'{esc(rd["rate_basis"])}.</div>', unsafe_allow_html=True)
            for note in rd["quality"]["notes"]:
                st.warning(note)
        else:
            st.caption("No usable reading recording yet.")

    if vz["z_voc"] is not None:
        tier = TIERS[vz["tier_voc"]]
        parts = []
        if vz["phon_used"]:
            parts.append(f"phonation {vz['z_phon']:+.1f} SD")
        if vz["ling_used"]:
            parts.append(f"speech timing {vz['z_ling']:+.1f} SD")
        st.markdown(
            f'<div class="nc-banner" style="background:{tier["bg"]};border-left-color:{tier["color"]}">'
            f'<div class="t" style="color:{tier["fg"]}">{tier["icon"]} Vocal Biomarker Z-Score: {vz["z_voc"]:+.1f} SD</div>'
            f'<div class="d">{esc(tier["name"])}. Average of {" and ".join(parts)}, each branch weighted equally.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("The Vocal Biomarker Z-Score appears once at least one recording passes the quality check.")

    with st.expander("Voice normative baseline matrix"):
        rows = [[c, f"{v['jitter'][0]:.2f} ± {v['jitter'][1]:.2f}", f"{v['shimmer'][0]:.1f} ± {v['shimmer'][1]:.1f}",
                 f"{v['hnr'][0]:.1f} ± {v['hnr'][1]:.1f}", f"{v['art_rate'][0]:.1f} ± {v['art_rate'][1]:.1f}",
                 f"{100 * v['pause_ratio'][0]:.0f} ± {100 * v['pause_ratio'][1]:.0f}"] for c, v in VOICE_NORMS.items()]
        active = [c == cohort_for_age(int(st.session_state.get("age", 68))) for c, *_ in rows]
        st.markdown(html_table(["Age cohort", "Jitter (%)", "Shimmer (%)", "HNR (dB)", "Articulation rate (syll/s)",
                                "Pause ratio (%)"], rows, active), unsafe_allow_html=True)
        st.markdown('<div class="nc-note">Mean ± SD. Prototype reference values for demonstration; they need validation '
                    "against recordings from a clinical cohort made with the same microphone setup.</div>",
                    unsafe_allow_html=True)


def render_voice_step(ctx):
    step_header(
        2, "Voice biomarker analysis",
        "Two branches: a sustained 'Ahhh' measures voice stability, and a short reading measures speech timing. "
        "Both are combined into one Vocal Biomarker Z-Score.",
    )
    if ctx["preset"] in PRESET_VOICE:
        vowel_bytes, reading_bytes = preset_voice_audio(ctx["preset"])
        st.info("Demo preset: this audio is synthetic, generated for the patient profile and analysed by the same "
                "pipeline as a live recording.")
        c1, c2 = st.columns(2, gap="large")
        c1.markdown("**Sustained vowel (about 5 s)**")
        c1.audio(vowel_bytes, format="audio/wav")
        c2.markdown("**Reading passage**")
        c2.audio(reading_bytes, format="audio/wav")
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Branch 1: sustained vowel**")
            st.caption("Sit upright, hold the device about 20 cm away, take a breath and say “Ahhh” at a steady, "
                       "comfortable pitch and loudness for about 5 seconds.")
            vowel_bytes = capture_audio("Record the sustained vowel", "voice_vowel", 44100)
        with c2:
            st.markdown("**Branch 2: reading passage**")
            with st.container(border=True):
                st.markdown(PASSAGE)
            reading_bytes = capture_audio("Record yourself reading the passage", "voice_reading", 16000)
        if not vowel_bytes and not reading_bytes:
            st.caption("Voice is optional. Without a recording the report uses the touch task alone.")

    V = run_voice(vowel_bytes, reading_bytes)
    for key, label in (("vowel", "Vowel recording"), ("reading", "Reading recording")):
        if key in V["errors"]:
            st.warning(f"{label}: {V['errors'][key]}")
    vz = voice_zscores(V, ctx["age"])
    if V["vowel"] or V["reading"]:
        render_voice_results(V, vz)
    return V, vz


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
        3, "Normative comparison and Z-score engine",
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
        if A["voice"]:
            vzs = A["voice"]["vz"]
            for label, z, used in (("Voice: phonation branch", vzs["z_phon"], vzs["phon_used"]),
                                   ("Voice: speech-timing branch", vzs["z_ling"], vzs["ling_used"]),
                                   ("Vocal Biomarker Z-Score", vzs["z_voc"], True)):
                if z is not None:
                    rows.append([label, "", "", f"{z:+.1f}", _voice_band(z, used)])
        st.markdown(html_table(["Metric", "Patient", "Cohort", "Z", "Band"], rows), unsafe_allow_html=True)
        st.markdown(
            '<div class="nc-note">Z below 1.0 SD is low risk, 1.0 to 2.0 SD is borderline, and 2.0 SD or more raises '
            "an alert. Z-scores are limited to the range -3 to +6.</div>",
            unsafe_allow_html=True,
        )

    final = TIERS[A["tier_final"]]
    voice_sentence = ""
    if A["z_voc"] is not None:
        voice_sentence = (f"Vocal Biomarker Z-Score {A['z_voc']:+.1f} SD, multimodal deviation {A['z_multi']:+.1f} SD "
                          f"({esc(TIERS[A['tier_multi']]['name'])}). ")
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
        f"{voice_sentence}The alert level is the highest of the available engines.</div></div>",
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
        4, "Clinical diagnostic dashboard",
        "Key results first, then performance across five clinical domains and against the age-adjusted control band.",
    )
    z_tier = TIERS[A["tier_z"]]["name"]
    lat_mean = A["norms"]["latency_ms"][0]
    p_mean = A["norms"]["pauses"][0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Score", f"{A['risk']:.0f}%", delta=TIERS[A["tier_ml"]]["name"], delta_color="off")
    c2.metric("Mean Latency", f"{A['lat']:.0f} ms", delta=f"{A['lat'] - lat_mean:+.0f} ms vs cohort", delta_color="inverse")
    c3.metric("Hesitation Pauses", f"{A['n_pauses']}", delta=f"{A['n_pauses'] - p_mean:+.1f} vs cohort", delta_color="inverse")
    c4.metric("Z-Score Deviation", f"{A['z_comp']:+.1f} SD", delta=z_tier, delta_color="off")
    if A["z_voc"] is not None:
        c5.metric("Vocal Biomarker Z", f"{A['z_voc']:+.1f} SD", delta=TIERS[A["voice"]["vz"]["tier_voc"]]["name"],
                  delta_color="off")
    else:
        c5.metric("Vocal Biomarker Z", "Not recorded", delta="Record voice in Step 2", delta_color="off")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Performance across five clinical domains**")
        show_plot(radar_figure(A))
        st.caption("Scores run from 0 to 100, higher is better. Verbal Recall is a digit-span proxy.")
    with right:
        st.markdown("**Patient against the age-adjusted normal control band**")
        show_plot(zbar_figure(A))
        st.caption("Bars to the right of zero are worse than the age-matched mean.")

    render_multimodal(A)


def render_multimodal(A):
    st.markdown("#### Voice spectrogram alongside touch-screen motor latency")
    V = A["voice"]["V"] if A["voice"] else None
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("**Voice spectrograms**")
        fig = spectrogram_figure(V) if V else None
        if fig is None:
            st.info("No voice recording yet. Record the vowel and the reading passage in Step 2 to see the "
                    "spectrograms here.")
        else:
            show_plot(fig)
            st.caption("Darker means more energy. Amber line: pitch track, with the dashed box marking the analysed "
                       "window. Red bands: speech pauses over 0.5 s. Triangles: detected syllables.")
    with right:
        st.markdown("**Touch-screen motor latency**")
        show_plot(latency_figure(A))
        st.caption("Each bar is one reaction trial. Green is within 1 SD of the age-matched mean, amber 1 to 2 SD, "
                   "red beyond.")
    st.markdown("**Multimodal Z-scores: touch and voice together**")
    show_plot(multimodal_figure(A))
    st.caption("The multimodal score averages the touch composite (latency and pauses) with the Vocal Biomarker "
               "Z-Score. Voice branches that fail the recording-quality check are left out.")


def render_step4(A, ctx):
    step_header(
        5, "Automated clinical process report",
        "What happened during the process, and why the score was generated.",
    )
    n_ok = A["errors"] == 0 and A["verbal_ratio"] >= 0.75
    paper_text = ("The trail was completed in the correct order with no errors and recall was intact. "
                  "On paper this session reads as normal." if n_ok
                  else f"The trail was completed with {A['errors']} wrong tap(s) and digit recall "
                       f"{A['digit_correct']}/{A['digit_length']}. A paper test would show these end results only.")
    proc_text = (f"{A['n_pauses']} hesitation pause(s), mean initiation latency {A['lat']:.0f} ms "
                 f"({A['z_lat']:+.1f} SD), movement variability {A['vel_cv']:.0f}%.")
    if A["z_voc"] is not None:
        proc_text += f" Vocal Biomarker Z-Score {A['z_voc']:+.1f} SD."
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

    try:
        V, vz = render_voice_step(ctx)
    except Exception as exc:
        st.error(f"Step 2 (voice) could not be displayed ({type(exc).__name__}: {exc}). "
                 "The touch analysis below is unaffected.")
        V, vz = None, None
    voice = {"V": V, "vz": vz} if (V and (V.get("vowel") or V.get("reading"))) else None

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
        A = analyze(raw, ctx["age"], ctx["edu"], ctx["vasc_yes"], model, voice)
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
        "are illustrative, the machine-learning model is trained on synthetic data, and demo-preset audio is synthetic.</div>",
        unsafe_allow_html=True,
    )


try:
    main()
except Exception as fatal:  # last-resort guard so a live presentation never shows a raw traceback
    st.error(f"Something went wrong ({type(fatal).__name__}: {fatal}). Reload the page and pick a demo preset.")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
