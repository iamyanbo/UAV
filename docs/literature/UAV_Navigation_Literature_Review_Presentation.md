---
marp: true
html: true
size: 16:9
paginate: true
theme: default
math: false
---

<style>
  :root{
    --bg:#ffffff; --paper:#fbfcfe; --panel:#f6f7fb; --panel2:#ffffff;
    --line:#d7dde7; --soft:#e8ebf2;
    --fg:#1b2433; --muted:#49546a; --faint:#7d8798;
    --blue:#1d4e89; --cyan:#1d4e89; --amber:#9a6a1a; --good:#2e7d4f; --bad:#b03a3a;
  }

  section{
    color:var(--fg);
    font-family:'Helvetica Neue',Arial,'Segoe UI',system-ui,sans-serif;
    font-size:17px; line-height:1.5;
    background: linear-gradient(180deg, var(--paper) 0%, var(--bg) 45%), var(--bg);
    padding: 58px 76px 64px 76px;
  }

  h1{ font-family: Georgia,'Times New Roman',serif; font-weight:700; font-size:40px; line-height:1.12; letter-spacing:-.01em; margin:0 0 12px 0; color:var(--fg); text-wrap:balance; }
  h2{ font-family: Georgia,'Times New Roman',serif; font-weight:700; font-size:25px; line-height:1.15; margin:0 0 8px 0; color:var(--fg); }
  h3{ font-family: Georgia,'Times New Roman',serif; font-weight:600; font-size:16px; margin:0 0 6px 0; color:var(--fg); }
  p{ margin:0 0 10px 0; color:var(--muted); }
  a{ color:var(--blue); text-decoration:none; }
  em{ color:var(--fg); font-style:italic; }
  b, strong{ color:var(--fg); font-weight:600; }
  code{ font-family:'Courier New',ui-monospace,Consolas,monospace; font-size:.88em; color:var(--blue); }
  .mono{ font-family:'Courier New',ui-monospace,Consolas,monospace; font-size:.88em; color:var(--fg); }

  .kicker{ display:flex; align-items:center; gap:10px; font-size:12px; font-weight:600; letter-spacing:.05em;
    text-transform:uppercase; color:var(--faint); margin-bottom:12px; }
  .kicker::before{ content:""; width:28px; height:2px; background:var(--blue); }
  .lede{ font-size:18px; color:var(--muted); max-width:68ch; margin:0 0 6px 0; line-height:1.55; }
  .sub{ font-size:22px; color:var(--fg); font-weight:500; max-width:56ch; line-height:1.4; margin-bottom:12px; }
  .tiny{ font-size:12px; color:var(--faint); }
  .accent{ color:var(--blue); }
  .warm{ color:var(--amber); }

  .rule{ height:1px; background:var(--line); margin:14px 0 20px 0; }

  .card{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  .kpill{ display:inline-block; font-size:11px; font-weight:600; letter-spacing:.04em;
    color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:2px 9px; margin-bottom:8px; }
  .kpill.cyan{ color:var(--blue); border-color:rgba(29,78,137,.45); }
  .kpill.amber{ color:var(--amber); border-color:rgba(154,106,26,.45); }
  .kpill.good{ color:var(--good); border-color:rgba(46,125,79,.45); }

  .insight{
    display:flex; gap:12px; align-items:flex-start;
    border:1px solid var(--line); border-left:3px solid var(--amber);
    border-radius:10px; background:#fdfbf6;
    padding:12px 16px; margin-top:16px; color:var(--muted); font-size:15px; line-height:1.5;
  }
  .insight.good{ border-left-color:var(--good); background:#f7fbf8; }
  .insight.warn{ border-left-color:var(--bad); background:#fcf7f7; }

  .grid2{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .grid3{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
  .two-col{ display:grid; grid-template-columns:1.05fr .95fr; gap:28px; align-items:center; }
  .dpair{ display:flex; flex-direction:column; gap:16px; }

  ul{ margin:4px 0 6px 0; padding-left:0; list-style:none; }
  li{ margin:0 0 8px 0; padding-left:18px; position:relative; color:var(--muted); font-size:15.5px; line-height:1.5; }
  li::before{ content:""; position:absolute; left:2px; top:.60em; width:6px; height:6px; background:var(--blue); opacity:.5; }
  li b{ color:var(--fg); }

  table{ border-collapse:collapse; width:100%; margin:4px 0 8px 0; }
  th{ font-size:11px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--faint);
    text-align:left; padding:7px 12px 7px 12px; border-bottom:1.5px solid var(--line); }
  td{ font-size:14px; line-height:1.45; color:var(--muted); padding:7px 12px; border-bottom:1px solid var(--soft); vertical-align:top; }
  td b{ color:var(--fg); font-weight:600; }
  tr:last-child td{ border-bottom:none; }

  .diagram{ width:auto; display:block; }
  .diagram text{ font-family:'Helvetica Neue',Arial,sans-serif; }

  section::after{ content:" " attr(data-marpit-pagination) " / 20"; position:absolute; right:60px; bottom:22px;
    font-size:10px; letter-spacing:.08em; color:var(--faint); }
  .deck-footer{ position:absolute; left:76px; right:76px; bottom:40px;
    font-size:10px; letter-spacing:.03em; text-transform:uppercase; color:var(--faint);
    display:flex; justify-content:space-between; }
</style>

<!-- ========== 1 · TITLE ========== -->

<!-- _class: lead -->

<div class="kicker">Literature review · UAV navigation</div>

# UAV Navigation

### World models · vision–language agents · 3D Gaussian scene memory

<p class="lede" style="font-size:19px">A short survey of where the field stands: the mechanisms that exist, what they actually establish, and the architecture that is still missing for aerial autonomy.</p>


---
<!-- ========== 2 · THE SHIFT ========== -->

<div class="kicker">01 · The shift</div>

# Autonomy is moving toward predictive agents

<p class="sub">The field is not converging on a single architecture. Capabilities are maturing in parallel, but the systems that actually fly today rest on classical geometric autonomy; the learned predictive stack reviewed here has not yet been deployed at scale.</p>

<div class="rule"></div>

<table style="margin-top:6px">
  <tr><th style="width:26%">Capability</th><th>Where it stands</th></tr>
  <tr><td><b>Vision–language navigation</b></td><td>models interpret instructions and select actions, but performance drops sharply on <b>unseen scenes</b> and long-horizon tasks</td></tr>
  <tr><td><b>World models</b></td><td>latent and generative models support planning from imagined futures, but <b>prediction is not control sufficiency</b></td></tr>
  <tr><td><b>Aerial world–action models</b></td><td>predict future geometry, views, or trajectories together with actions; <b>language grounding remains a separate problem</b></td></tr>
  <tr><td><b>3D scene memory</b></td><td>Gaussian Splatting provides view synthesis, semantic mapping, and simulation, but a visually convincing map is <b>not collision geometry</b></td></tr>
  <tr><td><b>Asynchronous autonomy</b></td><td>fast/slow branches and action-chunk correction target inference delay, yet <b>action age</b> is still rarely measured</td></tr>
  <tr><td><b>Safety &amp; uncertainty</b></td><td>barrier functions, certified maps, and calibrated action sets exist but are <b>rarely integrated</b> with learned planning</td></tr>
</table>

<div class="insight">Each capability has advanced largely in isolation; the central research problem is to connect them into one working system.</div>

---
<!-- ========== 3 · CENTRAL PROBLEM ========== -->

<div class="kicker">02 · The problem</div>

# The central research problem

<p class="sub">The goal is not simply to combine a <b>VLM, a JEPA, and a drone</b>. It is to learn a <b>predictive state</b> that preserves what the task depends on: geometry, ego-motion, visibility, identity, the consequences of actions, uncertainty, and the language-conditioned goal, while the aircraft keeps moving and observations arrive late.</p>

<div class="rule"></div>

<div class="two-col">
  <div>
    <svg class="diagram" width="520" height="360" viewBox="0 0 560 420" xmlns="http://www.w3.org/2000/svg">
      <style>.bx{fill:#f6f7fb;stroke:#d7dde7} .lb{fill:#49546a;font-size:11.5px} .hdL{fill:#1d4e89;font-size:11px;font-weight:600;letter-spacing:.1em} .hdR{fill:#9a6a1a;font-size:11px;font-weight:600;letter-spacing:.1em} .ar{stroke:#7d8798;stroke-width:1.6;fill:none}</style>
      <rect class="bx" x="14" y="14" width="252" height="160" rx="12" stroke="#1d4e89"/>
      <text x="30" y="38" class="hdL">SLOW · DELIBERATE</text>
      <text x="30" y="60" class="lb">VLM world-model planner</text>
      <text x="30" y="82" class="lb">subgoals · task costs</text>
      <text x="30" y="100" class="lb">memory queries</text>
      <text x="30" y="118" class="lb">reveal &amp; information-seeking</text>
      <text x="30" y="136" class="lb">long-horizon cost</text>
      <rect class="bx" x="294" y="14" width="252" height="160" rx="12" stroke="#9a6a1a"/>
      <text x="310" y="38" class="hdR">FAST · REACTIVE</text>
      <text x="310" y="60" class="lb">low-latency flight control</text>
      <text x="310" y="82" class="lb">stabilization</text>
      <text x="310" y="100" class="lb">local obstacle response</text>
      <text x="310" y="118" class="lb">tracking corrections</text>
      <text x="310" y="136" class="lb">safe fallback</text>
      <rect class="bx" x="96" y="206" width="368" height="52" rx="10"/>
      <text x="280" y="228" text-anchor="middle" fill="#1b2433" font-size="12" font-weight="600">SHARED TIME-INDEXED STATE</text>
      <text x="280" y="246" text-anchor="middle" class="lb">arbitration · time is part of the state</text>
      <rect class="bx" x="96" y="290" width="368" height="50" rx="10" stroke="#d7dde7" stroke-dasharray="5 4"/>
      <text x="280" y="312" text-anchor="middle" fill="#2e7d4f" font-size="11.5" font-weight="600">PERSISTENT 3DGS SCENE MEMORY</text>
      <text x="280" y="329" text-anchor="middle" class="lb">appearance · geometry · uncertainty · dynamics, held separately</text>
      <rect x="186" y="380" width="188" height="34" rx="17" fill="#ffffff" stroke="#1d4e89"/>
      <text x="280" y="401" text-anchor="middle" fill="#1b2433" font-size="11.5" font-weight="600">APPLIED FLIGHT ACTION</text>
      <line x1="280" y1="308" x2="280" y2="380" class="ar"/>
      <line x1="280" y1="258" x2="280" y2="290" class="ar"/>
      <line x1="280" y1="174" x2="280" y2="206" class="ar"/>
    </svg>
  </div>
  <div>
    <p style="font-size:16px; line-height:1.65">A practical frame is a <b>two-timescale agent</b>:</p>
    <ul>
      <li>A fast reactive controller handles <b>immediate flight</b>.</li>
      <li>A slower, VLM-configured planner thinks <b>ahead</b> by choosing subgoals, seeking information, and estimating long-horizon cost; it supervises the fast branch or distills policy into it.</li>
      <li>A persistent scene memory supports both branches, keeping appearance, geometry, uncertainty, and dynamics as <b>separate quantities</b>.</li>
    </ul>
    <div class="insight" style="margin-top:14px">Often the best action reveals information rather than simply reducing the distance to the goal. This is a central feature of aerial search.</div>
  </div>
</div>

---
<!-- ========== 4 · MISSIONS ========== -->

<div class="kicker">03 · Missions</div>

# Tracking, navigation, and exploration set different deadlines

<p class="lede">These are some motivating examples of missions a UAV might be asked to carry out.</p>

<div class="rule"></div>

<table>
  <tr><th style="width:19%">Mission</th><th style="width:36%">What success means</th><th>What must be fast</th></tr>
  <tr><td><b>Navigation</b></td><td>reach the correct destination under instruction, time, and safety constraints</td><td>get the decisive evidence before the route choice becomes costly to reverse</td></tr>
  <tr><td><b>Tracking</b></td><td>keep following the correct moving target, including through temporary loss of view</td><td>motion and perception timing that preserve a reachable recovery view; belief that updates before the target slips away</td></tr>
  <tr><td><b>Exploration</b></td><td>discover trustworthy space within a resource budget and keep a feasible return route</td><td>belief updated on independent evidence before an inferred connection is trusted</td></tr>
</table>

<div class="insight">Latency matters in every mission, but the deadline differs: route reversibility for navigation, target recoverability for tracking, and evidence independence for exploration.</div>

---
<!-- ========== 5 · TIMING MATH ========== -->

<div class="kicker">04 · Timing math</div>

# How fast must the models run?

<p class="lede">How often the model must update follows from <b>f = v / ε</b> (in Hz, for ground speed in m/s and a motion budget ε in metres): speed sets the x-axis, and ε decides how often you must update. The line below is drawn at ε = 1.0 m.</p>

<div class="rule"></div>

<svg class="diagram" width="1160" height="248" viewBox="0 0 1160 248" xmlns="http://www.w3.org/2000/svg" style="margin-top:2px">
  <style>.l{font-family:'Helvetica Neue',Arial,sans-serif} .lab{fill:#7d8798;font-size:11.5px} .gd{stroke:#eef1f6;stroke-width:1} .ax{stroke:#c9d2de;stroke-width:1.2}</style>
  <!-- control-loop band (50-60 Hz) -->
  <rect x="70" y="40" width="1040" height="14" fill="#eef3f8"/>
  <text x="76" y="53" class="l lab" fill="#5a6b85">control loop 50–200 Hz</text>
  <!-- camera 30 Hz -->
  <line x1="70" y1="120" x2="1110" y2="120" stroke="#9a6a1a" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="896" y="114" class="l lab" fill="#9a6a1a">camera ≈ 30 Hz</text>
  <!-- slow-branch band (0.5-2.5 Hz) -->
  <rect x="70" y="193.5" width="1040" height="5.5" fill="#eef6ef"/>
  <text x="76" y="200.5" class="l lab" fill="#2e7d4f">slow branch 1–2 Hz</text>
  <!-- vertical gridlines + x ticks -->
  <line x1="330" y1="40" x2="330" y2="200" class="gd"/><line x1="590" y1="40" x2="590" y2="200" class="gd"/>
  <line x1="850" y1="40" x2="850" y2="200" class="gd"/><line x1="1110" y1="40" x2="1110" y2="200" class="gd"/>
  <text x="70" y="214" class="l lab" text-anchor="middle">0</text><text x="330" y="214" class="l lab" text-anchor="middle">10</text>
  <text x="590" y="214" class="l lab" text-anchor="middle">20</text><text x="850" y="214" class="l lab" text-anchor="middle">30</text>
  <text x="1110" y="214" class="l lab" text-anchor="middle">40</text>
  <!-- horizontal gridlines + y ticks -->
  <line x1="70" y1="173.3" x2="1110" y2="173.3" class="gd"/><text x="64" y="177" class="l lab" text-anchor="end">10</text>
  <line x1="70" y1="146.7" x2="1110" y2="146.7" class="gd"/><text x="64" y="150" class="l lab" text-anchor="end">20</text>
  <line x1="70" y1="120" x2="1110" y2="120" class="gd"/><text x="64" y="124" class="l lab" text-anchor="end">30</text>
  <line x1="70" y1="93.3" x2="1110" y2="93.3" class="gd"/><text x="64" y="97" class="l lab" text-anchor="end">40</text>
  <line x1="70" y1="66.7" x2="1110" y2="66.7" class="gd"/><text x="64" y="70" class="l lab" text-anchor="end">50</text>
  <line x1="70" y1="40" x2="1110" y2="40" class="gd"/><text x="64" y="44" class="l lab" text-anchor="end">60</text>
  <!-- axes -->
  <line x1="70" y1="200" x2="1110" y2="200" class="ax"/>
  <line x1="70" y1="200" x2="70" y2="40" class="ax"/>
  <text x="590" y="234" class="l lab" text-anchor="middle">ground speed v (m/s)</text>
  <text x="40" y="122" class="l lab" text-anchor="middle" transform="rotate(-90 40 122)">required update rate f (Hz)</text>
  <!-- single reference line f = v / (1.0 m) -->
  <line x1="122" y1="194.7" x2="1110" y2="93.3" stroke="#1d4e89" stroke-width="2.4"/>
  <text x="196" y="64" class="l lab" fill="#1d4e89">f = v / ε, drawn at ε = 1.0 m</text>
  <!-- third variable: e slider at v = 20 m/s -->
  <line x1="590" y1="173.3" x2="590" y2="93.3" stroke="#9a6a1a" stroke-width="1.2" stroke-dasharray="3 3"/>
  <circle cx="590" cy="173.3" r="5" fill="#f2f5f9" stroke="#7d8798" stroke-width="1.6"/>
  <circle cx="590" cy="146.7" r="5" fill="#1d4e89" stroke="#ffffff" stroke-width="1.4"/>
  <circle cx="590" cy="93.3" r="5" fill="#f2f5f9" stroke="#9a6a1a" stroke-width="1.6"/>
  <text x="606" y="177" class="l lab" fill="#7d8798">ε = 2.0 m · 10 Hz · loose</text>
  <text x="606" y="150" class="l" fill="#1d4e89" font-size="12.5" font-weight="600">ε = 1.0 m · 20 Hz · reference</text>
  <text x="606" y="97" class="l lab" fill="#8a5f15">ε = 0.5 m · 40 Hz · tight</text>
</svg>

<p class="tiny" style="margin:6px 0 14px 0">The third variable ε sets how often you must update at a given speed: track tightly (ε = 0.5 m) and you double the reference rate; cruise loosely (ε = 2.0 m) and you can halve it.</p>

<div class="insight">Rate becomes comparable only when paired with speed. The distance per update, ε = v / f, is what the number means for the vehicle; report it, not Hz alone.</div>

---
<!-- ========== 5 · CONCEPTUAL MAP ========== -->

<div class="kicker">05 · A map of claims</div>

# What each component learns, and what it does not establish
<p class="lede">Different works study different components, and a model can be built for one job without establishing what a UAV actually needs.</p>

<div class="rule"></div>

<table>
  <tr><th style="width:22%">Object</th><th>It learns</th><th>It does not establish by itself</th></tr>
  <tr><td><b>Visual representation</b></td><td>features of images, video, geometry</td><td>controllable dynamics, memory, metric flight feasibility</td></tr>
  <tr><td><b>JEPA / latent predictor</b></td><td>future representations, conditioned on context &amp; actions</td><td>correct counterfactuals outside training coverage</td></tr>
  <tr><td><b>Generative world model</b></td><td>future images, video, scene observations</td><td>low latency, calibrated uncertainty, safe control</td></tr>
  <tr><td><b>VLM</b></td><td>relations between visual evidence and language</td><td>action consequences, metric accuracy, flight stability</td></tr>
  <tr><td><b>VLA</b></td><td>actions conditioned on vision–language–history</td><td>an explicit model for replanning or diagnosis</td></tr>
  <tr><td><b>World–action model</b></td><td>coupled future prediction and action generation</td><td>general language understanding, cross-mission transfer</td></tr>
  <tr><td><b>3DGS map</b></td><td>view-dependent appearance, spatial primitives</td><td>complete geometry, unknown-space safety, dynamics</td></tr>
  <tr><td><b>Belief model</b></td><td>alternative hidden states and uncertainty</td><td>correct belief updates or useful information-gathering</td></tr>
</table>

<div class="insight" style="margin-top:14px">A VLM can <em>name</em> a building without knowing how to approach it; a 3DGS render can treat unknown space as free; a VLA can imitate a route without being able to revise it after occlusion. None of this means the method is bad; each component serves a different purpose.</div>

---
<!-- ========== 6 · WORLD MODELS & JEPA ========== -->

<div class="kicker">06 · Predictive representations</div>

# World models &amp; JEPA-style prediction

<p class="sub">These foundations set the bar: a UAV predictor must clearly beat a small encoder trained from scratch, and must also improve on a pretrained representation used directly.</p>

<div class="rule"></div>

<div class="grid2">
  <div class="card">
    <span class="kpill cyan">Established baselines</span>
    <p style="font-size:15px"><b>DreamerV3</b> learns a compact environment model, <em>imagines</em> futures, and acts on those imagined outcomes.</p>
    <p style="font-size:15px"><b>DINO-WM</b> shows how a pretrained visual representation can support latent dynamics and goal-directed planning.</p>
    <p style="font-size:15px"><b>V-JEPA 2</b> makes the point that video-representation pretraining is <em>not</em> the same as action-conditioned control.</p>
  </div>
  <div class="card">
    <span class="kpill amber">The decisive test</span>
    <p style="font-size:15px">The research content is not the embedding loss; it is whether the representation supports the following:</p>
    <ul>
      <li><b>action ranking</b> and replanning,</li>
      <li>task <b>transfer</b>, and</li>
      <li><b>calibrated failure detection</b>.</li>
    </ul>
  </div>
</div>

<svg class="diagram" width="1160" height="204" viewBox="0 0 1200 236" xmlns="http://www.w3.org/2000/svg" style="margin-top:8px">
  <style>.bo{fill:#f6f7fb;stroke:#d7dde7} .th{fill:#1b2433;font-size:12.5px;font-weight:600;text-anchor:middle} .tl{fill:#7d8798;font-size:10.5px;text-anchor:middle} .ar{stroke:#1d4e89;stroke-width:1.8;fill:none} .du{stroke:#9a6a1a;stroke-width:1.8;fill:none;stroke-dasharray:6 4}</style>
  <rect class="bo" x="40" y="60" width="180" height="64" rx="10"/><text x="130" y="86" class="th">ENCODER</text><text x="130" y="106" class="tl">o_t · action history</text>
  <rect class="bo" x="400" y="60" width="180" height="64" rx="10" stroke="#1d4e89"/><text x="490" y="86" class="th" fill="#1d4e89">PREDICTOR</text><text x="490" y="106" class="tl">z_t · candidate a · Δt</text>
  <rect class="bo" x="760" y="60" width="180" height="64" rx="10" stroke="#2e7d4f"/><text x="850" y="86" class="th" fill="#2e7d4f">TARGET ENCODER</text><text x="850" y="106" class="tl">future o_t+k → latent</text>
  <line x1="220" y1="92" x2="400" y2="92" class="ar"/>
  <line x1="490" y1="124" x2="490" y2="150" class="du"/>
  <rect class="bo" x="390" y="150" width="200" height="54" rx="10"/><text x="490" y="176" class="th">ẑ_t+k</text><text x="490" y="194" class="tl">predicted</text>
  <line x1="850" y1="124" x2="850" y2="150" class="du"/>
  <rect class="bo" x="750" y="150" width="200" height="54" rx="10" fill="#ffffff"/><text x="850" y="176" class="th">z_t+k</text><text x="850" y="194" class="tl">matched target</text>
  <line x1="590" y1="177" x2="750" y2="177" class="du"/>
  <text x="670" y="170" text-anchor="middle" fill="#9a6a1a" font-size="11.5">compare (embedding loss)</text>
  <text x="600" y="217" text-anchor="middle" fill="#7d8798" font-size="12">What z must preserve → hidden geometry · target identity · visibility</text>
</svg>

---
<!-- ========== 6 · JEPA LANDSCAPE ========== -->

<div class="kicker">07 · JEPA landscape</div>

# What is already taken, and what stays open
<div class="rule"></div>

<div class="two-col">
  <div style="display:flex; flex-direction:column; gap:8px">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--bad); font-weight:600">GENERIC DIRECTIONS ALREADY COVERED</p>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>Factorized predictive state</b> <span style="color:var(--faint)">→</span> Orthogonal JEPA, JEPA-Anything</p></div>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>“Add physical meaning”</b> <span style="color:var(--faint)">→</span> PhyLatent, action-based representation learning</p></div>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>Object / masked prediction</b> <span style="color:var(--faint)">→</span> Causal-JEPA</p></div>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>Multi-step consistency</b> <span style="color:var(--faint)">→</span> Semigroup-JEPA, hierarchical latent WMs</p></div>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>Structured uncertainty</b> <span style="color:var(--faint)">→</span> UWM-JEPA, belief-latent work</p></div>
    <div class="card" style="padding:12px 16px"><p style="margin:0"><b>Counterfactual prediction</b> <span style="color:var(--faint)">→</span> Twin Rollouts, controlled world-model theory</p></div>
  </div>
  <div class="dpair">
    <div class="card" style="border-left:3px solid var(--good)">
      <p style="font-size:12px; letter-spacing:.03em; color:var(--good); font-weight:600">OPEN UAV QUESTIONS</p>
      <ul>
        <li><b>Action sufficiency:</b> what must survive compression</li>
        <li><b>Memory after compression:</b> the small obstacle still matters</li>
        <li><b>Partial observability:</b> belief, not one best map</li>
        <li><b>Language → dynamics binding:</b> goals meet physics</li>
      </ul>
    </div>
    <div class="insight warn">Average rollout error can hide the collapse of the rare future that actually decides the action.</div>
  </div>
</div>

---
<!-- ========== 7 · VLM/VLA FAMILIES ========== -->

<div class="kicker">08 · Vision–language in the loop</div>

# Three architectures for vision–language navigation
<p class="lede">The literature has converged on three patterns, which trade off control, safety, and debuggability differently; practice is moving toward hybrids of all three.</p>

<div class="rule"></div>

<div class="grid3">
  <div class="card" style="border-top:2px solid var(--cyan)">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan)">VLM AS PERCEPTION</p>
    <p style="font-size:14.5px">The model produces captions, object locations, semantic maps, or labels, which a separate planner and controller then use for navigation.</p>
    <p style="font-size:12px; color:var(--faint)">VLM → semantic map → planner → controller</p>
    <p class="tiny" style="color:var(--good); margin-top:8px">easy to deploy and inspect</p>
    <p class="tiny" style="color:var(--bad)">semantic errors amplify downstream, and the model never sees the consequences of its actions</p>
  </div>
  <div class="card" style="border-top:2px solid var(--amber)">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber)">VLM AS HIGH-LEVEL PLANNER</p>
    <p style="font-size:14.5px">The model proposes subgoals or route decisions, while MPC, sampling-based control, or a geometric controller handles the flight.</p>
    <p style="font-size:12px; color:var(--faint)">VLM → subgoals → MPC / geometry</p>
    <p class="tiny" style="color:var(--good); margin-top:8px">clear safety interface</p>
    <p class="tiny" style="color:var(--bad)">may select plans that are spatially or temporally infeasible unless feasibility is exposed to the model</p>
  </div>
  <div class="card" style="border-top:2px solid var(--cyan)">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan)">END-TO-END VLA</p>
    <p style="font-size:14.5px">A single model maps vision, language, and history directly to actions or action chunks; FlightGPT and AeroVLA exemplify this direction.</p>
    <p style="font-size:12px; color:var(--faint)">single network → actions</p>
    <p class="tiny" style="color:var(--good); margin-top:8px">tight perception–action coupling</p>
    <p class="tiny" style="color:var(--bad)">debugging, safety verification, and long-horizon replanning are harder, and so is transfer from simulation to real flight</p>
  </div>
</div>

<div class="insight good">The trend is hybrid: semantic slow branches, fast action branches, recurrent memory, waypoint interfaces, action chunks, and explicit uncertainty. This reflects onboard compute and delay constraints, not evidence that one architecture has won.</div>

---
<!-- ========== 8 · AERIAL SOTA ========== -->

<div class="kicker">09 · Aerial state of the art</div>

# Capable pieces, but no unified agent
<p class="lede">The field is fragmented: no current result demonstrates an agent that does all of the following at once.</p>

<div class="rule"></div>

<table style="margin-top:4px">
  <tr><th style="width:20%">Capability</th><th style="width:28%">Representative work</th><th>Where it stands</th></tr>
  <tr><td><b>Aerial VLN benchmark</b></td><td>OpenFly</td><td>shared evaluation infrastructure; not a strong baseline or a real-flight guarantee</td></tr>
  <tr><td><b>Language → action</b></td><td>FlightGPT · AeroVLA · Qwen-RobotNav</td><td>grounds language to waypoint/action output; <b>long horizon, calibration, unseen scenes</b> remain hard</td></tr>
  <tr><td><b>Future + action</b></td><td>WorldFly · FlowPilot</td><td>predicts future visual/depth with trajectories; <b>costly</b>; language and action sufficiency are separate</td></tr>
  <tr><td><b>Aerial world model</b></td><td>SkyJEPA</td><td>action-conditioned quadrotor dynamics, sim-to-real direction; <b>dynamics ≠ camera-based navigation</b></td></tr>
  <tr><td><b>Fast / slow inference</b></td><td>FSD-VLN · LiteVLA-H · AsyncVLA</td><td>decouples reasoning from control; <b>action age is rarely measured</b> end-to-end</td></tr>
  <tr><td><b>Memory for VLA</b></td><td>ReMem-VLA, recurrent-query</td><td>history improves temporal grounding, but may keep <b>appearance, not belief or state</b></td></tr>
  <tr><td><b>Open-vocab object nav</b></td><td>AirHunt · AeroBelief · AECNav · ConsistNav</td><td>semantic search, evidence consolidation, belief; <b>verification &amp; occlusion recovery</b> under-tested</td></tr>
</table>

<div class="insight">Within the work reviewed here, no single system yet combines language grounding, a calibrated spatial belief, action-conditioned prediction, occlusion-aware planning, flight dynamics, and transfer to new scenes.</div>

---
<!-- ========== 9 · LANGUAGE ROLES ========== -->

<div class="kicker">10 · Language</div>

# Language plays three different roles
<p class="lede">Treating language as one undifferentiated input produces agents that rewrite the physics when the instruction changes, or fail to update their belief when the evidence does.</p>

<div class="rule"></div>

<div class="grid3">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan); font-weight:600">01 · TASK SPECIFICATION</p>
    <p style="font-size:14.5px">Selecting a destination, a target, or an inspection objective.</p>
    <p class="tiny" style="margin-top:8px">changing the instruction changes the <b>task cost or subgoal</b>, not the physics</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber); font-weight:600">02 · EVIDENCE</p>
    <p style="font-size:14.5px">Reports such as “route blocked,” “target moved,” or “surface inspected.”</p>
    <p class="tiny" style="margin-top:8px">new evidence should <b>update both the belief and the future prediction</b></p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--good); font-weight:600">03 · KNOWLEDGE OF DYNAMICS</p>
    <p style="font-size:14.5px">Properties that change how objects and the environment behave.</p>
    <p class="tiny" style="margin-top:8px">carries constraints on <b>how things move</b>, conveyed in language</p>
  </div>
</div>

<div class="insight" style="margin-top:16px">Most VLM/VLA systems do not expose enough structure to test whether an instruction change really stops at the task layer.</div>

---
<!-- ========== 10 · 3D GAUSSIAN MEMORY ========== -->

<div class="kicker">11 · Scene memory</div>

# A Gaussian map is a memory, a simulator, or a state
<p class="lede">3D Gaussian Splatting serves three roles of increasing strength; the strongest is also the least understood.</p>

<div class="rule"></div>

<div class="grid3">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan); font-weight:600">PERSISTENT MEMORY</p>
    <p style="font-size:14.5px">It stores appearance, semantics, and spatial relations across viewpoints, which is useful for search and inspection from changing altitude and angle.</p>
    <p class="tiny" style="margin-top:8px">VISTA · ATLAS Navigator</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber); font-weight:600">SIMULATOR</p>
    <p style="font-size:14.5px">It provides fast novel-view synthesis and a visually rich environment for training.</p>
    <p class="tiny" style="margin-top:8px">SOUS VIDE / FiGS · GRaD-Nav</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--good); font-weight:600">PREDICTIVE STATE</p>
    <p style="font-size:14.5px">Gaussian features feed a learned controller or world model; this is the strongest role and the least verified.</p>
    <p class="tiny" style="margin-top:8px">must keep collision geometry, ego-motion, visibility, consequences</p>
  </div>
</div>

<div class="insight warn" style="margin-top:16px">A plausible render in unobserved space is not a safe occupancy estimate. The map therefore needs an explicit interface to geometric safety, state estimation, dynamic objects, and unknown space.</div>

---
<!-- ========== 11 · TIMING ========== -->

<div class="kicker">12 · Action age</div>

# Time is part of the state

<p class="sub">The aircraft keeps moving while the model reasons, so a plan computed from an old observation can be unsafe even when the model is exact at the time of its input.</p>

<div class="rule"></div>

<svg class="diagram" width="1160" height="244" viewBox="0 0 1200 250" xmlns="http://www.w3.org/2000/svg">
  <style>.b1{fill:#f6f7fb;stroke:#d7dde7} .b2{fill:#1b2433;font-size:12px;font-weight:600;text-anchor:middle} .b3{fill:#7d8798;font-size:10px;text-anchor:middle} .ar{stroke:#1d4e89;stroke-width:1.8}</style>
  <text x="600" y="20" text-anchor="middle" fill="#1b2433" font-size="13" font-weight="600">END-TO-END ACTION AGE, NOT “MODEL INFERENCE TIME”</text>
  <rect class="b1" x="24" y="50" width="150" height="62" rx="9"/><text x="99" y="76" class="b2">capture</text><text x="99" y="96" class="b3">observation</text>
  <rect class="b1" x="196" y="50" width="150" height="62" rx="9"/><text x="271" y="76" class="b2">sensor transfer</text><text x="271" y="96" class="b3">+ preprocessing</text>
  <rect class="b1" x="368" y="50" width="150" height="62" rx="9"/><text x="443" y="76" class="b2">encoder</text><text x="443" y="96" class="b3">+ memory update</text>
  <rect class="b1" x="540" y="50" width="150" height="62" rx="9" stroke="#9a6a1a"/><text x="615" y="76" class="b2" fill="#9a6a1a">VLM / world model</text><text x="615" y="96" class="b3">inference</text>
  <rect class="b1" x="712" y="50" width="150" height="62" rx="9"/><text x="787" y="76" class="b2">planner</text><text x="787" y="96" class="b3">+ safety filter</text>
  <rect class="b1" x="884" y="50" width="150" height="62" rx="9"/><text x="959" y="76" class="b2">command</text><text x="959" y="96" class="b3">transfer</text>
  <rect class="b1" x="1064" y="50" width="70" height="62" rx="9"/><text x="1099" y="76" class="b2" font-size="10">…</text>
  <rect class="b1" x="1029" y="150" width="140" height="52" rx="9" stroke="#2e7d4f" fill="#ffffff"/><text x="1099" y="174" class="b2" fill="#2e7d4f">actuator</text><text x="1099" y="192" class="b3">vehicle action</text>
  <line x1="174" y1="81" x2="196" y2="81" class="ar"/><line x1="346" y1="81" x2="368" y2="81" class="ar"/>
  <line x1="518" y1="81" x2="540" y2="81" class="ar"/><line x1="690" y1="81" x2="712" y2="81" class="ar"/>
  <line x1="862" y1="81" x2="884" y2="81" class="ar"/><line x1="1034" y1="81" x2="1064" y2="81" class="ar"/>
  <line x1="1099" y1="112" x2="1099" y2="150" class="ar"/>
  <line x1="30" y1="140" x2="1170" y2="140" stroke="#d7dde7" stroke-dasharray="3 6"/>
  <text x="30" y="162" fill="#7d8798" font-size="11">t=0 · observation</text>
  <text x="600" y="167" text-anchor="middle" fill="#7d8798" font-size="11">the vehicle kept moving during the whole pipeline</text>
</svg>

<div class="grid2" style="margin-top:14px">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--bad); font-weight:600">REPORT, NOT ASSUME</p>
    <p style="font-size:14px">Neural inference time is not the control-loop latency: a model reporting sub-20 ms inference can still act on a much older observation. Reports should include timestamped action age, motion during inference, and dropped or superseded actions.</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--good); font-weight:600">DEFENSIBLE SHAPE</p>
    <p style="font-size:14px">A <b>fast reactive actor</b> (stabilization, local obstacles, tracking, fallback) plus a <b>slower deliberative branch</b> (subgoals, costs, memory queries, reveal maneuvers), sharing a time-indexed state with explicit arbitration.</p>
  </div>
</div>

---
<!-- ========== 12 · OCCLUSION & BELIEF ========== -->

<div class="kicker">13 · Occlusion</div>

# Occlusion draws the line between memory and belief

<p class="sub">Remembering the last location of a target is not enough when the target may have moved. A capable agent must keep alternatives, predict how an action changes visibility, and update its belief on evidence.</p>

<div class="rule"></div>

<div class="two-col" style="grid-template-columns:1fr 520px">
  <div>
    <ul>
      <li><b>Memory</b> stores the last-seen state. <b>Belief</b> holds a weighted set of alternatives.</li>
      <li>A useful model predicts two things: how the <b>scene evolves</b> under a candidate action, <b>and</b> what <b>evidence</b> that action exposes and how belief changes.</li>
      <li>This is stronger than an exploration bonus or a confidence head.</li>
    </ul>
    <div class="card" style="margin-top:12px">
      <p style="font-size:12px; letter-spacing:.03em; color:var(--amber); font-weight:600">DESIGN RULE</p>
      <p style="font-size:14px; margin:0">Keep <b>imagined evidence</b> (produced inside the planning branch) separate from <b>observations admitted into live memory</b>.</p>
    </div>
  </div>
  <svg class="diagram" width="520" height="396" viewBox="0 0 600 470" xmlns="http://www.w3.org/2000/svg">
    <style>.pn{fill:#f6f7fb;stroke:#d7dde7} .pl{fill:#49546a;font-size:12.5px} .pk{fill:#1b2433;font-size:13.5px;font-weight:600}</style>
    <rect class="pn" x="20" y="26" width="560" height="200" rx="12"/>
    <text x="40" y="52" fill="#9a6a1a" class="pk" font-size="11" letter-spacing="2">MEMORY · LAST SEEN</text>
    <circle cx="120" cy="150" r="12" fill="#b03a3a" opacity=".85"/>
    <text x="148" y="154" class="pl" fill="#b03a3a">target</text>
    <circle cx="120" cy="150" r="2" fill="#1b2433" opacity=".6"/>
    <path d="M40 150 L86 150" stroke="#1d4e89" stroke-width="2"/>
    <text x="40" y="120" class="pl">camera</text>
    <path d="M204 40 L344 40 L344 226 L204 226 Z" fill="#1d4e89" opacity=".14" stroke="#1d4e89" stroke-dasharray="5 4"/>
    <text x="274" y="130" class="pl" fill="#1d4e89">occluder</text>
    <circle cx="470" cy="150" r="14" fill="none" stroke="#b03a3a" stroke-dasharray="4 3"/>
    <text x="470" y="196" text-anchor="middle" class="pl" fill="#b03a3a">stored here</text>
    <text x="470" y="212" text-anchor="middle" class="pl" fill="#b03a3a">but did it move?</text>
    <rect class="pn" x="20" y="252" width="560" height="200" rx="12"/>
    <text x="40" y="278" fill="#2e7d4f" class="pk" font-size="11" letter-spacing="2">BELIEF · COULD BE HERE</text>
    <path d="M204 266 L344 266 L344 452 L204 452 Z" fill="#1d4e89" opacity=".14" stroke="#1d4e89" stroke-dasharray="5 4"/>
    <circle cx="425" cy="328" r="15" fill="none" stroke="#2e7d4f" stroke-width="2" opacity=".9"/>
    <circle cx="470" cy="362" r="15" fill="none" stroke="#2e7d4f" stroke-width="2" opacity=".55"/>
    <circle cx="515" cy="328" r="15" fill="none" stroke="#2e7d4f" stroke-width="2" opacity=".3"/>
    <text x="470" y="418" text-anchor="middle" class="pl">hypotheses + likelihoods</text>
    <path d="M140 392 L260 392" stroke="#2e7d4f" stroke-width="1.8"/>
    <text x="140" y="420" class="pl" fill="#2e7d4f">reveal action → gather evidence → update</text>
  </svg>
</div>

---
<!-- ========== 13 · SAFETY ========== -->

<div class="kicker">14 · Safety &amp; uncertainty</div>

# What the learned part proposes, what the safety layer enforces

<p class="sub">Control Barrier Functions, backup trajectories, geofencing, certified mapping and perception-aware MPC give explicit constraints a language model must not be allowed to disable.</p>

<div class="rule"></div>

<svg class="diagram" width="1160" height="272" viewBox="0 0 1200 300" xmlns="http://www.w3.org/2000/svg">
  <style>.ly{fill:#f6f7fb;stroke:#d7dde7} .lt{fill:#1b2433;font-size:12.5px;font-weight:600} .ll{fill:#49546a;font-size:11.5px} .lx{stroke:#d7dde7;stroke-width:1.4}</style>
  <rect class="ly" x="40" y="14" width="1120" height="78" rx="12" stroke="#1d4e89"/>
  <text x="62" y="38" class="lt" fill="#1d4e89">LEARNED · PROPOSES</text>
  <text x="62" y="62" class="ll">task value · information gain · inspection value · subgoals</text>
  <text x="62" y="80" class="ll">a VLM may configure the task-facing cost, never the safety terms</text>
  <rect class="ly" x="40" y="104" width="1120" height="78" rx="12" stroke="#9a6a1a"/>
  <text x="62" y="128" class="lt" fill="#9a6a1a">SAFETY · ENFORCES</text>
  <text x="62" y="152" class="ll">control barrier functions · backup trajectory · geofence · vehicle limits</text>
  <text x="62" y="170" class="ll">sensor validity · recoverability · collision and clearance constraints</text>
  <rect class="ly" x="40" y="194" width="1120" height="62" rx="12" stroke="#2e7d4f"/>
  <text x="62" y="218" class="lt" fill="#2e7d4f">AIRCRAFT + ACTUATORS</text>
  <text x="62" y="240" class="ll">physics · limits · network · inner control loop</text>
  <line x1="600" y1="92" x2="600" y2="104" class="lx"/>
  <line x1="600" y1="182" x2="600" y2="194" class="lx"/>
  <text x="600" y="286" text-anchor="middle" fill="#7d8798" font-size="12">A language command cannot disable a barrier function.</text>
</svg>

<div class="insight warn" style="margin-top:12px">Two errors recur: a valid rendered depth sample is treated as proof of free space, and a nominal controller period is treated as sensor-to-actuator latency. The assumptions must be stated, or the proof does not cover the real stack.</div>

---
<!-- ========== 14 · OPEN AREAS ========== -->

<div class="kicker">15 · Open questions</div>

# Seven concrete research directions

<p class="lede">Each is a concrete mechanism that can be tested on route-level UAV tasks, rather than a vague ambition.</p>

<div class="rule"></div>

<div class="grid3">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan)">PREDICTION &amp; MEMORY</p>
    <p style="font-size:14px; margin-top:10px"><b>A.</b> <span style="color:var(--fg)">Action-preserving, task-conditioned world models</span>: a predictive state that preserves route feasibility, visibility, and safety-critical geometry when the task changes.</p>
    <p style="font-size:14px"><b>B.</b> <span style="color:var(--fg)">Belief dynamics for information-seeking flight</span>: predict the future scene, the evidence an action reveals, and the belief update that evidence should cause.</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber)">LANGUAGE &amp; ADAPTATION</p>
    <p style="font-size:14px; margin-top:10px"><b>C.</b> <span style="color:var(--fg)">Achievable language intentions</span>: “inspect the far side” should imply viewpoint, visibility, and motion requirements, not just an endpoint label.</p>
    <p style="font-size:14px"><b>D.</b> <span style="color:var(--fg)">Selective adaptation</span>: keep scene, vehicle-response, and sensor models separable, so that a change in payload, lighting, or wind updates one without corrupting the others.</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--good)">SYSTEM &amp; SAFETY</p>
    <p style="font-size:14px; margin-top:10px"><b>E.</b> <span style="color:var(--fg)">Continuous prediction under delay</span>: a time-indexed workspace that reconciles late results and rejects or repairs stale plans.</p>
    <p style="font-size:14px"><b>F.</b> <span style="color:var(--fg)">3DGS with control semantics</span>: occupancy confidence, unknown space, and dynamic objects, comparing view quality, collision prediction, and route success separately.</p>
    <p style="font-size:14px"><b>G.</b> <span style="color:var(--fg)">Safety-aware learned planning</span>: learned task value and information gain under formal constraints, with a recoverable fallback and explicitly stated assumptions and timing.</p>
  </div>
</div>

<div class="insight" style="margin-top:16px">The novelty bar is high: “use factorized JEPA,” “add latent uncertainty,” and “use Gaussian tokens” are already represented. A contribution must justify the UAV-specific state, update rule, or control interface.</div>

---
<!-- ========== 16 · ENVIRONMENTS ========== -->

<div class="kicker">16 · Environments</div>

# Where these systems are actually tested

<p class="lede">The papers reviewed here run on a small set of platforms and datasets. Each gives different evidence, and the gaps explain why real-flight guarantees are rare.</p>

<div class="rule"></div>

<table>
  <tr><th style="width:29%">Environment</th><th style="width:32%">What it provides</th><th>What it does not provide</th></tr>
  <tr><td><b>Photoreal RGB simulation</b><br><span class="tiny">AirSim · Isaac Sim · Gazebo (PX4 / ArduPilot)</span></td><td>controllable scenes, ground truth, repeatable episodes</td><td>full network, actuator, lighting, and calibration realism; scenes are often frozen while the model plans</td></tr>
  <tr><td><b>Gaussian renderers as simulators</b><br><span class="tiny">SOUS VIDE / FiGS · GRaD-Nav</span></td><td>fast novel views, visually rich training and evaluation</td><td>appearance is not collision geometry; metric scale and odometry transfer stay unproven</td></tr>
  <tr><td><b>Benchmarks and datasets</b><br><span class="tiny">OpenFly aerial VLN</span></td><td>shared, large-scale route evaluation</td><td>offline success measures, not closed-loop flight or measured latency</td></tr>
  <tr><td><b>Real aircraft</b><br><span class="tiny">PX4 / ArduPilot, onboard compute</span></td><td>true latency, power, and network behavior</td><td>expensive and rare in this literature; action age is seldom measured</td></tr>
</table>

<div class="insight">Comparable results need a matched sensor suite, action space, scene split, and compute budget. Simulation success, inference speed, and rendering quality are not closed-loop real-flight autonomy.</div>

---
<!-- ========== 15 · EVALUATION ========== -->

<div class="kicker">17 · Evaluation</div>

# Evaluate a UAV agent up a six-step ladder

<p class="lede">Start from one real task, such as point-to-point navigation, target tracking, or occlusion-aware inspection, and evaluate upward through the following levels.</p>

<div class="rule"></div>

<svg class="diagram" width="1160" height="310" viewBox="0 0 1200 316" xmlns="http://www.w3.org/2000/svg">
  <style>.sr{fill:#f6f7fb;stroke:#d7dde7} .st{fill:#1b2433;font-size:12.5px;font-weight:600} .sd{fill:#49546a;font-size:11px} .sn{fill:#ffffff;font-size:11px;font-weight:700;text-anchor:middle}</style>
  <g>
    <circle cx="70" cy="42" r="16" fill="#1d4e89"/><text x="70" y="47" class="sn">1</text>
    <rect class="sr" x="102" y="24" width="1000" height="36" rx="9" stroke="#1d4e89"/>
    <text x="126" y="47" class="st" fill="#1d4e89">Perception &amp; state</text>
    <text x="470" y="47" class="sd">localization · geometry · target identity · map uncertainty</text>
    <circle cx="70" cy="92" r="16" fill="#1d4e89"/><text x="70" y="97" class="sn">2</text>
    <rect class="sr" x="102" y="74" width="1000" height="36" rx="9" stroke="#1d4e89"/>
    <text x="126" y="97" class="st" fill="#1d4e89">Prediction</text>
    <text x="470" y="97" class="sd">one-step and multi-step action-conditioned latent prediction</text>
    <circle cx="70" cy="142" r="16" fill="#9a6a1a"/><text x="70" y="147" class="sn">3</text>
    <rect class="sr" x="102" y="124" width="1000" height="36" rx="9" stroke="#9a6a1a"/>
    <text x="126" y="147" class="st" fill="#9a6a1a">Decision</text>
    <text x="470" y="147" class="sd">action ranking · subgoal choice · reveal value · replanning</text>
    <circle cx="70" cy="192" r="16" fill="#9a6a1a"/><text x="70" y="197" class="sn">4</text>
    <rect class="sr" x="102" y="174" width="1000" height="36" rx="9" stroke="#9a6a1a"/>
    <text x="126" y="197" class="st" fill="#9a6a1a">Closed loop</text>
    <text x="470" y="197" class="sd">route success · final error · path length · collision / clearance · action age</text>
    <circle cx="70" cy="242" r="16" fill="#2e7d4f"/><text x="70" y="247" class="sn">5</text>
    <rect class="sr" x="102" y="224" width="1000" height="36" rx="9" stroke="#2e7d4f"/>
    <text x="126" y="247" class="st" fill="#2e7d4f">Generalization</text>
    <text x="470" y="247" class="sd">unseen scenes · weather / lighting · altitude · noise · moving objects · dynamics</text>
    <circle cx="70" cy="292" r="16" fill="#2e7d4f"/><text x="70" y="297" class="sn">6</text>
    <rect class="sr" x="102" y="274" width="1000" height="36" rx="9" stroke="#2e7d4f"/>
    <text x="126" y="297" class="st" fill="#2e7d4f">Realism</text>
    <text x="470" y="297" class="sd">motion during inference · timestamped commands · stale actions superseded · assumptions checked</text>
  </g>
</svg>

<div class="grid2" style="margin-top:14px">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan); font-weight:600">MINIMUM BASELINES</p>
    <p style="font-size:14px; margin:0">a geometric controller or MPC, a native aerial VLA / world-action model, a predictive baseline in the style of FlowPilot or SkyJEPA, a fast/slow or asynchronous system, and a safety-filtered version.</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber); font-weight:600">ABLATIONS</p>
    <p style="font-size:14px; margin:0">remove language configuration, persistent memory, belief updates, future prediction, delay handling, and safety constraints, one at a time.</p>
  </div>
</div>

---
<!-- ========== 16 · POSITION ========== -->

<div class="kicker">18 · Position</div>

# The question this literature sets up

<p class="sub" style="font-size:30px; line-height:1.35; max-width:30ch">
Can a UAV hold a <span class="warm">persistent, uncertain scene model</span> and use a <span class="accent">language-configured predictive branch</span> to choose long-horizon actions, while a <span style="color:var(--good)">fast reactive branch</span> keeps flight safe during delayed deliberation?
</p>

<div class="grid2" style="margin-top:28px">
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan); font-weight:600">WHY IT IS REAL</p>
    <p style="font-size:15px; margin:0">Flying changes what the vehicle can see, and the best action may be one that reveals information rather than one that shortens the distance to the goal. The question therefore ties together VLM task interpretation, JEPA prediction, 3DGS memory, active perception, asynchronous control, and safety filtering.</p>
  </div>
  <div class="card">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber); font-weight:600">MAKE IT DEFENSIBLE</p>
    <p style="font-size:15px; margin:0">A credible paper should claim <b>one</b> structural mechanism, such as action-preserving task-conditioned compression, belief-aware observation prediction, or continuous time-indexed inference, and demonstrate it on route-level tasks with measured delay and strong baselines.</p>
  </div>
</div>

---
<!-- ========== 17 · REFERENCES ========== -->

<div class="kicker">Appendix · Sources</div>

# Selected references
<div class="rule"></div>

<div class="grid2" style="row-gap:14px">
  <div class="card" style="padding:14px 18px">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan)">WORLD MODELS &amp; JEPA</p>
    <p class="tiny" style="line-height:1.9; margin:8px 0 0 0">LeCun, <i>A Path Towards Autonomous Machine Intelligence</i><br>DreamerV3 · DINO-WM · V-JEPA 2 · Dynalang<br>Orthogonal JEPA · JEPA-Anything · VL-JEPA · Gaussian-JEPA</p>
  </div>
  <div class="card" style="padding:14px 18px">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--cyan)">AERIAL &amp; VLA</p>
    <p class="tiny" style="line-height:1.9; margin:8px 0 0 0">OpenFly · SkyJEPA · FlowPilot · OpenVLA<br>FSD-VLN · AsyncVLA · LiteVLA-H · ReMem-VLA<br>AirHunt · AeroBelief · AECNav · ConsistNav</p>
  </div>
  <div class="card" style="padding:14px 18px">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber)">3DGS &amp; MAPPING</p>
    <p class="tiny" style="line-height:1.9; margin:8px 0 0 0">VISTA · ATLAS Navigator · SemSafe-3DGS<br>FastBridge · Certifiably-Correct Mapping<br>semantic and uncertainty-aware Gaussian maps</p>
  </div>
  <div class="card" style="padding:14px 18px">
    <p style="font-size:12px; letter-spacing:.03em; color:var(--amber)">SAFETY &amp; CONTROL</p>
    <p class="tiny" style="line-height:1.9; margin:8px 0 0 0">Control Barrier Functions · backup trajectories<br>PA-MPPI · gatekeeper · certified mapping<br>perception-aware MPC · belief-space planning</p>
  </div>
</div>

<p class="tiny" style="color:var(--faint); text-align:center; margin:6px 0 0 0">Full citations, with venues and years, appear in the companion literature review document.</p>

<div class="deck-footer"><span>UAV NAVIGATION · LITERATURE REVIEW</span><span>WORLD MODELS · VLMs · 3DGS MEMORY</span></div>

