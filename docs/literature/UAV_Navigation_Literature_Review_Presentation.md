---
marp: true
html: true
size: 16:9
paginate: true
theme: default
math: false
---

<style>
  :root{--bg:#fff;--paper:#fbfcfe;--panel:#f6f7fb;--line:#d7dde7;--soft:#e8ebf2;--fg:#1b2433;--muted:#49546a;--faint:#7d8798;--blue:#1d4e89;--green:#2e7d4f;--amber:#9a6a1a;--purple:#7563a7}
  section{color:var(--fg);font-family:'Helvetica Neue',Arial,'Segoe UI',system-ui,sans-serif;font-size:17px;line-height:1.45;background:linear-gradient(180deg,var(--paper) 0%,var(--bg) 48%),var(--bg);padding:56px 74px 64px}
  h1{font-family:Georgia,'Times New Roman',serif;font-weight:700;font-size:36px;line-height:1.1;letter-spacing:-.01em;margin:0 0 14px;color:var(--fg);text-wrap:balance}
  h2{font-family:Georgia,'Times New Roman',serif;font-weight:700;font-size:23px;line-height:1.14;margin:0 0 7px;color:var(--fg)}
  p{margin:0 0 9px;color:var(--muted)} a{color:var(--blue);text-decoration:none} b,strong{color:var(--fg);font-weight:650}
  .lede{font-size:17px;color:var(--muted);max-width:76ch;margin:0 0 10px;line-height:1.48}
  .sub{font-size:21px;color:var(--fg);font-weight:500;max-width:62ch;line-height:1.36;margin-bottom:12px}
  .tiny{font-size:12px;color:var(--faint);line-height:1.45}.small{font-size:14px;line-height:1.42}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 15px}
  .tag{display:inline-block;font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--blue);border:1px solid rgba(29,78,137,.36);border-radius:999px;padding:2px 8px;margin-bottom:6px}
  .tag.green{color:var(--green);border-color:rgba(46,125,79,.38)}.tag.amber{color:var(--amber);border-color:rgba(154,106,26,.38)}.tag.purple{color:var(--purple);border-color:rgba(117,99,167,.4)}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:13px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.two-col{display:grid;grid-template-columns:1.15fr .85fr;gap:22px;align-items:center}
  ul{margin:4px 0;padding-left:17px}li{margin:0 0 6px;color:var(--muted);font-size:14.5px;line-height:1.42}
  section table{border-collapse:separate;border-spacing:0;width:100%;margin:4px 0 7px;border:1px solid var(--line);table-layout:fixed;background:#fff}
  section table th{font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);text-align:left;padding:6px 9px;border:0;border-bottom:1.5px solid var(--line)}
  section table td{font-size:13.2px;line-height:1.38;color:var(--muted);padding:6px 9px;border:0;border-bottom:1px solid var(--soft);vertical-align:top;background:#fff}td b{color:var(--fg)}
  section table tr:last-child td{border-bottom:0}
  .impl-table th{font-size:10px;padding:8px 9px}.impl-table td{font-size:13.5px;line-height:1.36;padding:8px 9px}
  .env-table td{font-size:12.7px;line-height:1.32;padding:7px 9px}
  .related-note{font-size:13px;line-height:1.35;color:var(--muted);margin:0 0 12px}
  .gap-claim{font-family:Georgia,'Times New Roman',serif;font-size:28px;line-height:1.2;color:var(--fg);max-width:42ch;margin:20px 0 25px}
  .gap-row{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .gap-box{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:18px 20px;min-height:150px}
  .gap-box p{font-size:18px;line-height:1.38;color:var(--fg);margin:4px 0 0}
  .diagram{display:block;width:100%;height:auto}.diagram text{font-family:'Helvetica Neue',Arial,sans-serif}
  .rate-list{margin:8px 0 12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
  .rate-list div{display:flex;justify-content:space-between;align-items:baseline;gap:16px;padding:7px 2px;color:var(--muted);font-size:15px}
  .rate-list div+div{border-top:1px solid var(--soft)}
  section::after{content:attr(data-marpit-pagination);position:absolute;right:58px;bottom:22px;font-size:10px;letter-spacing:.08em;color:var(--faint)}
  section.lead::after{display:none}
</style>

<!-- _class: lead -->
# Literature Review

<p class="sub" style="margin-top:12px">Visual navigation for UAVs</p>

---

# Occam's razor: When is complex navigation needed?

<p class="lede">A question for the research agenda: could a planned route at a suitable cruise altitude handle much of ordinary delivery?</p>

<div class="grid2" style="margin-top:16px">
  <div class="card"><span class="tag green">Simple baseline</span><h2>Climb, cruise, approach</h2><p><a href="https://www.primeair.amazon/operations">Prime Air</a> publishes planned routes and cruise altitudes of 205–250 ft outbound and 325–370 ft returning; it descends near the delivery point.</p><p class="small">A street-level route must show a mission benefit over this option.</p></div>
  <div class="card"><span class="tag amber">Task requires proximity</span><h2>Reach an obscured surface</h2><p>Under-bridge inspection requires views that are unavailable from above. GNSS and obstacle clearance can also become difficult under the structure.</p><p class="small"><a href="https://www.skydio.com/customer-stories/japanese-infrastructure-waymark">Inspection operator account</a>: evidence for the task need, not a benchmark for a reviewed paper.</p></div>
</div>

<p class="small" style="margin-top:13px"><b>Open question:</b> Which missions defeat the simpler flight design, and does added navigation improve completion, interventions, time, or energy? The examples above motivate the question; they do not settle it for every mission.</p>

---

# Motivation: What visual navigation requires

<p class="lede">A navigation result depends on the goal, what the UAV knows about its position and surroundings, and whether it can execute a safe route.</p>

<div class="grid3" style="margin-top:18px">
  <div class="card" style="min-height:240px"><h2>Interpret the goal</h2><p>A waypoint, image, object, or route instruction leaves different decisions to the UAV.</p><p class="small"><a href="https://arxiv.org/abs/2308.06735">AerialVLN</a> studies route instructions; <a href="https://arxiv.org/abs/2601.12742">AirHunt</a> studies object search.</p></div>
  <div class="card" style="min-height:240px"><h2>Keep track of the scene</h2><p>Estimate position, remember landmarks, and distinguish observed space from guesses.</p><p class="small"><a href="https://arxiv.org/abs/2609.08164">AeroBelief</a> maintains target beliefs; <a href="https://arxiv.org/abs/2507.01125">VISTA</a> studies active mapping for exploration.</p></div>
  <div class="card" style="min-height:240px"><h2>Move safely in time</h2><p>Choose a feasible path and act before new obstacles or stale observations make it unsafe.</p><p class="small"><a href="https://arxiv.org/abs/2608.00635">FlowPilot</a> predicts trajectories; <a href="https://arxiv.org/abs/2605.00884">LiteVLA-H</a> separates fast and slow updates.</p></div>
</div>

---

# Motivation: Flight speed sets the update rate

<p class="lede">If no more than <b>s</b> metres should pass between usable updates, the minimum update rate is <b>f = v / s</b>.</p>

<div class="two-col" style="align-items:start;margin-top:6px">
  <svg class="diagram" viewBox="0 0 650 355" role="img" aria-label="Required update rate by ground speed for four travel distances between updates">
    <rect x="58" y="18" width="510" height="282" fill="#fff" stroke="#d7dde7"/>
    <g stroke="#e8ebf2" stroke-width="1">
      <line x1="58" y1="300" x2="568" y2="300"/><line x1="58" y1="253" x2="568" y2="253"/><line x1="58" y1="206" x2="568" y2="206"/>
      <line x1="58" y1="159" x2="568" y2="159"/><line x1="58" y1="112" x2="568" y2="112"/><line x1="58" y1="65" x2="568" y2="65"/>
      <line x1="228" y1="18" x2="228" y2="300"/><line x1="398" y1="18" x2="398" y2="300"/><line x1="568" y1="18" x2="568" y2="300"/>
    </g>
    <g fill="#49546a" font-size="12" text-anchor="end">
      <text x="49" y="304">0</text><text x="49" y="257">10</text><text x="49" y="210">20</text><text x="49" y="163">30</text><text x="49" y="116">40</text><text x="49" y="69">50</text><text x="49" y="22">60</text>
    </g>
    <g fill="#49546a" font-size="12" text-anchor="middle"><text x="58" y="321">0</text><text x="228" y="321">5</text><text x="398" y="321">10</text><text x="568" y="321">15</text></g>
    <text x="313" y="346" text-anchor="middle" fill="#1b2433" font-size="13">Ground speed (m/s)</text>
    <text x="15" y="160" text-anchor="middle" transform="rotate(-90 15 160)" fill="#1b2433" font-size="13">Update rate (Hz)</text>
    <line x1="58" y1="300" x2="568" y2="18" stroke="#1d4e89" stroke-width="3"/>
    <line x1="58" y1="300" x2="568" y2="159" stroke="#2e7d4f" stroke-width="3"/>
    <line x1="58" y1="300" x2="568" y2="253" stroke="#9a6a1a" stroke-width="3"/>
    <line x1="58" y1="300" x2="568" y2="276" stroke="#7563a7" stroke-width="3" stroke-dasharray="7 5"/>
    <circle cx="398" cy="112" r="5" fill="#1d4e89" stroke="#fff" stroke-width="2"/><circle cx="398" cy="206" r="5" fill="#2e7d4f" stroke="#fff" stroke-width="2"/><circle cx="398" cy="253" r="5" fill="#9a6a1a" stroke="#fff" stroke-width="2"/>
    <g font-size="12" font-weight="600"><text x="520" y="36" fill="#1d4e89">0.25 m</text><text x="520" y="151" fill="#2e7d4f">0.5 m</text><text x="520" y="246" fill="#9a6a1a">1.0 m</text><text x="520" y="291" fill="#7563a7">2.0 m</text></g>
  </svg>
  <div>
    <div class="card"><span class="tag">At 10 m/s</span>
      <div class="rate-list" aria-label="Required update rate at 10 metres per second"><div><span>1.0 m per update</span><b>10 Hz</b></div><div><span>0.5 m per update</span><b>20 Hz</b></div><div><span>0.25 m per update</span><b>40 Hz</b></div></div>
      <p class="small">These are perception or planning updates. At 5 m/s, a 0.4 s image-to-command delay means <b>2 m</b> travelled; at 10 m/s, <b>4 m</b>.</p>
    </div>
  </div>
</div>

---

# Environments: Where systems train and where they are tested

<table class="env-table" style="margin-top:8px">
  <tr><th style="width:18%">Study</th><th style="width:40%">Training environment / data</th><th>Test environment</th></tr>
  <tr><td><a href="https://arxiv.org/abs/2308.06735"><b>AerialVLN</b></a></td><td>Synthetic outdoor city scenes and instruction trajectories.</td><td>Held-out routes in its flight simulator; no physical flight.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2406.14240"><b>CityNav</b></a></td><td>Human trajectories in CityFlight, built from scans of Cambridge and Birmingham.</td><td>Held-out routes in CityFlight; the real-city scans are still a simulator.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2502.18041"><b>OpenFly</b></a></td><td>Rendered aerial trajectories from Unreal, GTA V, Google Earth, and 3DGS scenes.</td><td>Benchmark splits in rendered scenes; limited outdoor flight is reported separately.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2412.16346"><b>SOUS VIDE</b></a></td><td>FiGS: reconstructed Gaussian scene plus drone dynamics; expert trajectories distilled to a policy.</td><td>105 physical-flight trials, including mass, wind, lighting, and scene changes.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2606.03252"><b>AirDreamer</b></a></td><td>World-model and policy learning in simulation.</td><td>Unseen simulated environments and real drone flights without retuning.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2506.14009"><b>GRaD-Nav++</b></a></td><td>Differentiable drone dynamics in a photorealistic 3DGS simulator.</td><td>New simulated settings and real quadrotor flight.</td></tr>
</table>

---

# Goals: What the UAV is asked to reach

<table style="margin-top:8px">
  <tr><th style="width:19%">Input</th><th style="width:27%">What it gives the UAV</th><th style="width:31%">What still has to be solved</th><th>Examples</th></tr>
  <tr><td><b>Waypoint or pose</b></td><td>A target position in a stated coordinate frame.</td><td>Localization, frame conversion, route choice, obstacle clearance, and stopping.</td><td>Common in flight tasks; state the frame and map available.</td></tr>
  <tr><td><b>Destination image</b></td><td>A view of the target place or instance.</td><td>Recognize it from a new altitude or heading; search and remember visited places.</td><td>Less often isolated in the aerial papers reviewed here.</td></tr>
  <tr><td><b>Object or category</b></td><td>A phrase such as "find the water tower."</td><td>Search, distinguish candidates, estimate location, and confirm a match.</td><td><a href="https://arxiv.org/abs/2601.12742">AirHunt</a>, <a href="https://arxiv.org/abs/2609.08164">AeroBelief</a>, <a href="https://arxiv.org/abs/2608.10817">AECNav</a></td></tr>
  <tr><td><b>Route instruction</b></td><td>Landmarks, turns, and their order.</td><td>Ground each phrase in the view, retain subgoals, and recover after a missed landmark.</td><td><a href="https://arxiv.org/abs/2308.06735">AerialVLN</a>, <a href="https://arxiv.org/abs/2410.07087">OpenUAV</a>, <a href="https://arxiv.org/abs/2505.12835">FlightGPT</a></td></tr>
</table>

---

# Current implementations: VLMs and VLAs

<p class="related-note"><b>Related work:</b> <a href="https://arxiv.org/abs/2512.10942">VL-JEPA</a> learns visual-language answer features; <a href="https://arxiv.org/abs/2406.09246">OpenVLA</a> produces robot actions. Neither tests UAV flight.</p>
<table class="impl-table" style="margin-top:10px">
  <tr><th style="width:17%">Paper</th><th style="width:31%">Use on the UAV</th><th style="width:25%">Reported evidence</th><th>Limit in reviewed evidence</th></tr>
  <tr><td><a href="https://arxiv.org/abs/2505.12835"><b>FlightGPT</b></a></td><td>VLM selects actions from images and route instructions.</td><td>Improved unseen-route success on CityNav.</td><td>Response time unreported; no explicit staged planner.</td></tr>
  <tr><td><b>VLM-Nav</b></td><td>VLM labels obstacles; range sensing helps select five actions.</td><td>98% completion in two AirSim settings.</td><td>AirSim only; no action-consequence model.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2601.12742"><b>AirHunt</b></a></td><td>VLM object semantics guide continuous aerial search.</td><td>Improved object-search planning.</td><td>Target identity after occlusion is untested.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2507.06564"><b>SkyVLN</b></a></td><td>VLM plans a path; nonlinear MPC controls flight.</td><td>AirSim / Unreal urban simulation.</td><td>No physical-flight or response-time result in the paper.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2506.14009"><b>GRaD-Nav++</b></a></td><td>VLA trained in a differentiable Gaussian flight simulator.</td><td>Onboard hardware evaluation.</td><td>Calibrated uncertainty and delay handling unshown.</td></tr>
</table>

---

# Current implementations: Aerial world models

<p class="related-note"><b>Related work:</b> <a href="https://arxiv.org/abs/2301.04104">DreamerV3</a> learns from imagined rollouts; <a href="https://arxiv.org/abs/2411.04983">DINO-WM</a> plans with predicted visual features. Aerial deployment requires its own test.</p>
<table class="impl-table" style="margin-top:10px">
  <tr><th style="width:17%">Paper</th><th style="width:31%">Prediction and UAV use</th><th style="width:25%">Reported evidence</th><th>Limit in reviewed evidence</th></tr>
  <tr><td><a href="https://arxiv.org/abs/2606.23444"><b>SkyJEPA</b></a></td><td>Predicts quadrotor state under candidate actions for control.</td><td>Long-horizon sim-to-real control.</td><td>State input; no camera or language task.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2606.03252"><b>AirDreamer</b></a></td><td>Latent futures train a drone navigation policy.</td><td>Real flights up to 1.8 m/s without retuning.</td><td>Calibrated risk and safety semantics unshown.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2608.00635"><b>FlowPilot</b></a></td><td>Predicts future depth and an executable trajectory.</td><td>Real-time quadrotor deployment.</td><td>No language goal or end-to-end action-age result.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2606.06147"><b>WorldFly</b></a></td><td>Predicts future camera view with language-conditioned action.</td><td>87% seen; 31% unseen benchmark success.</td><td>Future-frame compute; real flight unshown.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2606.01205"><b>ImagineUAV</b></a></td><td>World-action prediction feeds a kinodynamic planner.</td><td>Aerial planning method in the source review.</td><td>Our notes describe the method, but do not audit real-flight evaluation.</td></tr>
</table>

---

# Current implementations: Fast and slow flight loops

<p class="lede">These systems separate slower semantic decisions from frequent flight updates, but place the boundary at different points.</p>
<svg class="diagram" style="height:170px" viewBox="0 0 812 195" role="img" aria-label="Camera and goal feed slow semantic and fast action branches; the slow branch updates guidance while the fast branch sends commands to the UAV, which supplies a new observation">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#49546a"/></marker></defs>
  <rect x="8" y="52" width="156" height="76" rx="8" fill="#f6f7fb" stroke="#d7dde7"/>
  <text x="86" y="83" text-anchor="middle" font-size="16" font-weight="700" fill="#1b2433">Camera + goal</text>
  <text x="86" y="105" text-anchor="middle" font-size="13" fill="#49546a">new observations</text>
  <rect x="234" y="8" width="224" height="68" rx="8" fill="#f3f5fb" stroke="#1d4e89" stroke-width="1.5"/>
  <text x="346" y="35" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4e89">Slow branch</text>
  <text x="346" y="56" text-anchor="middle" font-size="13" fill="#49546a">semantic / route guidance</text>
  <rect x="234" y="106" width="224" height="68" rx="8" fill="#f2f8f4" stroke="#2e7d4f" stroke-width="1.5"/>
  <text x="346" y="133" text-anchor="middle" font-size="16" font-weight="700" fill="#2e7d4f">Fast branch</text>
  <text x="346" y="154" text-anchor="middle" font-size="13" fill="#49546a">action / trajectory update</text>
  <rect x="554" y="106" width="178" height="68" rx="8" fill="#f6f7fb" stroke="#d7dde7"/>
  <text x="643" y="133" text-anchor="middle" font-size="16" font-weight="700" fill="#1b2433">UAV</text>
  <text x="643" y="154" text-anchor="middle" font-size="13" fill="#49546a">execute and observe</text>
  <path d="M164 75 L223 43" fill="none" stroke="#49546a" stroke-width="1.6" marker-end="url(#arrow)"/>
  <path d="M164 111 L223 139" fill="none" stroke="#49546a" stroke-width="1.6" marker-end="url(#arrow)"/>
  <path d="M346 76 L346 97" fill="none" stroke="#1d4e89" stroke-width="1.7" stroke-dasharray="5 4" marker-end="url(#arrow)"/>
  <path d="M458 140 L544 140" fill="none" stroke="#49546a" stroke-width="1.6" marker-end="url(#arrow)"/>
  <path d="M643 174 L643 185 L86 185 L86 138" fill="none" stroke="#9da7b7" stroke-width="1.3" marker-end="url(#arrow)"/>
</svg>
<table class="impl-table" style="margin-top:4px">
  <tr><th style="width:17%">Paper</th><th style="width:25%">Slow side</th><th style="width:28%">Fast side</th><th>Evidence / limit</th></tr>
  <tr><td><a href="https://arxiv.org/abs/2607.08359"><b>FSD-VLN</b></a></td><td>VLM semantic priors.</td><td>Diffusion action stream.</td><td>Long-route simulation gains; applied action age unreported.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2605.00884"><b>LiteVLA-H</b></a></td><td>Periodic semantic perception.</td><td>Action-token guidance: 50.65 ms onboard model step.</td><td>Full sensor-to-actuator delay unreported.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2609.18451"><b>VLM-MPPI</b></a></td><td>VLM selects a trajectory.</td><td>MPPI replans at 20 Hz.</td><td>Real quadrotor tests; 20 Hz is not VLM response rate.</td></tr>
</table>

---

# Current implementations: Architectural designs for memory

<table class="impl-table" style="margin-top:10px">
  <tr><th style="width:17%">Paper</th><th style="width:31%">Design</th><th style="width:25%">Reported evidence</th><th>Limit in reviewed evidence</th></tr>
  <tr><td><a href="https://arxiv.org/abs/2507.01125"><b>VISTA</b></a></td><td>Online semantic Gaussian map for exploration.</td><td>Real-quadrotor exploration.</td><td>Map is not a predictive or certified safety state.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2502.20386"><b>ATLAS Navigator</b></a></td><td>Language-embedded map selects task-relevant views.</td><td>Real ground-robot exploration.</td><td>Transfer to UAV flight and its safety constraints unshown.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2509.23563"><b>RAVEN</b></a></td><td>Persistent semantic-spatial memory for search.</td><td>Outdoor aerial object search.</td><td>Our review does not establish multi-stage route recovery.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2609.08164"><b>AeroBelief</b></a></td><td>Separate semantic and spatial target beliefs.</td><td>Aerial object-goal navigation.</td><td>Calibration and occlusion recovery unshown.</td></tr>
  <tr><td><a href="https://arxiv.org/abs/2608.10817"><b>AECNav</b></a></td><td>Chooses the next view to collect target evidence.</td><td>Open-vocabulary object navigation.</td><td>Identity persistence through missed detections unshown.</td></tr>
</table>

---

# Research gaps: Real-world validation

<p class="gap-claim">City-scale mapless route following has not been shown on a real UAV in this review.</p>
<div class="gap-row">
  <div class="gap-box"><span class="tag">Current evidence</span><p><a href="https://arxiv.org/html/2510.01348v1">SPRIN-D</a> reports a 1,371 m real sortie and a 1,023 m urban sortie using onboard sensors and prior geodata. Its 9 km course was the challenge target. <a href="https://arxiv.org/abs/2406.14240">CityNav</a> uses real-city scans, but routes run in simulation.</p></div>
  <div class="gap-box"><span class="tag amber">Needed</span><p>Fly held-out city routes with the same goal input and stated map access. Report arrival, collisions, minimum clearance, human intervention, time, and energy. Compare with planned overflight when the mission permits it.</p></div>
</div>

---

# What use do the papers establish?

<table class="impl-table" style="margin-top:12px">
  <tr><th style="width:25%">Research</th><th style="width:36%">Plausible job</th><th>Evidence boundary</th></tr>
  <tr><td><a href="https://arxiv.org/html/2608.00635v1"><b>FlowPilot</b></a> / <a href="https://arxiv.org/html/2606.03252v1"><b>AirDreamer</b></a></td><td>Local flight around clutter during an approach or inspection.</td><td>Real flights are short; they do not justify city-wide weaving for ordinary delivery.</td></tr>
  <tr><td><a href="https://arxiv.org/html/2510.01348v1"><b>SPRIN-D</b></a></td><td>Keep position estimates useful on long GNSS-denied waypoint missions.</td><td>Kilometer sorties use LiDAR and prior geographic data; not arbitrary city autonomy.</td></tr>
  <tr><td><a href="https://arxiv.org/html/2406.14240v3"><b>CityNav</b></a> / aerial VLN</td><td>Study grounding of language-described destinations in city scenes.</td><td>Language may be unnecessary when coordinates are available; benchmark success is simulated.</td></tr>
  <tr><td><b>World models</b></td><td>Potentially improve a decision under uncertainty.</td><td>Prediction quality matters only if it improves flight outcomes against a simpler planner.</td></tr>
</table>

<p class="small" style="margin-top:12px"><b>Application evidence and scientific evidence differ.</b> A paper can isolate a useful capability without proving that a complete commercial mission needs its full stack.</p>

---

# Research gaps: Long-horizon capability

<p class="gap-claim">Long routes are studied, but missed landmarks and stage-by-stage recovery remain weakly tested.</p>
<div class="gap-row">
  <div class="gap-box"><span class="tag">Current evidence</span><p><a href="https://arxiv.org/abs/2607.08359">FSD-VLN</a> reports gains on long aerial instruction routes. <a href="https://arxiv.org/abs/2509.23563">RAVEN</a> keeps spatial memory for outdoor object search. These test different goals.</p></div>
  <div class="gap-box"><span class="tag amber">Needed</span><p>Test multi-stage instructions when a landmark is missed or obscured. Report subgoal order, wrong turns, recovery distance, and final arrival on held-out routes.</p></div>
</div>

---

# Research gaps: Onboard computation

<p class="gap-claim">A model rate alone cannot tell us how far the UAV flies before a command applies.</p>
<div class="gap-row">
  <div class="gap-box"><span class="tag">Current evidence</span><p><a href="https://arxiv.org/abs/2605.00884">LiteVLA-H</a> reports a 50.65 ms fast model step, but our notes do not pair it with flight speed or full image-to-action delay. <a href="https://arxiv.org/abs/2609.18451">VLM-MPPI</a>'s 20 Hz is its planner rate, not VLM latency.</p></div>
  <div class="gap-box"><span class="tag amber">Needed</span><p>Illustration: at 5 m/s, a 0.4 s image-to-applied-action delay means 2 m of travel. Report both quantities in the same flight, plus late commands, clearance, and onboard power.</p></div>
</div>

---

# Research gaps: Dynamic obstacles

<p class="gap-claim">Moving obstacles can invalidate a path while the system is perceiving and planning.</p>
<div class="gap-row">
  <div class="gap-box"><span class="tag">Current evidence</span><p><a href="https://arxiv.org/abs/2606.01205">ImagineUAV</a> couples learned future prediction to a feasible-path planner. <a href="https://arxiv.org/abs/2607.01200">FastBridge</a> adds a safety filter whose guarantees depend on its map and model assumptions.</p></div>
  <div class="gap-box"><span class="tag amber">Needed</span><p>Cross moving obstacles at varied speeds and observation delays. Measure collision rate, minimum clearance, and whether the UAV replans when a prediction is wrong.</p></div>
</div>

---

# Research gaps: Sim-to-real robustness

<p class="gap-claim">Real-flight transfer is demonstrated, but the studies do not give a matched robustness comparison.</p>
<div class="gap-row">
  <div class="gap-box"><span class="tag">Current evidence</span><p><a href="https://arxiv.org/abs/2412.16346">SOUS VIDE / FiGS</a> trains a flight policy in a Gaussian-scene simulator and tests real flight under changes in mass, wind, and lighting. <a href="https://arxiv.org/abs/2606.03252">AirDreamer</a> flies a world-model policy at up to 1.8 m/s without retuning.</p></div>
  <div class="gap-box"><span class="tag amber">Needed</span><p>These are different flight tasks and tests. Hold the UAV, sensors, goal, and controller fixed; compare simulation with real flight under held-out wind, lighting, and sensor degradation. Report success and clearance.</p></div>
</div>

---

# References: Aerial tasks and policies

<div class="grid2" style="margin-top:12px">
  <div class="card"><span class="tag">Benchmarks and route tasks</span><p class="small"><a href="https://arxiv.org/abs/2308.06735">AerialVLN</a> - aerial vision-language navigation<br><a href="https://arxiv.org/abs/2406.14240">CityNav</a> - real-city scenes for aerial navigation<br><a href="https://arxiv.org/abs/2502.18041">OpenFly</a> - aerial VLN platform and dataset<br><a href="https://arxiv.org/abs/2410.07087">OpenUAV</a> - open-ended UAV object finding</p></div>
  <div class="card"><span class="tag green">Language and flight actions</span><p class="small"><a href="https://arxiv.org/abs/2505.12835">FlightGPT</a> - UAV vision-language navigation<br><a href="https://arxiv.org/abs/2507.06564">SkyVLN</a> - VLM path planning with nonlinear MPC<br><a href="https://arxiv.org/abs/2603.14363">AeroVLA / AerialVLA</a> - aerial vision-language-action<br><a href="https://arxiv.org/abs/2607.08359">FSD-VLN</a> - fast-slow aerial VLN<br><a href="https://arxiv.org/abs/2605.00884">LiteVLA-H</a> - dual-rate onboard guidance<br><a href="https://arxiv.org/abs/2609.18451">VLM-MPPI</a> - VLM selection over fast trajectories</p></div>
  <div class="card"><span class="tag amber">Aerial search and tracking</span><p class="small"><a href="https://arxiv.org/abs/2601.12742">AirHunt</a> - object search and continuous planning<br><a href="https://arxiv.org/abs/2609.08164">AeroBelief</a> - semantic-spatial target belief<br><a href="https://arxiv.org/abs/2608.10817">AECNav</a> - active evidence consolidation<br><a href="https://arxiv.org/abs/2509.23563">RAVEN</a> - persistent spatial memory<br><a href="https://arxiv.org/abs/2103.06522">Fast-Tracker 2.0</a> - active aerial tracking<br><a href="https://arxiv.org/abs/2604.21453">OA-VAT</a> - instance tracking and recovery</p></div>
  <div class="card"><span class="tag purple">VLM navigation</span><p class="small">VLM-Nav, "Mapless UAV navigation using monocular vision driven by vision-language models," PLOS ONE, 2026. See the supplementary review and local paper notes for its publication record.</p></div>
</div>

---

# References: Prediction and scene models

<div class="grid2" style="margin-top:12px">
  <div class="card"><span class="tag">Aerial world models</span><p class="small"><a href="https://arxiv.org/abs/2606.23444">SkyJEPA</a> - quadrotor state/action prediction<br><a href="https://arxiv.org/abs/2608.00635">FlowPilot</a> - future depth and trajectory<br><a href="https://arxiv.org/abs/2606.06147">WorldFly</a> - world-model VLA for UAV navigation<br><a href="https://arxiv.org/abs/2606.03252">AirDreamer</a> - drone navigation with world models<br><a href="https://arxiv.org/abs/2606.01205">ImagineUAV</a> - world-action model and kinodynamic planning</p></div>
  <div class="card"><span class="tag green">Maps and exploration</span><p class="small"><a href="https://github.com/HKUST-Aerial-Robotics/FUEL">FUEL</a> - frontier-based UAV exploration<br><a href="https://arxiv.org/abs/2507.01125">VISTA</a> - semantic Gaussian exploration on a quadrotor<br><a href="https://arxiv.org/abs/2502.20386">ATLAS Navigator</a> - language maps on a ground robot<br><a href="https://arxiv.org/abs/2606.06721">SCOUT</a> - semantic uncertainty on a ground robot<br><a href="https://arxiv.org/abs/2412.16346">SOUS VIDE / FiGS</a> - Gaussian-scene sim-to-real<br><a href="https://arxiv.org/abs/2506.14009">GRaD-Nav++</a> - differentiable simulator and aerial VLA</p></div>
  <div class="card"><span class="tag amber">Geometry and safety</span><p class="small"><a href="https://arxiv.org/abs/2609.19330">SemSafe-3DGS</a> - semantic risk in Gaussian maps<br><a href="https://arxiv.org/abs/2607.01200">FastBridge</a> - safety filters over Gaussian scenes<br><a href="https://arxiv.org/abs/2409.10283">ASMA</a> - scene-aware barrier functions<br><a href="https://arxiv.org/abs/2504.18713">Certifiably-Correct Mapping</a> - mapping with stated assumptions</p></div>
  <div class="card"><span class="tag purple">Companion source notes</span><p class="small"><a href="UAV%20General%20Literature%20Review%20-%202026-09-20.md">Supplementary literature review</a><br><a href="UAV%20Navigation%20Presentation%20Notes.md">Presentation notes</a><br><a href="UAV%20Navigation%20Papers%20in%20Depth.md">UAV papers in depth</a><br><a href="UAV%20VLM%20Literature%20Review%20-%20Full.md">Historical VLM source notes</a></p></div>
</div>

