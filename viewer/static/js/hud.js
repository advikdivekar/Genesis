// HTML overlay: clock, event feed, live metric charts, inspector, minimap, controls.
import { SEASONS, SEASON_COLORS, WEATHER, DEATH_COLORS, TERRAIN_HEX, ACTION_LABEL, toast } from './util.js';

const $ = (id) => document.getElementById(id);
const SPEEDS = [[0.25, '¼×'], [1, '1×'], [4, '4×'], [16, '16×'], [100, '100×'], [0, 'MAX']];

class Chart {
  constructor(parent, cfg) {
    this.cfg = cfg;
    const el = document.createElement('div');
    el.className = 'chart';
    el.innerHTML = `<div class="head"><span class="name">${cfg.name}</span><span class="big"></span></div><canvas></canvas>
      <div class="legend"></div>${cfg.hint ? `<div class="hint">${cfg.hint}</div>` : ''}`;
    parent.appendChild(el);
    this.big = el.querySelector('.big');
    this.legend = el.querySelector('.legend');
    this.canvas = el.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
  }
  draw(rows) {
    const { cfg, canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!rows.length) return;
    const n = rows.length;
    const series = cfg.series.map((s) => rows.map((r, i) => s.get(r, rows, i)));
    let lo = cfg.min, hi = cfg.max;
    if (lo === undefined || hi === undefined) {
      let a = Infinity, b = -Infinity;
      for (const v of series.flat()) if (v != null && isFinite(v)) { a = Math.min(a, v); b = Math.max(b, v); }
      if (cfg.ref !== undefined) { a = Math.min(a, cfg.ref); b = Math.max(b, cfg.ref); }
      if (!isFinite(a)) { a = 0; b = 1; }
      if (b - a < 1e-6) { b = a + 1; }
      lo = lo ?? a; hi = hi ?? b + (b - a) * 0.08;
    }
    const X = (i) => (n === 1 ? w : (i / (n - 1)) * w), Y = (v) => h - 2 - ((v - lo) / (hi - lo)) * (h - 6);
    // grid + reference
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h - 0.5); ctx.lineTo(w, h - 0.5); ctx.stroke();
    if (cfg.ref !== undefined) {
      ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.beginPath(); ctx.moveTo(0, Y(cfg.ref)); ctx.lineTo(w, Y(cfg.ref)); ctx.stroke(); ctx.setLineDash([]);
    }
    series.forEach((vals, si) => {
      const s = cfg.series[si];
      let svals = vals;
      if (s.norm) { const m = Math.max(1e-9, ...vals.filter((v) => v != null)); svals = vals.map((v) => (v == null ? null : lo + (v / m) * (hi - lo) * 0.92)); }
      ctx.beginPath();
      let started = false;
      svals.forEach((v, i) => { if (v == null || !isFinite(v)) return; if (!started) { ctx.moveTo(X(i), Y(v)); started = true; } else ctx.lineTo(X(i), Y(v)); });
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.stroke();
      if (cfg.fill && si === 0 && started) {
        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
        const g = ctx.createLinearGradient(0, 0, 0, h); g.addColorStop(0, s.color + '55'); g.addColorStop(1, s.color + '00');
        ctx.fillStyle = g; ctx.fill();
      }
    });
    const last = rows[n - 1];
    this.big.textContent = cfg.big ? cfg.big(last, rows) : '';
    this.legend.innerHTML = cfg.series.length > 1 || cfg.legendAlways ? cfg.series.map((s, i) => {
      const v = series[i][n - 1];
      return `<span><i style="background:${s.color}"></i>${s.label} <b>${v == null ? '–' : (s.fmt ? s.fmt(v) : v)}</b></span>`;
    }).join('') : '';
  }
}

export class HUD {
  constructor(net, app) {
    this.net = net; this.app = app;
    this.history = [];
    this.feedItems = [];
    this.pendingDeaths = [];
    this.deathFlushT = 0;
    this.fps = 0;
    this._buildControls();
  }

  init(init) {
    this.initMsg = init;
    this.history = (init.history || []).slice();
    this.spd = init.tpd / init.sample_every;   // samples per sim day
    $('seed-label').textContent = `seed ${init.seed} · ${init.n_initial} agents`;
    $('policy-pill').textContent = init.policy_desc || `${init.policy} baseline · non-learning`;
    $('rs-seed').value = init.seed; $('rs-agents').value = init.n_initial;
    $('rs-policy').innerHTML = init.policies.map((p) => `<option ${p === init.policy ? 'selected' : ''}>${p}</option>`).join('');
    const ECO = ['cooperative_large_prey', 'huddle_warmth'];
    const on = Object.entries(init.capabilities).filter(([, v]) => v).map(([k]) => k);
    const caps = on.filter((k) => !ECO.includes(k)), eco = on.filter((k) => ECO.includes(k));
    $('metrics-sub').textContent = `agent capabilities: ${caps.length ? caps.join(', ') : 'all off (default)'}` +
      (eco.length ? ` · ecology: ${eco.join(', ').replaceAll('_', ' ')}` : '');
    this._buildCharts();
    this._buildMinimap(init);
    $('feed').innerHTML = '';
    this.feedItems = [];
    for (const e of init.feed || []) this.addFeed(e, false);
    this.closeInspector();
  }

  // ------------------------------------------------------------------ charts
  _rate(key) {
    const d = this.spd;
    return (r, rows, i) => {
      if (i < d) return null;
      return (r[key] - rows[i - d][key]);
    };
  }
  _buildCharts() {
    const root = $('charts');
    root.innerHTML = '';
    const pct = (v) => (v * 100).toFixed(0) + '%';
    const int = (v) => Math.round(v).toLocaleString();
    this.charts = [
      new Chart(root, { name: 'Population', fill: true, min: 0,
        series: [{ get: (r) => r.pop, color: '#ffcf70', label: 'alive' }],
        big: (r) => `${r.pop} · ${r.dpy != null ? r.dpy.toFixed(2) : '–'} deaths/agent-yr` }),
      new Chart(root, { name: 'Body state (population mean)', min: 0, max: 1,
        series: [{ get: (r) => r.e, color: '#e0a050', label: 'energy', fmt: pct }, { get: (r) => r.hy, color: '#5fb6ff', label: 'hydration', fmt: pct },
          { get: (r) => r.hp, color: '#ff6b81', label: 'health', fmt: pct }, { get: (r) => r.w, color: '#ffe38a', label: 'warmth', fmt: pct }] }),
      new Chart(root, { name: 'Fauna', min: 0, max: 1,
        series: [{ get: (r) => r.sp, color: '#d8c3a0', label: 'small prey', norm: true, fmt: int },
          { get: (r) => r.lp, color: '#b07a4a', label: 'large prey', norm: true, fmt: int },
          { get: (r) => r.pr, color: '#ff5a5a', label: 'predators', norm: true, fmt: int }],
        hint: 'Each line is scaled to its own maximum.' }),
      new Chart(root, { name: 'Vegetation', min: 0, max: 1,
        series: [{ get: (r) => r.plant, color: '#7fdc6a', label: 'plant biomass', norm: true, fmt: int },
          { get: (r) => r.fruit, color: '#ff8c5a', label: 'fruit', norm: true, fmt: int }] }),
      new Chart(root, { name: 'Activity per sim day', min: 0, max: 1,
        series: [{ get: this._rate('eat'), color: '#9be58b', label: 'meals', norm: true, fmt: int },
          { get: this._rate('drink'), color: '#5fb6ff', label: 'drinks', norm: true, fmt: int },
          { get: this._rate('ks'), color: '#ff9a52', label: 'small kills', norm: true, fmt: int },
          { get: this._rate('kl'), color: '#ff4f6d', label: 'large kills', norm: true, fmt: int },
          { get: this._rate('scr'), color: '#ffd36b', label: 'ate others\' kill', norm: true, fmt: int }],
        hint: "“Ate others' kill” counts meals from a carcass that another agent killed or dropped. It measures co-feeding, not intentional sharing. Large prey only die when two or more agents attack within 6 ticks; that is a rule of the world." }),
      new Chart(root, { name: 'Spatial clustering (Clark–Evans R)', ref: 1,
        series: [{ get: (r) => r.ce, color: '#c9a0ff', label: 'R' }],
        big: (r) => (r.ce != null ? r.ce.toFixed(2) : '–'),
        hint: 'R < 1 means agents are aggregated, ≈ 1 random, > 1 dispersed. This uses nearest-neighbour distance over passable land. Baseline agents cluster around water and food, so this is not evidence of sociality.' }),
      new Chart(root, { name: 'Temperature (sea level)', ref: 0,
        series: [{ get: (r) => r.temp, color: '#7fd4ff', label: '°C' }], big: (r) => `${r.temp.toFixed(1)} °C` }),
    ];
    const dc = document.createElement('div');
    dc.className = 'chart';
    dc.innerHTML = `<div class="head"><span class="name">Deaths by cause (this run)</span><span class="big" id="deaths-total"></span></div>
      <div id="death-bar" style="display:flex;height:9px;border-radius:5px;overflow:hidden;background:rgba(255,255,255,0.06)"></div>
      <div class="legend" id="death-legend"></div>`;
    root.appendChild(dc);
    this._drawCharts();
  }
  _drawCharts() {
    if (!this.charts) return;
    const rows = this.history.slice(-360);
    for (const c of this.charts) c.draw(rows);
    const last = rows[rows.length - 1];
    if (!last) return;
    const names = this.initMsg.death_names;
    const tot = last.dc.reduce((a, b) => a + b, 0);
    $('deaths-total').textContent = tot;
    $('death-bar').innerHTML = tot ? last.dc.map((v, i) => v ? `<div title="${names[i]} ${v}" style="width:${v / tot * 100}%;background:${DEATH_COLORS[names[i]]}"></div>` : '').join('') : '';
    $('death-legend').innerHTML = last.dc.map((v, i) => v ? `<span><i style="background:${DEATH_COLORS[names[i]]}"></i>${names[i].replace('_', ' ')} <b>${v}</b></span>` : '').join('');
  }

  // ------------------------------------------------------------------ per frame
  onFrame(fr) {
    this.frame = fr;
    if (fr.samples?.length) {
      this.history.push(...fr.samples);
      if (this.history.length > 2000) this.history.splice(0, this.history.length - 2000);
      const now = performance.now();
      if (now - (this._chartT || 0) > 250) { this._chartT = now; this._drawCharts(); }
    }
    for (const e of fr.fx) {
      if (e.k === 'death') this.pendingDeaths.push(e);
      else if (['kill_large', 'fire_start', 'flood_start', 'drought_start', 'famine_start', 'cold_snap_start'].includes(e.k)) this.addFeed(e, true);
    }
    this._flushDeaths();
    // clock
    const init = this.initMsg;
    const rt = fr.tick - (init.spinup_ticks || 0);
    const yr = Math.floor(rt / init.year_len) + 1;
    const dayInSeason = Math.floor(((fr.tick + init.tpd / 4) % init.season_len) / init.tpd) + 1;   // tick 0 is 06:00
    const mins = Math.floor(fr.tod * 24 * 60);
    const hh = String(Math.floor(mins / 60)).padStart(2, '0'), mm = String(mins % 60).padStart(2, '0');
    $('clock-icon').textContent = fr.night > 0.5 ? '☾' : '☀';
    $('clock-text').innerHTML = `Year ${yr} · <span style="color:${SEASON_COLORS[fr.season]}">${SEASONS[fr.season]}</span> · Day ${dayInSeason} · ${hh}:${mm}`;
    const snowing = fr.weather === 1 && fr.temp < 0;
    $('weather-text').textContent = snowing ? '❄ Snow' : ['☀ Clear', '🌧 Rain', '🌫 Fog'][fr.weather];
    $('temp-text').textContent = `${fr.temp.toFixed(1)} °C`;
    const al = [];
    if (fr.drought) al.push(`<span class="pill warn">drought ${fr.drought}d</span>`);
    if (fr.cold_snap) al.push(`<span class="pill warn">cold snap ${fr.cold_snap}d</span>`);
    if (fr.famine) al.push(`<span class="pill warn">famine ${fr.famine}d</span>`);
    if (fr.fires) al.push(`<span class="pill warn">wildfire</span>`);
    $('alerts').innerHTML = al.join('');
    $('btn-play').textContent = fr.paused ? '▶' : '⏸';
    document.querySelectorAll('#speeds button').forEach((b) => b.classList.toggle('on', Number(b.dataset.v) === fr.speed));
  }

  engineStats(s) {
    const fr = this.frame;
    if (!fr) return;
    $('engine').innerHTML = `sim <b>${fr.tps.toFixed(0)}</b> ticks/s${fr.target_tps ? ` (target ${fr.target_tps})` : ' (max)'} · tick ${fr.tick}<br>
      render <b>${s.fps.toFixed(0)}</b> fps · ${s.agents} agents · ${s.fauna} animals · net ${(s.kbps).toFixed(0)} kB/s<br>
      ${fr.rec ? `<span style="color:#ff6b6b">●</span> recording every agent, every tick: ${fr.rec.rows.toLocaleString()} rows · ${fr.rec.mb} MB on disk${fr.rec.obs ? ' · +observations' : ''}<br>` : 'recording off<br>'}
      log: ${this.initMsg.run_dir}`;
  }

  // deaths are batched so the feed stays readable when many agents die per second
  _flushDeaths() {
    const now = performance.now();
    if (!this.pendingDeaths.length || now - this.deathFlushT < 1500) return;
    this.deathFlushT = now;
    const ds = this.pendingDeaths;
    this.pendingDeaths = [];
    if (ds.length === 1) { this.addFeed(ds[0], true); return; }
    const by = {};
    for (const d of ds) by[d.cause] = (by[d.cause] || 0) + 1;
    const parts = Object.entries(by).sort((a, b) => b[1] - a[1]).map(([c, n]) => `${n} ${c.replace('_', ' ')}`);
    const last = ds[ds.length - 1];
    this.addFeed({ k: 'deaths', t: last.t, x: last.x, y: last.y, n: ds.length, txt: parts.join(', '),
      cause: Object.entries(by).sort((a, b) => b[1] - a[1])[0][0] }, true);
  }

  addFeed(e, live) {
    const init = this.initMsg;
    let icon = '•', txt = '', color = '#cfd8e5';
    switch (e.k) {
      case 'death': icon = '☠'; color = DEATH_COLORS[e.cause] || color; txt = `#${e.uid} died of ${e.cause.replace('_', ' ')} (age ${e.age}d)`; break;
      case 'deaths': icon = '☠'; color = DEATH_COLORS[e.cause] || color; txt = `${e.n} deaths: ${e.txt}`; break;
      case 'kill_large': icon = '🦬'; color = '#ffb347'; txt = `#${e.uid} and others (${e.n} attackers) brought down a large prey`; break;
      case 'fire_start': icon = '🔥'; color = '#ff7a3d'; txt = e.v ? 'Wildfire started (manual)' : 'Wildfire ignited'; break;
      case 'flood_start': icon = '🌊'; color = '#5fb6ff'; txt = `River flood (${Math.round(e.v)} cells)`; break;
      case 'drought_start': icon = '☀'; color = '#ffd36b'; txt = `Drought for ${Math.round(e.v)} days`; break;
      case 'famine_start': icon = '🥀'; color = '#e0a050'; txt = `Famine for ${Math.round(e.v)} days`; break;
      case 'cold_snap_start': icon = '❄'; color = '#b8e3ff'; txt = `Cold snap for ${Math.round(e.v)} days`; break;
      default: return;
    }
    const rt = e.t - (init.spinup_ticks || 0);
    const day = Math.floor(rt / init.tpd) + 1;
    const li = document.createElement('li');
    li.innerHTML = `<span class="t">d${day}</span><span style="color:${color}">${icon}</span><span>${txt}</span>`;
    li.onclick = () => this.app.flyTo(e.x, e.y, 30);
    const ul = $('feed');
    ul.prepend(li);
    while (ul.children.length > 9) ul.lastChild.remove();
  }

  // ------------------------------------------------------------------ inspector
  inspector(d) {
    const el = $('inspector');
    el.classList.remove('hidden');
    if (!d.alive) {
      $('insp-action').innerHTML = '<span style="color:#ff8a8a">This agent has died.</span>';
      return;
    }
    const s = this.app.creatures?.get(d.uid);
    if (s) $('insp-swatch').style.background = `#${s.color.getHexString()}`;
    $('insp-title').textContent = `Agent #${d.uid}`;
    $('insp-sub').textContent = `generation ${d.gen}${d.parent ? ` · parent #${d.parent}` : ''} · age ${d.age_days.toFixed(1)} / ${d.lifespan_days} days`;
    const [ic, lab] = ACTION_LABEL[d.action] || ['•', d.action];
    $('insp-action').innerHTML = `${ic} <b>${lab}</b> <span class="muted">(${d.action})</span>${d.sheltered ? ' · <span class="muted">sheltered</span>' : ''}`;
    const bar = (name, v, c) => `<div class="bar-row"><span>${name}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%;background:${c}"></div></div><span class="v">${(v * 100).toFixed(0)}</span></div>`;
    $('insp-bars').innerHTML = bar('energy', d.energy, '#e0a050') + bar('hydration', d.hydration, '#5fb6ff') +
      bar('health', d.health, '#ff6b81') + bar('warmth', d.warmth, '#ffe38a') + bar('life used', d.age_days / d.lifespan_days, '#9aa5b5');
    const st = d.stats;
    const kv = [['position', `${d.x}, ${d.y}`], ['felt temp', `${d.temp_eff} °C`], ['carrying', d.carry === 'none' ? '—' : `${d.carry} ${d.carry_amt}`],
      ['energy eaten', st.energy_eaten.toFixed(2)], ['water drunk', st.water_drunk.toFixed(2)], ['distance', `${st.distance.toFixed(0)} cells`],
      ['small kills', st.kill_small], ['large kills', st.kill_large], ['food from others', st.food_from_others.toFixed(2)], ['damage taken', st.damage_taken.toFixed(2)]];
    $('insp-grid').innerHTML = kv.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join('');
    $('insp-genes').innerHTML = d.genes.map((g, i) => `<div title="g${i} = ${g}" style="height:${((g + 1) / 2 * 100).toFixed(0)}%"></div>`).join('');
    $('insp-hist').innerHTML = d.history.slice(-16).map((a) => `<span>${(ACTION_LABEL[a] || ['•'])[0]} ${a.replace('move_', '')}</span>`).join('');
    $('insp-follow').classList.toggle('on', this.app.followUid === d.uid);
    if (d.mind) {
      const top = Object.entries(d.mind.probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
      $('insp-mind').innerHTML = `<h4>Learner's decision</h4><div class="insp-grid"><span class="k">value estimate</span><span class="v">${d.mind.value.toFixed(3)}</span></div>` +
        top.map(([a, p]) => `<div class="bar-row"><span>${a.replace('move_', '→ ')}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%;background:#c9a0ff"></div></div><span class="v">${(p * 100).toFixed(0)}%</span></div>`).join('');
      $('insp-note').textContent = 'Bars are body variables. The value estimate is how much future reward the network expects; the probabilities are how likely it was to choose each action.';
    } else {
      $('insp-mind').innerHTML = '';
      $('insp-note').textContent = "Bars are the agent's own body variables. This agent runs a hand-written baseline policy, so it has no learned internal states.";
    }
  }
  closeInspector() { $('inspector').classList.add('hidden'); }

  // ------------------------------------------------------------------ minimap
  _buildMinimap(init) {
    const { W, H } = init;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const g = c.getContext('2d');
    const img = g.createImageData(W, H);
    const t = init.data.terrain, h = init.data.height;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, hex = TERRAIN_HEX[t[i]];
      const shade = 0.75 + Math.max(-0.35, Math.min(0.35, ((h[i] - (h[i - W - 1] ?? h[i])) * 0.35)));
      img.data[i * 4] = ((hex >> 16) & 255) * shade; img.data[i * 4 + 1] = ((hex >> 8) & 255) * shade; img.data[i * 4 + 2] = (hex & 255) * shade;
      img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    this.miniBase = c;
    const mm = $('minimap');
    const pick = (ev) => {
      const r = mm.getBoundingClientRect();
      this.app.flyTo((ev.clientX - r.left) / r.width * W, (ev.clientY - r.top) / r.height * H);
    };
    mm.onmousedown = (ev) => { pick(ev); mm.onmousemove = (e2) => e2.buttons && pick(e2); };
    mm.onmouseup = () => { mm.onmousemove = null; };
  }
  drawMinimap(creatures, world, camCell) {
    if (!this.miniBase) return;
    const mm = $('minimap'), g = mm.getContext('2d');
    const { W, H } = this.initMsg;
    g.imageSmoothingEnabled = false;
    g.drawImage(this.miniBase, 0, 0, mm.width, mm.height);
    const sx = mm.width / W, sy = mm.height / H;
    if (world.fireCells?.length) { g.fillStyle = '#ff6a2a'; for (const i of world.fireCells) g.fillRect((i % W) * sx, ((i / W) | 0) * sy, 1.5, 1.5); }
    g.fillStyle = 'rgba(255,90,90,0.95)';
    for (const s of creatures.fauna.values()) if (s.sp === 2) g.fillRect(s.x * sx - 1, s.y * sy - 1, 2.5, 2.5);
    g.fillStyle = 'rgba(176,122,74,0.9)';
    for (const s of creatures.fauna.values()) if (s.sp === 1) g.fillRect(s.x * sx - 1, s.y * sy - 1, 2, 2);
    g.fillStyle = '#ffffff';
    for (const s of creatures.agents.values()) g.fillRect(s.x * sx - 0.75, s.y * sy - 0.75, 1.6, 1.6);
    if (camCell) {
      g.strokeStyle = 'rgba(255,210,122,0.9)'; g.lineWidth = 1.2;
      g.beginPath(); g.arc(camCell[0] * sx, camCell[1] * sy, Math.max(6, camCell[2] * sx), 0, Math.PI * 2); g.stroke();
    }
  }

  // ------------------------------------------------------------------ controls
  _buildControls() {
    const net = this.net, app = this.app;
    $('speeds').innerHTML = SPEEDS.map(([v, l], i) => `<button data-v="${v}" title="Speed ${l} (key ${i + 1})">${l}</button>`).join('');
    document.querySelectorAll('#speeds button').forEach((b) => b.onclick = () => net.send({ cmd: 'speed', value: Number(b.dataset.v) }));
    $('btn-play').onclick = () => net.send({ cmd: this.frame?.paused ? 'play' : 'pause' });
    $('btn-step').onclick = () => net.send({ cmd: 'step' });
    document.querySelectorAll('[data-cmd]').forEach((b) => b.onclick = () => {
      const c = b.dataset.cmd;
      net.send({ cmd: c });
      toast(`${b.textContent.trim()} triggered (logged)`);
    });
    document.querySelectorAll('[data-tool]').forEach((b) => b.onclick = () => app.setTool(app.tool === b.dataset.tool ? null : b.dataset.tool));
    $('btn-restart').onclick = () => $('restart-dlg').classList.toggle('hidden');
    $('rs-cancel').onclick = () => $('restart-dlg').classList.add('hidden');
    $('rs-go').onclick = () => {
      net.send({ cmd: 'restart', seed: Number($('rs-seed').value), policy: $('rs-policy').value, n_agents: Number($('rs-agents').value) });
      $('restart-dlg').classList.add('hidden');
      document.getElementById('loading').classList.remove('done');
      document.getElementById('loading-msg').textContent = 'generating world…';
    };
    $('btn-help').onclick = () => $('help').classList.toggle('hidden');
    $('help-close').onclick = () => $('help').classList.add('hidden');
    $('insp-close').onclick = () => app.select(null);
    $('insp-follow').onclick = () => app.toggleFollow();
    $('overlay').onchange = (e) => app.setOverlay(Number(e.target.value));
    $('btn-director').onclick = () => app.toggleDirector();
    $('btn-shot').onclick = () => app.screenshot();
    $('btn-rec').onclick = () => app.toggleRecord();
    $('btn-quality').onclick = () => app.cycleQuality();
  }
  setToolButtons(tool) {
    document.querySelectorAll('[data-tool]').forEach((b) => b.classList.toggle('on', b.dataset.tool === tool));
    const hint = $('tool-hint');
    hint.classList.toggle('hidden', !tool);
    if (tool) hint.textContent = tool === 'fire' ? 'Click the map to start a wildfire (Esc cancels)' : 'Click near a river to flood its floodplain (Esc cancels)';
  }
}
