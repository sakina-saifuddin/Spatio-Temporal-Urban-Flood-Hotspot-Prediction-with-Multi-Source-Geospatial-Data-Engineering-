// FloodSense 2.0, guided by Scout.
// Sections:
//   1. Settings: score levels, colors, feature names
//   2. Map setup: street map, labels on top
//   3. Load data and paint the smooth color glow
//   4. Clicking the map + detail card + plain-words "why" (SHAP → sentences)
//   5. Storm switcher (311 report dots)
//   6. Top 10 hotspots
//   7. Place search
//   8. Scout's messages, welcome and About screens
//   9. Small helpers

// =====================================================================
// 1. Settings
// =====================================================================
// A 0–1 score becomes a plain word. 0.5 is the model's own cutoff:
// at or above it, the model says "this looks like a spot that flooded".
const LEVELS = [
  { min: 0.75, label: "Very high", color: "#d7263d", text: "#fff" },
  { min: 0.5,  label: "High",      color: "#f08a3c", text: "#fff" },
  { min: 0.25, label: "Medium",    color: "#f2cf45", text: "#1a1a1a" },
  { min: 0,    label: "Low",       color: "#3fa66b", text: "#fff" },
];
const levelFor = (score) => LEVELS.find((l) => score >= l.min);

// Smooth color ramp for the glow: green → yellow → orange → red (like the slide-6 mockup)
const RAMP = [[0, [63, 166, 107]], [0.3, [242, 207, 69]], [0.55, [240, 138, 60]], [0.8, [215, 38, 61]], [1, [170, 16, 40]]];

const FEATURES = {
  elev:  "Elevation",
  bayou: "Distance to bayou",
  slope: "Terrain slope",
  land:  "Land use",
  hwy:   "Near highways",
};
const HOUSTON = [29.8, -95.4];

// Filled in once the data loads
const state = {
  cells: [],
  byId: new Map(),      // grid id → cell
  medians: {},          // typical value of each feature, for "low"/"close"/"flat" wording
  storms: [],
  stormLayer: null,
  repeatLayer: null,    // purple repeat-flooding glow, when that view is on
  repeat: null,
  activeStorm: null,
  pin: null,            // the marker showing what you clicked or searched
  hoverDot: null,       // the 311 dot under the mouse, if any
  flight: Promise.resolve(), // resolves when the current fly-to animation ends
  glow: null,
};

// The research data groups locations into 0.01° squares: id = round(lat*100)_round(lon*100)
const cellIdAt = (lat, lon) => `${Math.round(lat * 100)}_${Math.round(lon * 100)}`;

// =====================================================================
// 2. Map setup: real street map underneath, street names on top
// =====================================================================
// Easy zooming: mouse wheel / trackpad zooms quickly and smoothly, double-click zooms in.
const map = L.map("map", {
  preferCanvas: true, zoomControl: false, minZoom: 9,
  scrollWheelZoom: true, wheelPxPerZoomLevel: 70, zoomSnap: 0.25, zoomDelta: 1, doubleClickZoom: true,
}).setView(HOUSTON, 10);
L.control.zoom({ position: "bottomright" }).addTo(map);

// Layer order (bottom → top): street map  →  see-through color glow  →  311 dots
map.createPane("glow").style.zIndex = 350;

// Detailed street map (Esri World Street Map). If its tiles fail to load, switch to OpenStreetMap.
const streets = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Map &copy; Esri, HERE, Garmin, USGS, NGA, OpenStreetMap contributors" }
).addTo(map);
let tileErrors = 0;
streets.on("tileerror", () => {
  if (++tileErrors !== 6) return;   // a few failures in a row → use the backup map
  map.removeLayer(streets);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
});

// Zoomed in close, fade the glow a bit more so street names stay easy to read
// The closer you zoom, the lighter the glow, so the street map stays crisp
function applyGlowOpacity() {
  const z = map.getZoom();
  const base = z >= 15 ? 0.35 : z >= 13 ? 0.5 : 1;
  state.glow?.setOpacity(state.repeatLayer ? base * 0.6 : base);   // soften the score colors under the pins
}
map.on("zoomend", applyGlowOpacity);

// Legend: a smooth gradient bar matching the glow
document.getElementById("legend-items").innerHTML = `
  <div class="gradient" style="background:linear-gradient(to right, ${RAMP.map(([s, c]) => `rgb(${c}) ${s * 100}%`).join(", ")})"></div>
  <div class="gradient-labels"><span>Low</span><span>Medium</span><span>High</span><span>Very high</span></div>`;

// =====================================================================
// 3. Load data and paint the smooth color glow
// =====================================================================
// DATA_V: bump when the files in data/ change, so browsers never mix old data with new code
const DATA_V = 2;
const dataUrl = (name) => `data/${name}.json?v=${DATA_V}`;
Promise.all([getJSON(dataUrl("cells")), getJSON(dataUrl("storms")), getJSON(dataUrl("hotspots")), getJSON(dataUrl("repeat"))])
  .then(([cells, storms, hotspots, repeat]) => {
    state.cells = cells;
    state.storms = storms;
    state.repeat = repeat;
    for (const c of cells) state.byId.set(c.id, c);
    state.medians = {
      elev: median(cells.map((c) => c.elev)),
      bayou: median(cells.map((c) => c.bayou)),
      slope: median(cells.map((c) => c.slope)),
      hwy: median(cells.map((c) => c.hwy)),
    };

    state.glow = paintGlow(cells).addTo(map);
    map.fitBounds(state.glow.getBounds(), { padding: [10, 10] });

    buildStormChips();
    buildHotspots(hotspots);
    buildAboutStats();
    scout(DEFAULT_MSG);
  })
  .catch((err) => {
    console.error(err);
    scout("Uh-oh, I couldn't load the map data. If you opened index.html directly, start a local server instead (see web/README.md).");
  });

// Turn the ~1 km grid into one soft, see-through image (no hard boxes):
// 1) draw one pixel per grid cell, 2) scale it up with smoothing + blur so neighbors blend,
// 3) lay it over the street map. Places that weren't sampled stay transparent.
function paintGlow(cells, pixel = scorePixel) {
  const rows = cells.map((c) => Math.round(c.lat * 100));
  const cols = cells.map((c) => Math.round(c.lon * 100));
  const [r0, r1, c0, c1] = [Math.min(...rows), Math.max(...rows), Math.min(...cols), Math.max(...cols)];
  const W = c1 - c0 + 1, H = r1 - r0 + 1;

  const small = document.createElement("canvas");
  small.width = W; small.height = H;
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(W, H);
  cells.forEach((c, k) => {
    const x = cols[k] - c0, y = r1 - rows[k];            // north is up
    const rgba = pixel(c);
    if (rgba) img.data.set(rgba, (y * W + x) * 4);
  });
  sctx.putImageData(img, 0, 0);

  const S = 14;                                         // upscale factor
  const big = document.createElement("canvas");
  big.width = W * S; big.height = H * S;
  const bctx = big.getContext("2d");
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "high";
  bctx.filter = `blur(${S * 0.45}px)`;
  bctx.drawImage(small, 0, 0, W * S, H * S);

  const bounds = [[(r0 - 0.5) / 100, (c0 - 0.5) / 100], [(r1 + 0.5) / 100, (c1 + 0.5) / 100]];
  return L.imageOverlay(big.toDataURL(), bounds, { pane: "glow", interactive: false });
}

// Default glow pixel: score color, see-through; higher scores a bit stronger
function scorePixel(c) {
  const [r, g, b] = rampColor(c.score);
  return [r, g, b, Math.round((0.25 + 0.35 * c.score) * 255)];
}

function rampColor(t) {
  for (let i = 1; i < RAMP.length; i++) {
    const [s1, a] = RAMP[i - 1], [s2, b] = RAMP[i];
    if (t <= s2) {
      const f = (t - s1) / (s2 - s1);
      return a.map((v, j) => Math.round(v + (b[j] - v) * f));
    }
  }
  return RAMP[RAMP.length - 1][1];
}

// =====================================================================
// 4. Clicking the map + detail card + plain-words "why"
// =====================================================================
// Hover: a small label follows the mouse over scored areas
const hoverTip = L.tooltip({ direction: "top", offset: [0, -8], className: "hover-tip" });
map.on("mousemove", (e) => {
  if (state.hoverDot) return;  // a 311 dot shows its own street card instead
  const c = state.byId.get(cellIdAt(e.latlng.lat, e.latlng.lng));
  map.getContainer().style.cursor = c ? "pointer" : "";
  if (!c) { map.closeTooltip(hoverTip); return; }
  let tip = `<b>${levelFor(c.score).label}</b> · click for details`;
  if (state.repeatLayer) {
    const n = Object.keys(c.reports).length;
    tip += `<br>Flood reports in <b>${n} of 4</b> storms`;
  } else if (state.activeStorm) {
    const n = c.reports[state.activeStorm] || 0;
    tip += `<br>${n ? `${n} flood report${n === 1 ? "" : "s"}` : "No flood reports"} nearby in ${state.activeStorm}`;
  }
  hoverTip.setLatLng(e.latlng).setContent(tip);
  if (!map.hasLayer(hoverTip)) hoverTip.addTo(map);
});
map.on("mouseout", () => map.closeTooltip(hoverTip));

// Click anywhere: drop a pin and explain that ~1 km area
map.on("click", (e) => pickPoint(e.latlng.lat, e.latlng.lng));

function dropPin(lat, lon) {
  if (state.pin) state.pin.setLatLng([lat, lon]);
  else state.pin = L.marker([lat, lon], { keyboard: false }).addTo(map);
}

// Show a spot: pin + detail card + sidebar panel. The street name is looked up and filled in
// a moment later. `storm` is set when the spot is a 311 report dot.
let pickToken = 0;
function pickPoint(lat, lon, label = null, { storm = null } = {}) {
  const token = ++pickToken;
  clearFlood();
  dropPin(lat, lon);
  const c = state.byId.get(cellIdAt(lat, lon));
  const where = label || "Finding the street name…";
  if (c) showCard(c, where);
  else showNoData(lat, lon, where);
  markHotspot(c?.id);
  showSpot({ lat, lon, c, storm, street: null, area: null, label });
  if (storm) addFloodNote(storm, "Painting the flooded street…");

  streetName(lat, lon).then((place) => {
    if (token !== pickToken) return;   // user already picked somewhere else
    const { street, area } = place;
    const whereEl = document.querySelector("#card .where");
    if (whereEl && !label) whereEl.innerHTML = `${icon("pin")}${esc(street)}${area ? `, ${esc(area)}` : ""}`;
    showSpot({ lat, lon, c, storm, street, area, label });
    showPhotos(lat, lon, street, storm, token);
    if (storm) drawFloodedStreet(lat, lon, place, storm, token);
  });
}

// Fly to a spot and remember when the flight ends (the water waits for it before filling in)
function flyToSpot(lat, lon, zoom) {
  state.flight = new Promise((resolve) => {
    map.once("moveend", resolve);
    setTimeout(resolve, 2500);   // safety net if no move happens
  });
  map.flyTo([lat, lon], zoom, { duration: 1.1 });
}

// =====================================================================
// 4b. Blue flood on the street: when you click a 311 report dot, the street lookup also returns
//     that street's shape. Once the map has zoomed in, water spreads out along the street from
//     the reported spot (only the stretch near this storm's reports), then keeps gently flowing.
// =====================================================================
const FLOOD_REACH_M = 90;      // paint street parts within this distance of a 311 report
const FILL_MS = 2200;          // how long the water takes to spread along the street
const floodRenderer = L.svg(); // SVG (not canvas) so the water can animate with CSS
let floodLayer = null;
let waterLines = [];           // [{ line, role }] so widths can follow the zoom

function clearFlood() {
  if (floodLayer) map.removeLayer(floodLayer);
  floodLayer = null;
  waterLines = [];
}

// Water as wide as a real street (~12 m), whatever the zoom level
function waterWidth() {
  const mPerPx = (40075016.686 * Math.cos(map.getCenter().lat * Math.PI / 180)) / 2 ** (map.getZoom() + 8);
  return Math.max(5, Math.min(44, 12 / mPerPx));
}
const ROLE_WIDTH = { glow: 1.9, body: 1, surface: 0.55, shimmer: 0.18 };
map.on("zoomstart", () => waterLines.forEach(({ line }) => finishFill(line)));
map.on("zoomend", () => {
  const w = waterWidth();
  waterLines.forEach(({ line, role }) => line.setStyle({ weight: Math.max(1.5, w * ROLE_WIDTH[role]) }));
});

async function drawFloodedStreet(lat, lon, { street, lines }, year, token) {
  await state.flight;                       // wait until the map has finished zooming in
  if (token !== pickToken) return;
  clearFlood();
  const st = state.storms.find((s) => s.year === year);
  // This storm's reports within ~600 m decide which parts of the street turn blue
  const reports = st.dots.filter(([a, o]) => meters(lat, lon, a, o) < 600);
  if (!reports.some(([a, o]) => a === lat && o === lon)) reports.push([lat, lon]);

  // Split every wet stretch at the point closest to the clicked report, so water spreads both ways from there
  const halves = (lines || []).flatMap((line) => floodedRuns(line, reports)).flatMap((run) => {
    let k = 0, best = Infinity;
    run.forEach(([a, o], i) => { const d = meters(lat, lon, a, o); if (d < best) { best = d; k = i; } });
    return [run.slice(0, k + 1).reverse(), run.slice(k)]
      .filter((h) => h.length > 1 && meters(h[0][0], h[0][1], h.at(-1)[0], h.at(-1)[1]) > 4);  // skip tiny pieces
  });

  floodLayer = L.layerGroup().addTo(map);
  const w = waterWidth();
  const opts = (role, extra) => ({ renderer: floodRenderer, interactive: false, lineCap: "round", lineJoin: "round", weight: Math.max(1.5, w * ROLE_WIDTH[role]), ...extra });

  // Ripples where the report was made
  L.circleMarker([lat, lon], { renderer: floodRenderer, radius: 14, color: "#2f8cff", weight: 3, fill: false, className: "ripple", interactive: false }).addTo(floodLayer);
  L.circleMarker([lat, lon], { renderer: floodRenderer, radius: 14, color: "#2f8cff", weight: 3, fill: false, className: "ripple ripple-2", interactive: false }).addTo(floodLayer);

  if (!halves.length) {
    L.circle([lat, lon], { renderer: floodRenderer, radius: 45, color: "#2f8cff", weight: 2, fillColor: "#2f8cff", fillOpacity: 0.35, className: "flood-pulse", interactive: false }).addTo(floodLayer);
    addFloodNote(year, `${icon("drop")}<b>Blue</b> = where flooding was reported during ${st.name} ${st.year} (street shape unavailable).`);
    return;
  }

  // Layers of water: soft glow → deep body → lighter surface, each "drawn on" from the report outward
  for (const half of halves) {
    for (const [role, style] of [
      ["glow",    { color: "#1e6fd9", opacity: 0.22 }],
      ["body",    { color: "#1565c0", opacity: 0.6 }],
      ["surface", { color: "#4fa3ff", opacity: 0.85 }],
    ]) {
      const line = L.polyline(half, opts(role, style)).addTo(floodLayer);
      waterLines.push({ line, role });
      startFill(line);
    }
  }
  // After the street has filled, add moving shimmer so the water looks alive
  setTimeout(() => {
    if (token !== pickToken || !floodLayer) return;
    for (const half of halves) {
      const line = L.polyline(half, opts("shimmer", { color: "#e8f3ff", opacity: 0.9, dashArray: "4 16", className: "flood-flow" })).addTo(floodLayer);
      waterLines.push({ line, role: "shimmer" });
    }
  }, FILL_MS);

  addFloodNote(year, `${icon("drop")}<b>Blue water</b> = the stretch of ${esc(street)} near 311 flood reports during ${st.name} ${st.year}. It shows where flooding was reported, not measured water depth.`);
}

// "Draw on" animation: hide the line with a dash as long as itself, then slide the dash in
function startFill(line) {
  const path = line.getElement();
  if (!path || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const len = path.getTotalLength();
  path.style.transition = "none";
  path.style.strokeDasharray = `${len} ${len}`;
  path.style.strokeDashoffset = `${len}`;
  path.getBoundingClientRect();                       // apply the hidden state before animating
  path.style.transition = `stroke-dashoffset ${FILL_MS}ms cubic-bezier(.3,.6,.3,1)`;
  path.style.strokeDashoffset = "0";
  line._fillTimer = setTimeout(() => finishFill(line), FILL_MS + 50);
}

// Clear the animation dash (needed before zooming, because Leaflet redraws the line)
function finishFill(line) {
  clearTimeout(line._fillTimer);
  const path = line.getElement();
  if (!path || line.options.dashArray) return;       // shimmer keeps its own dashes
  path.style.transition = "none";
  path.style.strokeDasharray = "";
  path.style.strokeDashoffset = "";
}

// Walk along the street in ~8 m steps; keep the parts within FLOOD_REACH_M of any report.
function floodedRuns(geometry, reports) {
  const runs = [];
  let run = [];
  for (let i = 1; i < geometry.length; i++) {
    const a = geometry[i - 1], b = geometry[i];
    const steps = Math.max(1, Math.ceil(meters(a.lat, a.lon, b.lat, b.lon) / 8));
    for (let s = i === 1 ? 0 : 1; s <= steps; s++) {
      const t = s / steps;
      const p = [a.lat + (b.lat - a.lat) * t, a.lon + (b.lon - a.lon) * t];
      const wet = reports.some(([ra, ro]) => meters(p[0], p[1], ra, ro) < FLOOD_REACH_M);
      if (wet) run.push(p);
      else if (run.length) { if (run.length > 1) runs.push(run); run = []; }
    }
  }
  if (run.length > 1) runs.push(run);
  return runs;
}

function meters(lat1, lon1, lat2, lon2) {
  const dx = (lon2 - lon1) * 111320 * Math.cos(((lat1 + lat2) / 2) * Math.PI / 180);
  const dy = (lat2 - lat1) * 110540;
  return Math.hypot(dx, dy);
}

// A line in the card (under the score) explaining the blue
function addFloodNote(year, html) {
  const cardEl = document.getElementById("card");
  let note = cardEl.querySelector(".flood-note");
  if (!note) {
    note = document.createElement("div");
    note.className = "flood-note";
    (cardEl.querySelector(".score") || cardEl.querySelector(".where"))?.after(note);
  }
  note.innerHTML = html;
}

// ---------- Street names: OpenStreetMap reverse lookup, max 1 request per second, cached ----------
const streetCache = new Map();
let geoQueue = Promise.resolve();
let lastGeo = 0;
function streetName(lat, lon) {
  const key = `${lat.toFixed(4)},${lon.toFixed(4)}`;
  if (streetCache.has(key)) return streetCache.get(key);
  const job = geoQueue.then(async () => {
    const wait = 1100 - (Date.now() - lastGeo);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastGeo = Date.now();
    try {
      // polygon_geojson=1 also returns the street's shape, used to paint the blue flood
      const d = await getJSON(`https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=17&polygon_geojson=1&lat=${lat}&lon=${lon}`);
      const a = d.address || {};
      const g = d.geojson || {};
      const toPts = (coords) => coords.map(([x, y]) => ({ lat: y, lon: x }));
      const lines = g.type === "LineString" ? [toPts(g.coordinates)]
        : g.type === "MultiLineString" ? g.coordinates.map(toPts) : [];
      return {
        street: a.road || a.pedestrian || a.residential || a.footway || "Unnamed street",
        area: a.neighbourhood || a.suburb || a.quarter || a.city_district || a.city || "",
        lines,
      };
    } catch {
      streetCache.delete(key);         // allow a retry later
      return { street: "Street name unavailable", area: "", lines: [] };
    }
  });
  geoQueue = job;
  streetCache.set(key, job);
  return job;
}

// ---------- Sidebar "Selected spot" panel + recently viewed streets ----------
const recent = [];
function showSpot({ lat, lon, c, storm, street, area, label }) {
  const panel = document.getElementById("spot-panel");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  const lvl = c ? levelFor(c.score) : null;
  const st = state.storms.find((s) => s.year === storm);
  document.getElementById("spot-info").innerHTML = `
    <div class="spot-street">${street ? esc(street) : `<span class="loading">Finding the street…</span>`}</div>
    ${area || label ? `<div class="spot-area">${esc([label && area && label.includes(area) ? "" : area, label && label !== street ? label : ""].filter(Boolean).join(" · "))}</div>` : ""}
    <div class="spot-row">
      ${lvl ? `<span class="level sm" style="background:${lvl.color};color:${lvl.text}">${lvl.label}</span><span>area flood-likeness</span>`
            : `<span class="fine">Outside the sampled area</span>`}
    </div>
    ${st ? `<p class="spot-report">${icon("rain")}Flooding reported here during <b>${st.name} ${st.year}</b></p>` : ""}`;

  if (!street || street.startsWith("Street name")) return;
  const i = recent.findIndex((r) => r.street === street);
  if (i >= 0) recent.splice(i, 1);
  recent.unshift({ lat, lon, street, area, storm });
  recent.length = Math.min(recent.length, 5);
  document.getElementById("spot-history").innerHTML = recent.length < 2 ? "" : `
    <p class="hint" style="margin:10px 0 4px">Recently viewed</p>
    ${recent.slice(1).map((r, k) => `<button class="recent" data-k="${k + 1}">${esc(r.street)}${r.area ? ` <span>${esc(r.area)}</span>` : ""}</button>`).join("")}`;
}
document.getElementById("spot-history").addEventListener("click", (e) => {
  const b = e.target.closest(".recent");
  if (!b) return;
  const r = recent[Number(b.dataset.k)];
  flyToSpot(r.lat, r.lon, r.storm ? 17.5 : 16);
  pickPoint(r.lat, r.lon, null, { storm: r.storm });
});

function selectCell(id, { fly = false, label = null } = {}) {
  const c = state.byId.get(id);
  if (!c) return;
  if (fly) map.flyTo([c.lat, c.lon], 14, { duration: 0.8 });
  pickPoint(c.lat, c.lon, label);
}

function showCard(c, placeLabel) {
  const lvl = levelFor(c.score);
  const s = c.n === 1 ? "" : "s";
  const maxAbs = Math.max(...Object.values(c.why).map(Math.abs), 0.01);

  const bars = Object.keys(FEATURES).map((k) => {
    const v = c.why[k];
    const w = Math.round((Math.abs(v) / maxAbs) * 50);
    return `<span>${FEATURES[k]}</span>
      <span class="track" title="${v >= 0 ? "Raises" : "Lowers"} the score">
        <span class="bar ${v >= 0 ? "up" : "down"}" style="width:${w}%"></span>
      </span>`;
  }).join("");

  const stormRows = state.storms.map((st) => {
    const n = c.reports[st.year] || 0;
    return `<tr><td>${st.name} ${st.year}</td><td class="${n ? "" : "zero"}">${n ? n : "none"}</td></tr>`;
  }).join("");

  card(`
    <button class="close" aria-label="Close">&times;</button>
    ${placeLabel ? `<p class="where">${icon("pin")}${esc(placeLabel)}</p>` : ""}
    <span class="level" style="background:${lvl.color};color:${lvl.text}">${lvl.label}</span>
    <p class="score">Score ${c.score.toFixed(2)} out of 1 · for the ~1 km area around the pin, averaged over ${c.n} sampled spot${s}</p>
    ${c.n < 3 ? `<div class="warn">Only ${c.n} sampled spot${s} here, so treat this score with extra caution.</div>` : ""}

    <h4>Why? Scout explains</h4>
    <div class="scout-says">
      <img src="assets/scout.png" alt="" class="avatar">
      <p class="bubble">${explain(c)}</p>
    </div>

    <h4>What pushed the score</h4>
    <div class="why">${bars}</div>
    <div class="why-axis"><span>◀ lowers</span><span>raises ▶</span></div>

    <h4>311 flood reports in this area</h4>
    <table class="storm-table">${stormRows}</table>

    <h4>The land here</h4>
    <dl class="facts">
      <dt>Elevation</dt><dd>${c.elev} m above sea level</dd>
      <dt>Nearest bayou</dt><dd>${c.bayou.toLocaleString()} m away</dd>
      <dt>Slope</dt><dd>${c.slope}°</dd>
      <dt>Land use</dt><dd>${c.land} development</dd>
      <dt>Near highways</dt><dd>${Math.round(c.hwy * 100)}% of the area</dd>
    </dl>
    <p class="fine">The score shows how much this area's land resembles spots with 311 flood reports in past storms. It's not a forecast.</p>
  `);
}

// Not sampled: say so honestly and offer the nearest sampled area
function showNoData(lat, lon, label) {
  const near = nearestCell(lat, lon);
  card(`
    <button class="close" aria-label="Close">&times;</button>
    ${label ? `<p class="where">${icon("pin")}${esc(label)}</p>` : ""}
    <div class="scout-says">
      <img src="assets/scout.png" alt="" class="avatar">
      <p class="bubble">Hmm, this spot is <b>outside the area we sampled</b>, so I don't have a score for it. That doesn't mean it's safe or risky, just that we don't know.</p>
    </div>
    ${near ? `<p>The nearest sampled area is <b>${near.km.toFixed(1)} km away</b> (it's a different place, so its score doesn't apply here).
      <br><button class="link-btn" id="go-near">Show it anyway</button></p>` : ""}
  `);
  document.getElementById("go-near")?.addEventListener("click", () => selectCell(near.cell.id, { fly: true }));
}

// Turn SHAP numbers into a sentence. A feature "raises" the score if its SHAP value is positive.
// Wording uses the actual value vs. the typical (median) area, so we never call a high place "low".
function phrase(k, c, up) {
  const m = state.medians;
  switch (k) {
    case "elev":
      if (up) return c.elev < m.elev ? `it sits low, about ${c.elev} m above sea level` : `its elevation (${c.elev} m) is like past flood spots`;
      return c.elev >= m.elev ? `it sits higher, about ${c.elev} m above sea level` : `its elevation (${c.elev} m) isn't typical of flood spots`;
    case "bayou":
      if (up) return c.bayou < m.bayou ? `it's only ${c.bayou.toLocaleString()} m from a bayou or stream` : `its distance to the nearest bayou (${c.bayou.toLocaleString()} m) matches past flood spots`;
      return c.bayou >= m.bayou ? `it's farther from bayous (${c.bayou.toLocaleString()} m)` : `its bayou distance (${c.bayou.toLocaleString()} m) isn't typical of flood spots`;
    case "slope":
      if (up) return c.slope < m.slope ? "the ground is very flat, so water has nowhere to go" : "its terrain slope matches past flood spots";
      return c.slope >= m.slope ? "the ground slopes more, which helps water drain" : "its terrain slope isn't typical of flood spots";
    case "hwy":
      if (up) return c.hwy >= m.hwy ? "much of the area is close to major highways" : "its closeness to highways matches past flood spots";
      return c.hwy < m.hwy ? "less of the area is near major highways" : "its closeness to highways isn't typical of flood spots";
    case "land":
      return up ? `its ${c.land.toLowerCase()} development matches past flood spots` : `its ${c.land.toLowerCase()} development lowers the score`;
  }
}

function explain(c) {
  const lvl = levelFor(c.score).label.toLowerCase();
  const sorted = Object.entries(c.why).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const ups = sorted.filter(([, v]) => v > 0.2).slice(0, 2).map(([k]) => phrase(k, c, true));
  const downs = sorted.filter(([, v]) => v < -0.2).slice(0, 2).map(([k]) => phrase(k, c, false));
  const join = (a) => a.length === 2 ? `${a[0]}, and ${a[1]}` : a[0];

  if (c.score >= 0.5) {
    let s = `This area scores <b>${lvl}</b> mainly because ${join(ups.length ? ups : ["its land features match past flood spots"])}.`;
    if (downs.length) s += ` On the plus side, ${downs[0]}.`;
    return s;
  }
  let s = `This area scores <b>${lvl}</b> mainly because ${join(downs.length ? downs : ["its land features don't match past flood spots"])}.`;
  if (ups.length) s += ` Still, ${ups[0]}.`;
  return s;
}

function closeCard() {
  document.getElementById("card").classList.add("hidden");
  if (state.pin) { map.removeLayer(state.pin); state.pin = null; }
  clearFlood();
  markHotspot(null);
}

// =====================================================================
// 5. Storm switcher: a small dot for every 311 flood report in that storm
// =====================================================================
function buildStormChips() {
  const box = document.getElementById("storm-chips");
  const options = [{ year: null, label: "None" }, ...state.storms.map((s) => ({ year: s.year, label: `${shortName(s.name)} ’${String(s.year).slice(2)}` }))];
  box.innerHTML = options.map((o) =>
    `<button class="chip" data-year="${o.year ?? ""}" aria-pressed="${o.year === null}">${o.label}</button>`).join("") +
    `<button class="chip chip-repeat" data-year="repeat" aria-pressed="false">${icon("repeat")}Repeat flooding</button>`;
  box.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    if (btn.dataset.year === "repeat") showRepeat();
    else showStorm(btn.dataset.year ? Number(btn.dataset.year) : null);
  });
}

function showStorm(year) {
  clearRepeat();
  state.activeStorm = year;
  document.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.year || null) == year)));
  clearFlood();
  if (state.stormLayer) map.removeLayer(state.stormLayer);
  state.stormLayer = null;

  const info = document.getElementById("storm-info");
  const panel = document.getElementById("storm-panel");
  const st = state.storms.find((s) => s.year === year);
  document.getElementById("legend-storm").classList.toggle("hidden", !st);
  document.getElementById("dot-hint").classList.toggle("hidden", !st);
  panel.classList.toggle("hidden", !st);
  if (!st) {
    scout(DEFAULT_MSG);
    return;
  }
  document.getElementById("legend-storm-name").textContent = `${st.name} ${st.year}`;

  // Small, soft dots. Hover one → street card; click one → zoom in to that spot.
  state.stormLayer = L.layerGroup(st.dots.map(([lat, lon]) => {
    const dot = L.circleMarker([lat, lon], { ...dotStyle(), renderer: dotRenderer, bubblingMouseEvents: false });
    dot.on("mouseover", () => hoverDotOn(dot, st));
    dot.on("mouseout", () => hoverDotOff(dot));
    dot.on("click", () => {
      hoverDotOff(dot);
      flyToSpot(lat, lon, 17.5);   // zoom right down to the street
      pickPoint(lat, lon, null, { storm: st.year });
    });
    return dot;
  })).addTo(map);

  // Plain-language result: "caught X out of every 100 places that flooded", with a fold-out for details
  const pct = Math.round(st.recall * 100);
  const prec = Math.round(st.precision * 100);
  const missed = st.reports - st.caught;
  const falseAlarms = st.flagged - st.caught;
  info.innerHTML = `
    <p class="result-head">How well did the model do?</p>
    <div class="big">${pct} <span>out of every 100</span></div>
    <p>places that flooded were <b>caught</b> by the model, using only the land.</p>
    <div class="meter" role="img" aria-label="${pct} percent caught"><span style="width:${pct}%"></span></div>
    <p class="fine">${st.test === "independent_test"
      ? "Imelda was kept completely hidden until the final test, so this is the fairest check."
      : "The model never saw this storm while it was learning."}</p>
    <details class="info-more">
      <summary>${icon("info")}What does this mean?</summary>
      <ul>
        <li><b>${st.reports.toLocaleString()}</b> places had a 311 flood report during ${st.name}.</li>
        <li>The model said "looks like a flood spot" (score ≥ 0.5) for <b>${st.caught.toLocaleString()}</b> of them, and missed <b>${missed.toLocaleString()}</b>.</li>
        <li><b>False alarms:</b> it also flagged ${falseAlarms.toLocaleString()} places with no report. Of everything it flagged, about <b>${prec} in 100</b> really had a report.</li>
        <li>Some "false alarms" may have flooded without anyone calling 311.</li>
      </ul>
      <p class="fine">In machine-learning terms: recall ${pct}%, precision ${prec}%.</p>
    </details>`;
  document.getElementById("storm-panel-title").textContent = `${st.name} ${st.year}`;
  scout(`During <b>${st.name} (${st.year})</b>, each small dot is a place where someone reported flooding to 311. <b>Hover a dot</b> to see its street, and <b>click it</b> to zoom right in.`);
}

// Dot look + a canvas renderer with a bigger "hit area", so tiny dots are easy to hover
// Dots shrink and fade when zoomed out (so they don't form a dark blob), full size when zoomed in
function dotStyle() {
  const z = map.getZoom();
  if (z < 11) return { radius: 2.4, stroke: false, fillColor: "#1d2b44", fillOpacity: 0.5 };
  if (z < 13) return { radius: 3.6, stroke: true, color: "#fff", weight: 0.8, fillColor: "#1d2b44", fillOpacity: 0.75 };
  return { radius: 5.5, stroke: true, color: "#fff", weight: 1.5, fillColor: "#1d2b44", fillOpacity: 0.9 };
}
map.on("zoomend", () => {
  if (!state.stormLayer) return;
  const s = dotStyle();
  state.stormLayer.eachLayer((d) => d.setStyle(s));
});
const dotRenderer = L.canvas({ tolerance: 6 });
const dotTip = L.tooltip({ direction: "top", offset: [0, -8], className: "dot-tip", opacity: 1 });
let hoverTimer = null;

function dotCard(lat, lon, st, place) {
  const c = state.byId.get(cellIdAt(lat, lon));
  const lvl = c ? levelFor(c.score) : null;
  return `
    <div class="dt-street">${icon("pin")}${place ? esc(place.street) : `<span class="loading">Finding street…</span>`}</div>
    ${place?.area ? `<div class="dt-area">${esc(place.area)}</div>` : ""}
    <div class="dt-row">${icon("rain")}Flooding reported · ${shortName(st.name)} ${st.year}</div>
    ${lvl ? `<div class="dt-row"><span class="level sm" style="background:${lvl.color};color:${lvl.text}">${lvl.label}</span> area flood-likeness</div>` : ""}
    <div class="dt-hint">Click to zoom in</div>`;
}

function hoverDotOn(dot, st) {
  state.hoverDot = dot;
  map.closeTooltip(hoverTip);
  map.getContainer().style.cursor = "pointer";
  dot.setStyle({ radius: 9, fillColor: "#f5b82e", fillOpacity: 1, stroke: true, color: "#6e3a0f", weight: 2 }); // sheriff-star gold
  const { lat, lng } = dot.getLatLng();
  dotTip.setLatLng([lat, lng]).setContent(dotCard(lat, lng, st, null)).addTo(map);
  // Look up the street only once the pointer rests on the dot (avoids lookups while sweeping across)
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    streetName(lat, lng).then((place) => {
      if (state.hoverDot === dot) dotTip.setContent(dotCard(lat, lng, st, place));
    });
  }, 250);
}

function hoverDotOff(dot) {
  clearTimeout(hoverTimer);
  dot.setStyle(dotStyle());
  if (state.hoverDot === dot) state.hoverDot = null;
  map.closeTooltip(dotTip);
}

const shortName = (n) => n.replace("Tropical Storm ", "").replace("Hurricane ", "").replace(" Flood", "");

// =====================================================================
// 5b. Repeat flooding: numbered pins on the 20 areas with the most 311 flood reports
//     across all 4 storms (spread at least 2.5 km apart), plus a sidebar summary.
// =====================================================================
const STORM_SHORT = { 2015: "Memorial Day ’15", 2016: "Tax Day ’16", 2017: "Harvey ’17", 2019: "Imelda ’19" };
const areaName = (t) => (t.name && t.name !== "Houston" ? t.name : `Area near ${t.lat.toFixed(3)}, ${t.lon.toFixed(3)}`);

function clearRepeat() {
  if (!state.repeatLayer) return;
  map.removeLayer(state.repeatLayer);
  state.repeatLayer = null;
  document.getElementById("repeat-panel").classList.add("hidden");
  document.getElementById("legend-repeat").classList.add("hidden");
  applyGlowOpacity();
}

// Small bar chart of an area's reports in each storm (used in the pin tooltip)
function stormBars(byStorm) {
  if (!byStorm) return "";
  const max = Math.max(...Object.values(byStorm), 1);
  return Object.entries(byStorm).map(([y, n]) => `
    <div class="sb-row"><span>${STORM_SHORT[y]}</span><span class="sb-bar" style="width:${Math.max(4, Math.round((n / max) * 100))}%"></span><b>${n}</b></div>`).join("");
}

function showRepeat() {
  showStorm(null);                                   // clear any storm dots first
  const { by_count: counts, top } = state.repeat;
  const maxReports = Math.max(...top.map((t) => t.reports));

  // Numbered pins: bigger = more flood reports in total
  const pins = top.map((t, i) => {
    const size = Math.round(26 + 16 * (t.reports / maxReports));
    const pin = L.marker([t.lat, t.lon], {
      icon: L.divIcon({ className: "rep-pin-wrap", html: `<div class="rep-pin" style="--s:${size}px">${i + 1}</div>`, iconSize: [size, size] }),
      riseOnHover: true, keyboard: false,
    });
    pin.bindTooltip(`
      <div class="rt-name">${i + 1}. ${esc(areaName(t))}</div>
      <div class="rt-sub">Flooded in all 4 storms · <b>${t.reports}</b> reports</div>
      <div class="sb">${stormBars(t.by_storm)}</div>
      <div class="dt-hint">Click to zoom in</div>`, { direction: "top", offset: [0, -size / 2], className: "rep-tip", opacity: 1 });
    pin.on("click", () => selectCell(t.id, { fly: true, label: `#${i + 1} ${areaName(t)} · flooded in all 4 storms` }));
    return pin;
  });
  state.repeatLayer = L.layerGroup(pins).addTo(map);
  applyGlowOpacity();
  map.flyToBounds(L.latLngBounds(top.map((t) => [t.lat, t.lon])), { padding: [70, 70], duration: 0.9 });
  document.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.year === "repeat")));
  document.getElementById("legend-repeat").classList.remove("hidden");

  const all4 = counts.find((x) => x.storms === 4);
  const maxScore = Math.max(...counts.map((x) => x.avg_score));
  document.getElementById("repeat-info").innerHTML = `
    <div class="big">${all4.areas.toLocaleString()} <span>areas</span></div>
    <p>had 311 flood reports in <b>all 4 storms</b>: the same places, flooding again and again. The numbered pins mark the 20 worst.</p>
    <p class="result-head" style="margin-top:12px">Top 20: most reports across all 4 storms</p>
    <ol class="rep-list">${top.map((t, i) => `
      <li><button data-i="${i}"><b class="rep-num">${i + 1}</b><span>${esc(areaName(t))}</span><small>${t.reports}</small></button></li>`).join("")}
    </ol>
    <p class="result-head" style="margin-top:14px">Average flood-likeness score</p>
    <div class="rep-chart">
      ${[...counts].reverse().map((x) => `
        <div class="rep-row"><span>${x.storms === 0 ? "No reports" : `${x.storms} storm${x.storms > 1 ? "s" : ""}`}</span>
          <span class="rep-bar s${x.storms}" style="width:${Math.round((x.avg_score / maxScore) * 100)}%"></span>
          <b>${x.avg_score.toFixed(2)}</b></div>`).join("")}
    </div>
    <p class="fine">The more storms an area flooded in, the higher the model scored its land, even though the model only looks at the land.</p>`;
  document.getElementById("repeat-panel").classList.remove("hidden");
  document.getElementById("repeat-panel").scrollIntoView({ block: "nearest", behavior: "smooth" });
  scout("The <b>numbered pins</b> are the 20 areas that flooded in <b>every one of the 4 storms</b> and had the most reports. <b>Hover</b> a pin to see each storm, <b>click</b> to zoom in.");
}

document.getElementById("repeat-info").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-i]");
  if (!b) return;
  const t = state.repeat.top[Number(b.dataset.i)];
  selectCell(t.id, { fly: true, label: `#${Number(b.dataset.i) + 1} ${areaName(t)} · flooded in all 4 storms` });
});

// =====================================================================
// 6. Top 10 hotspots
// =====================================================================
function buildHotspots(list) {
  const ol = document.getElementById("hotspots");
  ol.innerHTML = list.map((h) =>
    `<li><button data-id="${esc(h.id)}"><span class="name">${esc(h.name || `Near ${h.lat.toFixed(2)}, ${h.lon.toFixed(2)}`)}</span><span class="sc">${h.score.toFixed(2)}</span></button></li>`
  ).join("");
  ol.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const h = list.find((x) => x.id === btn.dataset.id);
    selectCell(h.id, { fly: true, label: h.name ? `${h.name} · hotspot #${list.indexOf(h) + 1}` : null });
  });
}

function markHotspot(id) {
  document.querySelectorAll(".hotspots button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
}

// =====================================================================
// 7. Place search (OpenStreetMap Nominatim, limited to the Houston area)
// =====================================================================
const searchForm = document.getElementById("search");
const searchInput = document.getElementById("search-input");
const resultsBox = document.getElementById("search-results");
const VIEWBOX = "-95.85,30.25,-94.95,29.40"; // west,north,east,south around the study area

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = searchInput.value.trim();
  if (!q) return;
  resultsBox.innerHTML = `<li class="empty">Searching…</li>`;
  resultsBox.classList.remove("hidden");
  try {
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&bounded=1&viewbox=${VIEWBOX}&q=${encodeURIComponent(q)}`;
    const places = await getJSON(url);
    if (!places.length) {
      resultsBox.innerHTML = `<li class="empty">No places found in the Houston area. Try a street, park or landmark name.</li>`;
      return;
    }
    resultsBox.innerHTML = places.map((p, i) =>
      `<li role="option" tabindex="0" data-i="${i}">${esc(p.display_name.split(", ").slice(0, 3).join(", "))}</li>`
    ).join("") + `<li class="attr">Search by OpenStreetMap Nominatim</li>`;
    resultsBox.onclick = resultsBox.onkeydown = (ev) => {
      const li = ev.target.closest("li[data-i]");
      if (!li || (ev.type === "keydown" && ev.key !== "Enter")) return;
      const p = places[li.dataset.i];
      const lat = Number(p.lat), lon = Number(p.lon);
      map.flyTo([lat, lon], 14, { duration: 0.8 });
      pickPoint(lat, lon, p.display_name.split(", ").slice(0, 2).join(", "));
      resultsBox.classList.add("hidden");
    };
  } catch (err) {
    console.error(err);
    resultsBox.innerHTML = `<li class="empty">Search isn't available right now. You can still click the map.</li>`;
  }
});
document.addEventListener("click", (e) => { if (!searchForm.contains(e.target)) resultsBox.classList.add("hidden"); });

function nearestCell(lat, lon) {
  let best = null;
  for (const c of state.cells) {
    const km = Math.hypot((c.lat - lat) * 111, (c.lon - lon) * 96.5);
    if (!best || km < best.km) best = { cell: c, km };
  }
  return best;
}

// =====================================================================
// 8. Scout's messages, welcome and About screens
// =====================================================================
const DEFAULT_MSG = "Howdy! <b>Click anywhere on the map</b> and I'll explain that area's color. Pick a <b>past storm</b> from the bar on the map to see where people reported flooding.";

function scout(html) { document.getElementById("scout-msg").innerHTML = html; }

function buildAboutStats() {
  const im = state.storms.find((s) => s.test === "independent_test");
  if (!im) return;
  document.getElementById("about-stats").innerHTML = `
    <div><b>${Math.round(im.recall * 100)}%</b>of Imelda's reported flood spots flagged (${im.caught.toLocaleString()} of ${im.reports.toLocaleString()})</div>
    <div><b>${Math.round(im.accuracy * 100)}%</b>of Imelda's spots classified correctly overall</div>
    <div><b>${state.storms.length}</b>storms studied, 2015–2019</div>
    <div><b>${state.cells.length.toLocaleString()}</b>~1 km areas scored</div>`;
}

function openModal(id) { document.getElementById(id).classList.remove("hidden"); }
function closeModal(id) { document.getElementById(id).classList.add("hidden"); }

// ---------- Tabs: Flood Map · The Project · How We Built It (also reachable as #map, #project, #build) ----------
function showView(name) {
  if (!["map", "project", "build"].includes(name)) name = "map";
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${name}`));
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === name)));
  document.body.dataset.view = name;
  if (name === "map") setTimeout(() => map.invalidateSize(), 0);   // the map was hidden, so let Leaflet re-measure
  else document.getElementById(`view-${name}`).scrollTop = 0;
  // The map is the home page, so it keeps a clean address (no "#map"); the other tabs get #project / #build
  const want = name === "map" ? "" : `#${name}`;
  if (location.hash !== want) history.replaceState(null, "", location.pathname + location.search + want);
}
document.querySelectorAll(".tab, [data-goto]").forEach((el) =>
  el.addEventListener("click", () => showView(el.dataset.view || el.dataset.goto)));
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
showView(location.hash.slice(1) || "map");
document.getElementById("brand").onclick = () => openModal("welcome");
document.getElementById("welcome-go").onclick = () => closeModal("welcome");
document.querySelectorAll(".modal").forEach((m) => m.addEventListener("click", (e) => {
  if (e.target === m || e.target.closest("[data-close]")) m.classList.add("hidden");
}));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
  closeCard();
});

// Show Scout's welcome every time someone opens the map (not when they land directly on The Project / How We Built It)
if (document.body.dataset.view === "map") openModal("welcome");

// =====================================================================
// 9. Small helpers
// =====================================================================
function card(html) {
  const el = document.getElementById("card");
  el.innerHTML = html;
  el.classList.remove("hidden");
  el.scrollTop = 0;
  el.querySelector(".close").onclick = closeCard;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// Inline line icon from the SVG sprite in index.html
function icon(name) {
  return `<svg class="i" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

// Legend: collapse/expand the explanation
document.getElementById("legend-toggle").addEventListener("click", () => {
  const lg = document.getElementById("legend");
  const collapsed = lg.classList.toggle("collapsed");
  document.getElementById("legend-toggle").setAttribute("aria-expanded", String(!collapsed));
});

// =====================================================================
// 10. Street photos (sidebar)
//     1) Flood photos taken near the street, from Wikimedia Commons (free to reuse, credited).
//     2) If none are close by: a real aerial photo of the street (Esri World Imagery),
//        plus a link to see it at street level in Google Street View.
// =====================================================================
const STORM_WORDS = { 2015: /memorial day/i, 2016: /tax day/i, 2017: /harvey/i, 2019: /imelda/i };
const PHOTO_RADIUS_KM = 3;
const NEAR_PHOTO_M = 1000;   // only show flood photos taken this close to the clicked spot
const photoCache = new Map();
let aerialMap = null;

async function findFloodPhotos(lat, lon) {
  const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
  if (photoCache.has(key)) return photoCache.get(key);
  // One small search per keyword (Commons' search returns nothing for some long OR lists), merged
  const url = (word) => "https://commons.wikimedia.org/w/api.php?action=query&format=json&origin=*" +
    "&generator=search&gsrnamespace=6&gsrlimit=20&gsrsearch=" +
    encodeURIComponent(`${word} nearcoord:${PHOTO_RADIUS_KM}km,${lat},${lon}`) +
    "&prop=imageinfo|coordinates&iiprop=url|extmetadata&iiurlwidth=560";
  const job = Promise.all(["flood", "flooding", "harvey", "imelda"].map((w) => getJSON(url(w)).catch(() => ({}))))
    .then((results) => {
      const byId = new Map();
      for (const res of results) for (const p of Object.values(res.query?.pages || {})) byId.set(p.pageid, p);
      return [...byId.values()];
    })
    .then((pages) => pages
    .filter((p) => p.imageinfo?.[0]?.thumburl && p.coordinates?.[0])
    .map((p) => {
      const info = p.imageinfo[0], meta = info.extmetadata || {};
      const text = (v) => (v?.value || "").replace(/<[^>]*>/g, "").trim();
      return {
        title: p.title.replace(/^File:/, "").replace(/\.\w+$/, "").replace(/\s*\(\d+\)$/, ""),
        thumb: info.thumburl,
        page: info.descriptionurl,
        artist: text(meta.Artist) || "Unknown author",
        license: text(meta.LicenseShortName),
        date: text(meta.DateTimeOriginal).slice(0, 10),
        words: `${p.title} ${text(meta.ImageDescription)} ${text(meta.Categories)}`,
        meters: meters(lat, lon, p.coordinates[0].lat, p.coordinates[0].lon),
      };
    })
    .filter((ph) => /flood|harvey|imelda|high water|tax day|memorial day/i.test(ph.words))
  ).catch(() => []);
  photoCache.set(key, job);
  return job;
}

async function showPhotos(lat, lon, street, storm, token) {
  const panel = document.getElementById("photo-panel");
  const body = document.getElementById("photo-body");
  panel.classList.remove("hidden");

  // 1) Always: a real aerial photo of THIS exact spot (changes with every click)
  const streetView = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lon}`;
  body.innerHTML = `
    <p class="hint">${esc(street)} from above</p>
    <div id="aerial" class="aerial" role="img" aria-label="Aerial photo of ${esc(street)}"></div>
    <p class="photo-credit">Aerial photo: Esri World Imagery. The gold dot marks the spot.</p>
    <a class="sv-link" href="${streetView}" target="_blank" rel="noopener">${icon("eye")}See ${esc(street)} at street level ↗</a>
    <div id="flood-photos"><p class="loading">Checking for flood photos taken nearby…</p></div>`;
  if (aerialMap) aerialMap.remove();
  aerialMap = L.map("aerial", {
    zoomControl: false, attributionControl: false, dragging: false, scrollWheelZoom: false,
    doubleClickZoom: false, boxZoom: false, keyboard: false, touchZoom: false,
  }).setView([lat, lon], 18);
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }).addTo(aerialMap);
  L.circleMarker([lat, lon], { radius: 7, color: "#fff", weight: 2, fillColor: "#f5b82e", fillOpacity: 1 }).addTo(aerialMap);

  // 2) Only if one exists close by (within NEAR_PHOTO_M): real flood photos from Wikimedia Commons
  let photos = await findFloodPhotos(lat, lon);
  if (token !== pickToken) return;
  const box = document.getElementById("flood-photos");
  const match = STORM_WORDS[storm];
  // Rank: describes water/flooding › from the selected storm › closest; "aftermath" (relief) photos last
  const rank = (p) => (/flood|water|inundat|rain/i.test(p.words) ? 2 : 0) + (match && match.test(p.words) ? 1 : 0) - (/aftermath|relief|rabbi|volunteer/i.test(p.words) ? 4 : 0);
  photos = photos.filter((p) => p.meters <= NEAR_PHOTO_M && rank(p) >= 0)
    .sort((a, b) => rank(b) - rank(a) || a.meters - b.meters).slice(0, 3);
  const away = (m) => `${Math.round(m / 10) * 10} m`;

  if (!photos.length) {
    box.innerHTML = `<p class="photo-credit">No flood photos taken within 1 km of this spot are available online.</p>`;
    return;
  }
  box.innerHTML = `
    <p class="hint photo-sub">Flood photos taken nearby</p>
    <div class="photo-row">${photos.map((p) => `
      <a href="${esc(p.page)}" target="_blank" rel="noopener" title="${esc(p.title)}${p.date ? ` (${esc(p.date)})` : ""}">
        <img src="${esc(p.thumb)}" alt="${esc(p.title)}" loading="lazy"><span>${away(p.meters)} away</span></a>`).join("")}</div>
    <p class="photo-credit">Photos: ${photos.map((p) => `${esc(p.artist)}${p.license ? ` (${esc(p.license)})` : ""}`).join("; ")}, via Wikimedia Commons.</p>`;
}

// Small screens: start with the legend folded so it doesn't cover the map
if (innerHeight < 760 || innerWidth < 820) {
  document.getElementById("legend").classList.add("collapsed");
  document.getElementById("legend-toggle").setAttribute("aria-expanded", "false");
}

// Long pages: build "jump to section" chips from each page's headings
document.querySelectorAll(".page").forEach((page) => {
  const heads = [...page.querySelectorAll(".page-inner h2")];
  const nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "Jump to section");
  heads.forEach((h, i) => {
    h.id = h.id || `${page.id}-s${i}`;
    const b = document.createElement("button");
    b.textContent = h.textContent.replace(/\s*\(.*\)$/, "");   // short label: drop "(…)"
    b.addEventListener("click", () => page.scrollTo({ top: h.offsetTop - nav.offsetHeight - 12, behavior: "smooth" }));
    nav.appendChild(b);
  });
  page.querySelector(".page-hero").after(nav);
  // Highlight the section you're reading
  page.addEventListener("scroll", () => {
    let cur = 0;
    heads.forEach((h, i) => { if (h.offsetTop - nav.offsetHeight - 24 <= page.scrollTop) cur = i; });
    nav.querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === cur));
  }, { passive: true });
});
