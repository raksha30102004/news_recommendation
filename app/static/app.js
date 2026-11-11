"use strict";

/* ------------------------------------------------------------------ utilities */
const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString();
const safe = (fn, fallback = null) => { try { return fn(); } catch { return fallback; } };

let memUid = null; // fallback when storage is blocked (e.g. Safari in the Hugging Face Spaces iframe)
const session = {
  get uid() { return safe(() => localStorage.getItem("uid")) ?? memUid; },
  set uid(v) { memUid = v || null; safe(() => (v ? localStorage.setItem("uid", v) : localStorage.removeItem("uid"))); },
  reasons: {}, // article id -> "why recommended", for the article page
};

let me = null;
let feed = null;        // cached recommendations
let feedDirty = true;   // set after a click so the next Home visit re-ranks
let readSinceFeed = 0;

async function api(path, { method = "GET", body, admin } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (session.uid) headers["X-User-Id"] = session.uid;
  if (admin) headers["X-Admin-Token"] = admin;
  const res = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  if (res.status === 401) { signOut(); throw new Error("signed out"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------------ category art */
const PALETTE = {
  news: ["#3d5a80", "#28394f"], sports: ["#4f8a5b", "#2f5c3a"], finance: ["#2c7a7b", "#1d4f50"],
  foodanddrink: ["#c27c2c", "#8a5113"], lifestyle: ["#b5657a", "#7f3f52"], travel: ["#4a8bb8", "#2c5f86"],
  video: ["#5b5fa8", "#3a3d78"], weather: ["#3f9bb0", "#24697a"], health: ["#3f9a7a", "#256650"],
  autos: ["#5f6670", "#3a3f46"], tv: ["#7b5ea7", "#523c78"], music: ["#a55497", "#723669"],
  movies: ["#9b4d3c", "#6a2f23"], entertainment: ["#8a55b0", "#5d357c"], kids: ["#c9a227", "#8f7213"],
};
const ICONS = {
  news: '<path d="M4 5h12v14H6.5A2.5 2.5 0 0 1 4 16.5z"/><path d="M16 9h3.5v7.5A2.5 2.5 0 0 1 17 19"/><path d="M7.5 9h5M7.5 12.5h5M7.5 15.5h3"/>',
  sports: '<circle cx="12" cy="12" r="8"/><path d="M12 4v16M4 12h16M6.4 6.4c2.9 3 2.9 8.2 0 11.2M17.6 6.4c-2.9 3-2.9 8.2 0 11.2"/>',
  finance: '<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  foodanddrink: '<path d="M5 8h12v5a6 6 0 0 1-12 0z"/><path d="M17 9.5h1.5a2.5 2.5 0 0 1 0 5H17"/><path d="M8 3.5v2M11 3.5v2M14 3.5v2"/>',
  lifestyle: '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M18.5 15.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>',
  travel: '<path d="M12 21s-7-6.2-7-12a7 7 0 0 1 14 0c0 5.8-7 12-7 12z"/><circle cx="12" cy="9" r="2.5"/>',
  video: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="m10 9 5 3-5 3z"/>',
  weather: '<path d="M7 18.5h10a4 4 0 0 0 .4-8 6 6 0 0 0-11.5 1.8A3.2 3.2 0 0 0 7 18.5z"/>',
  health: '<path d="M20.4 6.6a5 5 0 0 0-8.4-1.4 5 5 0 0 0-8.4 1.4C2.3 11 12 20 12 20s9.7-9 8.4-13.4z"/><path d="M7 12h2.5l1.5-2.5 2 5 1.5-2.5H17"/>',
  autos: '<path d="M5 16v-4.5L7 6.5h10l2 5V16"/><rect x="3" y="15.5" width="18" height="3" rx="1"/><circle cx="7.5" cy="13" r=".8"/><circle cx="16.5" cy="13" r=".8"/>',
  tv: '<rect x="3" y="7" width="18" height="12.5" rx="2"/><path d="m8.5 3 3.5 4 3.5-4"/>',
  music: '<path d="M9 18V5.5l11-2V16"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
  movies: '<rect x="3" y="8.5" width="18" height="11.5" rx="1.5"/><path d="m3.5 8.5 2-4.5h14l-1.5 4.5M9 4l-1.8 4.5M14 4l-1.8 4.5"/>',
  entertainment: '<path d="m12 3.5 2.6 5.3 5.8.8-4.2 4.1 1 5.8L12 16.8l-5.2 2.7 1-5.8-4.2-4.1 5.8-.8z"/>',
  kids: '<circle cx="12" cy="12" r="8.5"/><path d="M8.5 14a4 4 0 0 0 7 0M9 9.5h.01M15 9.5h.01"/>',
  _: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5a13 13 0 0 1 0 17M12 3.5a13 13 0 0 0 0 17"/>',
};
const UI = {
  gear: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
  back: '<svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/></svg>',
  ext: '<svg viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>',
  bookmark: '<svg viewBox="0 0 24 24"><path d="M6 3.5h12v17l-6-4.5-6 4.5z"/></svg>',
  chev: '<svg viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>',
};
const art = (cat) => {
  const [a, b] = PALETTE[cat] || ["#6b7280", "#3f444c"];
  return { bg: `linear-gradient(135deg, ${a}, ${b})`, icon: `<svg viewBox="0 0 24 24">${ICONS[cat] || ICONS._}</svg>` };
};

/* ------------------------------------------------------------------ shared rendering */
function itemHTML(a, src, { rank, meta, reason } = {}) {
  const { bg, icon } = art(a.category);
  return `<li class="item" tabindex="0" role="link" data-id="${esc(a.id)}" data-src="${src}">
    <div class="thumb" style="background:${bg}">${icon}${rank ? `<span class="rank">${rank}</span>` : ""}</div>
    <div>
      <h3 class="ititle">${a.is_new ? '<span class="badge">New</span>' : ""}${esc(a.title)}</h3>
      <div class="imeta">${esc(meta ?? `${a.category_label} · ${a.subcategory_label}`)}</div>
      ${reason ? `<div class="ireason">${esc(reason)}</div>` : ""}
    </div></li>`;
}
const listHTML = (items, src, opts = () => ({})) => `<ul class="list">${items.map((a, i) => itemHTML(a, src, opts(a, i))).join("")}</ul>`;
const skeleton = (n = 6) => `<ul class="list skeleton" aria-hidden="true">${'<li class="item"><div class="thumb"></div><div><div class="sk" style="width:92%"></div><div class="sk" style="width:70%"></div><div class="sk" style="width:35%;height:10px"></div></div></li>'.repeat(n)}</ul>`;
const empty = (title, body) => `<div class="empty"><strong>${title}</strong>${body}</div>`;
const errorBox = (e) => empty("Something went wrong", esc(e.message));

function header({ title = "News", back = false, search = true, q = "" } = {}) {
  return `<header class="top">
    <div class="titlebar">
      ${back ? `<button class="iconbtn" data-act="back" aria-label="Back">${UI.back}</button>` : "<span></span>"}
      <h1>${esc(title)}</h1>
      <a class="iconbtn" href="#/profile" aria-label="Settings">${UI.gear}</a>
    </div>
    ${search ? `<form class="search" role="search" data-act="search">${UI.search}<input name="q" type="search" placeholder="Search" value="${esc(q)}" aria-label="Search articles" autocomplete="off"></form>` : ""}
  </header>`;
}

function setNav(name) {
  $("#tabbar").hidden = !me;
  document.querySelectorAll("#tabbar a").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
}

/* ------------------------------------------------------------------ views */
async function viewOnboarding() {
  setNav(null);
  view.innerHTML = `<section class="hero-on"><h1>News, ranked for you.</h1>
      <p>A personalised feed that learns from what you read, using an ALS collaborative-filtering model trained on Microsoft MIND.</p></section>
    <form class="card" id="newUser"><h2>Start fresh</h2><p>Pick a few topics to warm up your feed. It personalises as you read.</p>
      <input class="field" name="name" placeholder="Your name" maxlength="40" required autocomplete="given-name">
      <div class="chips" id="interestChips">${skeletonChips()}</div>
      <button class="btn primary" type="submit">Create my feed</button></form>
    <section class="card"><h2>Or browse as a real MIND reader</h2><p>Jump into the account of an actual user from the dataset and see what ALS recommends from their click history.</p>
      <div id="demoUsers"></div></section>
    <p class="note" style="padding:0 20px 24px">Profiles are anonymous and stored only on this server; your browser keeps a random ID.</p>`;
  const [cats, demos] = await Promise.all([api("/api/categories"), api("/api/demo-users")]);
  $("#interestChips").innerHTML = cats.map((c) => `<button type="button" class="chip" aria-pressed="false" data-cat="${esc(c.id)}">${esc(c.label)}</button>`).join("");
  $("#demoUsers").innerHTML = demos.map((d) => `<button class="demo" data-mind="${esc(d.mind_user_id)}">
      <div><b>Reader ${esc(d.mind_user_id)}</b><small>${d.clicks} articles read · mostly ${esc(d.top_categories.map(labelOf).join(", "))}</small></div>${UI.chev}</button>`).join("");
  $("#interestChips").addEventListener("click", (e) => {
    const c = e.target.closest(".chip");
    if (c) c.setAttribute("aria-pressed", c.getAttribute("aria-pressed") !== "true");
  });
  $("#newUser").addEventListener("submit", async (e) => {
    e.preventDefault();
    const interests = [...document.querySelectorAll('#interestChips [aria-pressed="true"]')].map((c) => c.dataset.cat);
    await signIn({ name: e.target.name.value, interests });
  });
  $("#demoUsers").addEventListener("click", (e) => {
    const b = e.target.closest(".demo");
    if (b) signIn({ name: `Reader ${b.dataset.mind}`, interests: [], mind_user_id: b.dataset.mind });
  });
}
const skeletonChips = () => '<span class="chip" style="width:80px">&nbsp;</span>'.repeat(6);
const LABELS = { foodanddrink: "Food & Drink", tv: "TV" };
const labelOf = (c) => LABELS[c] || c.charAt(0).toUpperCase() + c.slice(1);

async function signIn(body) {
  try {
    const u = await api("/api/users", { method: "POST", body });
    session.uid = u.id;
    me = u;
    feed = null; feedDirty = true;
    location.hash = "#/home";
    route();
  } catch (e) { toast(e.message); }
}
function signOut() {
  session.uid = null; me = null; feed = null;
  location.hash = "#/";
  route();
}

async function viewHome(tab = "recommended") {
  setNav("home");
  const tabs = [["recommended", "Recommended<br>Articles"], ["seen", "Seen<br>Before"], ["trending", "Trending<br>Articles"]];
  view.innerHTML = header() + `<div class="top" style="position:static;padding-top:0"><div class="tabs" role="tablist">${tabs
    .map(([k, l]) => `<button role="tab" aria-selected="${k === tab}" data-tab="${k}">${l}</button>`).join("")}</div></div><div id="feed">${skeleton()}</div>`;
  view.querySelector(".tabs").addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (b) location.hash = b.dataset.tab === "recommended" ? "#/home" : `#/home/${b.dataset.tab}`;
  });
  const box = $("#feed");
  try {
    if (tab === "recommended") {
      const refreshed = feedDirty && feed !== null && readSinceFeed > 0;
      if (feedDirty || !feed) { feed = await api("/api/feed/recommended"); feedDirty = false; }
      feed.forEach((a) => (session.reasons[a.id] = a.reason));
      box.innerHTML = `<div class="section-h"><h2>Top ${feed.length} for you</h2><span>ALS · updated live</span></div>` +
        listHTML(feed, "recommended", (a) => ({ reason: a.reason })) +
        `<div class="foot"><button class="btn" data-act="refresh">${UI.refresh} Refresh</button></div>`;
      if (refreshed) toast("Feed re-ranked with what you just read");
      readSinceFeed = 0;
    } else if (tab === "seen") {
      const items = await api("/api/feed/seen");
      box.innerHTML = items.length ? listHTML(items, "seen")
        : empty("Nothing here yet", "Articles you open will show up here, and they shape your recommendations.");
    } else {
      const items = await api("/api/feed/trending");
      box.innerHTML = `<div class="section-h"><h2>Most clicked today</h2></div>` +
        listHTML(items, "trending", (a, i) => ({ rank: i + 1, meta: `${a.category_label} · ${fmt(a.clicks_24h)} clicks` }));
    }
  } catch (e) { box.innerHTML = errorBox(e); }
}

async function viewArticle(id) {
  setNav(null);
  $("#tabbar").hidden = false;
  view.innerHTML = header({ title: "", back: true, search: false }) + `<div class="article">${skeleton(1)}</div>`;
  try {
    const a = await api(`/api/articles/${encodeURIComponent(id)}`);
    const { bg, icon } = art(a.category);
    const why = session.reasons[a.id];
    const isMind = /assets\.msn\.com\/labs\/mind/.test(a.url) || !a.url;
    const link = isMind ? `https://www.bing.com/news/search?q=${encodeURIComponent(a.title)}` : a.url;
    view.querySelector(".article").innerHTML = `
      <div class="thumb hero" style="background:${bg}">${icon}</div>
      <div class="kicker">${esc(a.category_label)} · ${esc(a.subcategory_label)}</div>
      <h1>${esc(a.title)}</h1>
      <p class="abs">${esc(a.abstract)}</p>
      <div class="actions">
        <a class="btn primary" href="${esc(link)}" target="_blank" rel="noopener">${isMind ? "Find the full story" : "Read the full story"} ${UI.ext}</a>
        <button class="btn" data-act="bookmark" data-id="${esc(a.id)}" aria-pressed="${a.bookmarked}">${UI.bookmark} <span>${a.bookmarked ? "Saved" : "Save"}</span></button>
      </div>
      ${why ? `<div class="why"><b>Why you're seeing this.</b> ${esc(why)}.</div>` : ""}
      ${isMind ? '<p class="note">MIND (2019) articles no longer have live MSN pages, so this link searches for the story.</p>' : ""}
      ${a.related.length ? `<div class="section-h" style="padding:20px 0 0"><h2>More like this</h2></div>` +
        listHTML(a.related, "related").replace('class="list"', 'class="list" style="padding:6px 0"') : ""}`;
  } catch (e) { view.querySelector(".article").innerHTML = errorBox(e); }
}

async function viewExplore(cat, sub) {
  setNav("explore");
  view.innerHTML = header({ title: "Explore", back: !!cat }) + `<div id="ex">${skeleton()}</div>`;
  const box = $("#ex");
  try {
    if (!cat) {
      const cats = await api("/api/categories");
      box.innerHTML = `<div class="section-h"><h2>Browse topics</h2></div><div class="catgrid">${cats.map((c) => {
        const { bg, icon } = art(c.id);
        return `<a class="cat" href="#/explore/${esc(c.id)}" style="background:${bg}">${icon}<div><b>${esc(c.label)}</b><br><small>${fmt(c.count)} articles</small></div></a>`;
      }).join("")}</div>`;
      return;
    }
    const data = await api(`/api/explore/${encodeURIComponent(cat)}${sub ? `?sub=${encodeURIComponent(sub)}` : ""}`);
    const chips = [`<a class="chip" href="#/explore/${esc(cat)}" aria-pressed="${!sub}">All</a>`]
      .concat(data.subcategories.map((s) => `<a class="chip" href="#/explore/${esc(cat)}/${esc(s.id)}" aria-pressed="${s.id === sub}">${esc(s.label)}</a>`));
    box.innerHTML = `<div class="section-h"><h2>${esc(labelOf(cat))}</h2><span>most read</span></div><div class="chips">${chips.join("")}</div>` +
      (data.articles.length ? listHTML(data.articles, "explore") : empty("No articles", ""));
  } catch (e) { box.innerHTML = errorBox(e); }
}

async function viewSearch(q) {
  setNav("explore");
  view.innerHTML = header({ title: "Search", back: true, q }) + `<div id="sr">${skeleton()}</div>`;
  try {
    const items = await api(`/api/search?q=${encodeURIComponent(q)}`);
    $("#sr").innerHTML = items.length ? `<div class="section-h"><h2>Results for “${esc(q)}”</h2><span>${items.length}</span></div>` + listHTML(items, "search")
      : empty("No matches", `Nothing found for “${esc(q)}”. Try fewer or different words.`);
  } catch (e) { $("#sr").innerHTML = errorBox(e); }
}

async function viewBookmarks() {
  setNav("bookmarks");
  view.innerHTML = header({ title: "Bookmarks", search: false }) + `<div id="bm">${skeleton(3)}</div>`;
  try {
    const items = await api("/api/bookmarks");
    $("#bm").innerHTML = items.length ? listHTML(items, "bookmark") : empty("No saved articles", "Tap Save on any article to keep it here.");
  } catch (e) { $("#bm").innerHTML = errorBox(e); }
}

async function viewProfile() {
  setNav("profile");
  view.innerHTML = header({ title: "Profile", search: false }) + `<div id="pf">${skeleton(2)}</div>`;
  try {
    const [u, cats, m] = await Promise.all([api("/api/me"), api("/api/categories"), api("/api/model")]);
    me = u;
    const row = (name, r, cls = "") => `<tr class="${cls}"><td>${name}</td><td>${r.auc.toFixed(4)}</td><td>${r.mrr.toFixed(4)}</td><td>${r["ndcg@5"].toFixed(4)}</td><td>${r["ndcg@10"].toFixed(4)}</td></tr>`;
    const rep = m.reproduced;
    $("#pf").innerHTML = `
      <section class="card"><h2>${esc(u.name)}</h2>
        <p>${u.mind_user_id ? `Linked to MIND reader ${esc(u.mind_user_id)}` : "Anonymous profile"}</p>
        <div class="stats"><div class="stat"><b>${fmt(u.history_size)}</b><span>articles read</span></div>
          <div class="stat"><b>${fmt(u.bookmarks)}</b><span>saved</span></div>
          <div class="stat"><b>${u.interests.length}</b><span>topics followed</span></div></div></section>
      <section class="card"><h2>Topics you follow</h2><p>Used as a prior for category boosting. It matters most before you've read much.</p>
        <div class="chips" id="myInterests">${cats.map((c) => `<button type="button" class="chip" aria-pressed="${u.interests.includes(c.id)}" data-cat="${esc(c.id)}">${esc(c.label)}</button>`).join("")}</div>
        <button class="btn" data-act="save-interests">Save topics</button></section>
      <section class="card"><h2>How your feed is ranked</h2>
        <p>${esc(m.algorithm)}. Score = cosine(user, article) + ${m.boost.category}·P(category) + ${m.boost.subcategory}·P(subcategory), top 10 with at most 3 per subcategory and 6 per category for diversity.</p>
        <table class="metrics"><thead><tr><th>MIND-small dev</th><th>AUC</th><th>MRR</th><th>nDCG@5</th><th>nDCG@10</th></tr></thead><tbody>
          ${rep ? row("This deployment", rep.als_category_boost, "hl") + row("ALS without boost", rep.als) : ""}
          ${Object.entries(m.report).map(([k, r]) => row(`Report: ${k}`, r)).join("")}
        </tbody></table>
        <p class="note">${rep ? `Measured on ${fmt(m.eval_impressions)} held-out dev impressions (boost weights tuned on a separate 10% slice).` : "Run <code>python train.py</code> to measure."}</p>
        <dl class="kv">
          <dt>Hyperparameters</dt><dd>${Object.entries(m.params).map(([k, v]) => `${k}=${v}`).join(", ")}</dd>
          <dt>Training data</dt><dd>${fmt(m.dataset.users)} users · ${fmt(m.dataset.articles)} articles · ${fmt(m.dataset.interactions)} clicks</dd>
          <dt>Model trained</dt><dd>${esc(new Date(m.model.trained_at).toLocaleString())} · ${esc(m.model.source)}</dd>
          <dt>Retraining</dt><dd>${m.retrain_interval_hours > 0 ? `every ${m.retrain_interval_hours}h` : "manual only"}${m.retraining ? " · running now…" : ""}</dd>
          <dt>Live feed</dt><dd>${fmt(m.app.impressions)} impressions · ${fmt(m.app.clicks)} clicks${m.app.feed_ctr != null ? ` · CTR ${(m.app.feed_ctr * 100).toFixed(1)}%` : ""}</dd>
        </dl>
        ${m.admin_enabled ? `<div style="margin-top:14px"><button class="btn" data-act="retrain">${UI.refresh} Retrain now</button></div>` : ""}
      </section>
      <div class="foot"><button class="btn" data-act="signout">Sign out</button></div>`;
    $("#myInterests").addEventListener("click", (e) => {
      const c = e.target.closest(".chip");
      if (c) c.setAttribute("aria-pressed", c.getAttribute("aria-pressed") !== "true");
    });
  } catch (e) { $("#pf").innerHTML = errorBox(e); }
}

/* ------------------------------------------------------------------ events */
async function openArticle(id, src) {
  if (me) {
    api("/api/clicks", { method: "POST", body: { news_id: id, source: src } }).catch(() => {});
    feedDirty = true;
    readSinceFeed++;
  }
  location.hash = `#/article/${encodeURIComponent(id)}`;
}

document.addEventListener("click", async (e) => {
  const item = e.target.closest(".item");
  if (item && !item.closest(".skeleton")) return openArticle(item.dataset.id, item.dataset.src);
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (!act || act === "search") return;
  if (act === "back") return history.length > 1 ? history.back() : (location.hash = "#/home");
  if (act === "refresh") { feedDirty = true; return viewHome("recommended"); }
  if (act === "signout") return signOut();
  if (act === "bookmark") {
    const b = e.target.closest("[data-act]");
    const on = b.getAttribute("aria-pressed") !== "true";
    try {
      await api(`/api/bookmarks/${encodeURIComponent(b.dataset.id)}`, { method: on ? "PUT" : "DELETE" });
      b.setAttribute("aria-pressed", on);
      b.querySelector("span").textContent = on ? "Saved" : "Save";
      toast(on ? "Saved to bookmarks" : "Removed from bookmarks");
    } catch (err) { toast(err.message); }
  }
  if (act === "save-interests") {
    const interests = [...document.querySelectorAll('#myInterests [aria-pressed="true"]')].map((c) => c.dataset.cat);
    try { await api("/api/me/interests", { method: "PUT", body: { interests } }); feedDirty = true; toast("Topics saved — your feed will reflect them"); }
    catch (err) { toast(err.message); }
  }
  if (act === "retrain") {
    const token = safe(() => sessionStorage.getItem("adminToken")) || prompt("Admin token");
    if (!token) return;
    const b = e.target.closest("[data-act]");
    b.disabled = true;
    b.lastChild.textContent = " Retraining…";
    try {
      const r = await api("/api/admin/retrain", { method: "POST", admin: token });
      safe(() => sessionStorage.setItem("adminToken", token));
      feedDirty = true;
      toast(r.status === "ok" ? `Retrained in ${r.seconds}s on ${fmt(r.app_clicks)} app clicks + MIND` : "A retrain is already running");
      viewProfile();
    } catch (err) {
      safe(() => sessionStorage.removeItem("adminToken"));
      toast(err.message);
      b.disabled = false;
      b.lastChild.textContent = " Retrain now";
    }
  }
});
document.addEventListener("keydown", (e) => {
  const item = e.target.closest?.(".item");
  if (item && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openArticle(item.dataset.id, item.dataset.src); }
});
document.addEventListener("submit", (e) => {
  const f = e.target.closest('[data-act="search"]');
  if (!f) return;
  e.preventDefault();
  const q = f.q.value.trim();
  if (q) location.hash = `#/search/${encodeURIComponent(q)}`;
});

/* ------------------------------------------------------------------ router */
async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (!session.uid) return viewOnboarding();
  if (!me) {
    try { me = await api("/api/me"); } catch { return; }
  }
  window.scrollTo(0, 0);
  const [page, a, b] = parts;
  if (page === "article" && a) return viewArticle(a);
  if (page === "explore") return viewExplore(a, b);
  if (page === "search" && a) return viewSearch(a);
  if (page === "bookmarks") return viewBookmarks();
  if (page === "profile") return viewProfile();
  return viewHome(page === "home" && a ? a : "recommended");
}
window.addEventListener("hashchange", route);
route();
