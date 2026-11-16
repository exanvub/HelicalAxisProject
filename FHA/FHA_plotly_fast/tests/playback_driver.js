/* Loaded into the dashboard by tests/test_playback_browser.py. It works the playback
   controls the way a person would, reads back what the page then shows, and posts that
   to the test (window.__report is the address). Nothing here is part of the dashboard. */
(function () {
    var out = {errors: []};
    window.addEventListener("error", function (e) { out.errors.push(String(e.message)); });

    function gd(id) { var h = document.getElementById(id); return h && h.querySelector(".js-plotly-plot"); }
    function shapeOf(id) { var g = gd(id); return g ? (g.layout.shapes || []).filter(function (x) { return x.name === "playhead"; })[0] : null; }
    function line(id) { var s = shapeOf(id); return s ? s.x0 : null; }
    function ring(id) { var s = shapeOf(id); return s ? [s.xanchor, s.yanchor, s.visible] : null; }
    function trace(id, meta) {
        var g = gd(id);
        if (!g || !g._fullData) { return null; }
        for (var i = 0; i < g._fullData.length; i++) { if (g._fullData[i].meta === meta) { return g._fullData[i]; } }
        return null;
    }
    function xyz(t) { return t ? [Array.prototype.slice.call(t.x), Array.prototype.slice.call(t.y), Array.prototype.slice.call(t.z)] : null; }
    /* What the GL scene itself holds for a body: its first vertices, back in data coordinates. */
    function drawn(meta) {
        var g = gd("fig-scene"), t = trace("fig-scene", meta), scene = g._fullLayout.scene._scene;
        var mesh = scene.traces[t.uid].mesh, scale = scene.dataScale;
        return mesh && mesh.positions ? mesh.positions.slice(0, 3).map(function (p) { return [p[0] / scale[0], p[1] / scale[1], p[2] / scale[2]]; }) : null;
    }
    /* Where the playhead shape of a figure sits on the screen, relative to that figure. */
    function onscreen(id) {
        var g = gd(id), j = -1;
        if (!g) { return null; }
        (g.layout.shapes || []).forEach(function (x, i) { if (x.name === "playhead") { j = i; } });
        var el = g.querySelector('.shapelayer path[data-index="' + j + '"]');
        if (!el || el.style.display === "none") { return null; }
        var r = el.getBoundingClientRect(), o = g.getBoundingClientRect();
        return [r.left + r.width / 2 - o.left, r.top + r.height / 2 - o.top];
    }
    var SHAPED = ["fig-direction", "fig-position", "fig-buildup", "fig-swing", "fig-series", "fig-overview"];
    function screens() { var o = {}; SHAPED.forEach(function (id) { o[id] = onscreen(id); }); return o; }
    function slider() { var el = document.getElementById("playhead"); return el ? Number(el.value) : null; }
    function playing() { var el = document.getElementById("transport"); return el ? el.getAttribute("data-playing") : null; }
    function label() {
        var button = document.getElementById("play"), shown = "";
        if (!button) { return null; }
        button.querySelectorAll("span").forEach(function (s) { if (getComputedStyle(s).display !== "none") { shown += s.textContent; } });
        return shown;
    }
    function text(id) { var el = document.getElementById(id); return el ? el.textContent : null; }
    function inView(id) { var g = gd(id); if (!g) { return false; } var r = g.getBoundingClientRect(); return r.bottom > 0 && r.top < window.innerHeight; }
    function bringIntoView(id) { var g = gd(id); window.scrollTo(0, window.scrollY + g.getBoundingClientRect().top - 60); }
    function later(ms) { return new Promise(function (ok) { setTimeout(ok, ms); }); }
    function until(what, cond, tries) {
        return new Promise(function (ok, no) {
            (function poll(n) {
                var v; try { v = cond(); } catch (e) { v = false; }
                if (v) { ok(v); } else if (n <= 0) { no(new Error("timed out waiting for " + what)); } else { setTimeout(function () { poll(n - 1); }, 100); }
            })(tries || 300);
        });
    }
    function send() { return fetch(window.__report, {method: "POST", mode: "no-cors", body: JSON.stringify(out)}); }
    function copy(o) { return JSON.parse(JSON.stringify(o)); }
    function now(extra) {
        var k = st.step, o = {slider: slider(), step: k, label: label(), playing: playing(), info: text("playhead-info"), t: st.data.t[k]};
        for (var key in (extra || {})) { o[key] = extra[key]; }
        return o;
    }
    var api, st;

    until("the page to be drawn", function () {
        return gd("fig-scene") && trace("fig-scene", "playback:body2") && line("fig-series") !== null &&
               document.getElementById("playhead") && window.dash_clientside.playback._state().data;
    })
    .then(function () { return later(1500); })
    .then(function () {
        api = window.dash_clientside.playback; st = api._state();
        var data = st.data;
        out.plotly = window.Plotly.version; out.steps = data.n; out.rate = data.rate;
        out.initial = now({max: Number(document.getElementById("playhead").max)});

        // 1. Every figure put on a few steps, directly and through Plotly's own calls.
        out.placed = {};
        [0, Math.floor(data.n / 3), Math.floor(data.n / 3) + 3, data.n - 1].forEach(function (k) {
            st.direct = true; api._apply(k);
            var how = copy(st.paths), direct = drawn("playback:body2"), screen_direct = screens();
            st.direct = false; api._apply(k);
            var plotly = drawn("playback:body2"), screen_plotly = screens(), forced = copy(st.paths);
            st.direct = true;
            out.placed[k] = {body1: xyz(trace("fig-scene", "playback:body1")), body2: xyz(trace("fig-scene", "playback:body2")),
                             axis: xyz(trace("fig-scene", "playback:axis")), series: line("fig-series"), overview: line("fig-overview"),
                             direction: ring("fig-direction"), position: ring("fig-position"), buildup: ring("fig-buildup"), swing: ring("fig-swing"),
                             how: how, how_forced: forced, drawn_direct: direct, drawn_plotly: plotly,
                             screen_direct: screen_direct, screen_plotly: screen_plotly};
        });
        var t0 = performance.now(), n = 20, i;
        for (i = 0; i < n; i++) { api._apply((i * 7) % data.n); }
        out.ms_direct = (performance.now() - t0) / n;
        st.direct = false; t0 = performance.now();
        for (i = 0; i < n; i++) { api._apply((i * 7) % data.n); }
        out.ms_plotly = (performance.now() - t0) / n;
        st.direct = true;
        api._show(0);

        // 2. Play, through the button, with the 3D view on screen.
        bringIntoView("fig-scene");
        return later(400);
    })
    .then(function () { document.getElementById("play").click(); return later(2000); })
    .then(function () {
        var k = st.step;
        out.playing = now({scene_in_view: inView("fig-scene"), axis: xyz(trace("fig-scene", "playback:axis")), want_axis: st.data.axis[k],
                           body2_x0: trace("fig-scene", "playback:body2").x[0]});
        document.getElementById("play").click();          // pause
        return later(400);
    })
    .then(function () { out.paused = now(); return later(500); })
    .then(function () {
        out.paused_later = now({overview_in_view: inView("fig-overview"), overview: line("fig-overview")});
        // 3. A figure that was out of sight catches up when it is scrolled into view.
        bringIntoView("fig-overview");
        return later(600);
    })
    .then(function () {
        out.scrolled = {overview: line("fig-overview"), overview_in_view: inView("fig-overview")};
        bringIntoView("fig-scene");
        return later(300);
    })
    .then(function () {
        // 4. Dragging the slider (the events a drag gives).
        var el = document.getElementById("playhead");
        el.value = 123; el.dispatchEvent(new Event("input", {bubbles: true})); el.dispatchEvent(new Event("change", {bubbles: true}));
        return later(400);
    })
    .then(function () {
        out.dragged = now({axis: xyz(trace("fig-scene", "playback:axis")), want_axis: st.data.axis[123]});
        // 5. Four times the speed, to the end: it stops there.
        var speeds = document.querySelectorAll("#speed input");
        speeds[speeds.length - 1].click();
        return later(600);
    })
    .then(function () { out.speed = st.speed; document.getElementById("play").click(); return until("play to start", function () { return playing() === "true"; }, 50); })
    .then(function () { return until("the end of the movement", function () { return playing() === "false"; }, 900); })
    .then(function () { out.ended = now(); return later(600); })
    .then(function () {
        // 6. A change of setting while parked at the end: the server redraws every figure.
        out.stamp_before = st.data.stamp;
        var themes = document.querySelectorAll("#theme input");
        themes[themes.length - 1].click();
        return until("the server's new figures", function () { return st.data.stamp !== out.stamp_before; }, 200);
    })
    .then(function () { return later(2500); })
    .then(function () {
        out.redrawn = now({theme: document.getElementById("root").className, series: line("fig-series"), server_step: st.data.step,
                           axis: xyz(trace("fig-scene", "playback:axis")), want_axis: st.data.axis[st.step]});
        return send();
    })
    .catch(function (e) { out.errors.push(String(e)); return send(); });
})();
