/* Playback for the helical axis explorer.

   The server draws every figure with the playhead at one position and sends, once per
   analysis, what is needed to move it (the "playback" store, built by
   FHA_visualizer_fast.playback_store). Everything else happens here, in the browser:
   the play button, the playhead slider, the clock, and moving the two bodies in the 3D
   view, the axis of the instant, the ring on the maps and on the build-up, and the
   vertical line on the plots that have time along x. Nothing goes through Dash or the
   server while the motion plays; Dash is told where the playhead stopped, so that the
   next figures it draws have it in the same place.

   The figures say which of their parts follow the playhead. In the 3D view the traces
   with meta = "playback:body1" | "playback:body2" | "playback:axis". In every other
   figure the shape named "playhead": a vertical line where time runs along x, a ring
   elsewhere.

   HOW THE PARTS ARE MOVED. Plotly's own calls (restyle, relayout) would do, and they are
   what this script falls back on. But each of them recomputes and redraws its whole
   figure: about 20 ms for the 3D view and 5-17 ms for each plot, which with everything
   on screen leaves a dozen frames a second, and would leave far fewer with real bone
   surfaces. So the moving parts are moved directly:

     - 3D view: the new coordinates are handed to the scene's own object for that trace,
       and the scene redraws on its next frame;
     - plots: the shape Plotly drew is slid across the screen, and Plotly's record of
       where it is is updated, so that whatever Plotly redraws later (zoom, resize)
       lands in the same place.

   Both reach into Plotly's internals. Every such step checks that what it needs is
   there and otherwise leaves the job to restyle / relayout. Setting
   window.dash_clientside.playback._state().direct = false switches the direct way off. */
(function () {
    "use strict";

    var state = {
        data: null,     // the playback store
        step: 0,        // where the playhead is
        pos: 0,         // the same while playing, as a fraction of a step
        time: 0,        // clock at the last frame drawn
        playing: false,
        speed: 1,
        loop: false,
        direct: true,   // move parts directly (see above); false: only through Plotly's own calls
        paths: {}       // per figure, how it was last moved: "direct" or "plotly"
    };

    function plot(id) {
        var holder = document.getElementById(id);
        var gd = holder && holder.querySelector(".js-plotly-plot");
        return gd && gd.data && gd.layout ? gd : null;
    }

    function inView(gd) {
        var box = gd.getBoundingClientRect();
        return box.bottom > 0 && box.top < (window.innerHeight || document.documentElement.clientHeight);
    }

    // ------------------------------------------------------------------ the 3D view

    /* Rotation matrix, row by row, of a unit quaternion (x, y, z, w). */
    function rotation(q) {
        var x = q[0], y = q[1], z = q[2], w = q[3];
        return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
    }

    /* The vertices of a body, given in its own coordinates, where its pose at step k puts them. */
    function placed(body, k) {
        var R = rotation(body.q[k]), p = body.p[k], v = body.v, n = v.length;
        var X = new Array(n), Y = new Array(n), Z = new Array(n);
        for (var i = 0; i < n; i++) {
            var a = v[i][0], b = v[i][1], c = v[i][2];
            X[i] = R[0] * a + R[1] * b + R[2] * c + p[0];
            Y[i] = R[3] * a + R[4] * b + R[5] * c + p[1];
            Z[i] = R[6] * a + R[7] * b + R[8] * c + p[2];
        }
        return [X, Y, Z];
    }

    /* The traces of the 3D view that follow the playhead. */
    function movers(gd) {
        var which = [];
        for (var i = 0; i < gd.data.length; i++) {
            var meta = gd.data[i].meta;
            if (typeof meta === "string" && meta.indexOf("playback:") === 0) { which.push(i); }
        }
        return which;
    }

    /* Their coordinates at step k: [x, y, z], one entry per trace of `which`. */
    function sceneAt(gd, which, data, k) {
        var xs = [], ys = [], zs = [];
        which.forEach(function (i) {
            var meta = gd.data[i].meta, xyz = [[], [], []];
            if (meta === "playback:axis") {
                var a = data.axis[k], there = a && a[0] !== null;
                xyz = there ? [[a[0], a[3]], [a[1], a[4]], [a[2], a[5]]] : [[null, null], [null, null], [null, null]];
            } else {
                for (var b = 0; b < data.bodies.length; b++) {
                    if (data.bodies[b].meta === meta) { xyz = placed(data.bodies[b], k); }
                }
            }
            xs.push(xyz[0]); ys.push(xyz[1]); zs.push(xyz[2]);
        });
        return [xs, ys, zs];
    }

    function moveScene(gd, data, k, token) {
        var which = movers(gd);
        if (!which.length || which.every(function (i) { return gd.data[i]._playhead === token; })) { return null; }
        var at = sceneAt(gd, which, data, k), direct = false;
        if (state.direct) {
            try {
                var layout = gd._fullLayout, scene = layout && layout.scene && layout.scene._scene;
                var ready = scene && scene.traces && scene.glplot && which.every(function (i) {
                    var full = gd._fullData[i], drawn = full && scene.traces[full.uid];
                    return full && full.visible === true && drawn && typeof drawn.update === "function";
                });
                if (ready) {
                    which.forEach(function (i, n) {
                        var full = gd._fullData[i];
                        full.x = gd.data[i].x = at[0][n];
                        full.y = gd.data[i].y = at[1][n];
                        full.z = gd.data[i].z = at[2][n];
                        scene.traces[full.uid].update(full);
                    });
                    if (typeof scene.glplot.redraw === "function") { scene.glplot.redraw(); }
                    direct = true;
                }
            } catch (error) {
                direct = false;
            }
        }
        if (!direct) { window.Plotly.restyle(gd, {x: at[0], y: at[1], z: at[2]}, which); }
        which.forEach(function (i) { gd.data[i]._playhead = token; });
        return direct ? "direct" : "plotly";
    }

    // ------------------------------------------------------------------ the plots

    function playheadShape(gd) {
        var shapes = gd.layout.shapes || [];
        for (var j = 0; j < shapes.length; j++) {
            if (shapes[j].name === "playhead") { return j; }
        }
        return -1;
    }

    /* The axis a shape is placed on ("x" -> xaxis, "y2" -> yaxis2), with Plotly's
       mapping from value to pixel; null if the shape is not on an axis. */
    function axisOf(gd, ref) {
        if (typeof ref !== "string" || !/^[xy]\d*$/.test(ref)) { return null; }
        var axis = gd._fullLayout && gd._fullLayout[ref.charAt(0) + "axis" + ref.slice(1)];
        return axis && typeof axis.l2p === "function" && axis.range ? axis : null;
    }

    /* Inside the range of an axis, give or take half a percent (values are sent rounded,
       and a playhead on the very edge of a plot should still show). */
    function within(axis, value) {
        var a = Number(axis.range[0]), b = Number(axis.range[1]), slack = 0.005 * Math.abs(b - a);
        return value >= Math.min(a, b) - slack && value <= Math.max(a, b) + slack;
    }

    /* Slide the playhead shape Plotly drew to (x[, y]) and record the new place in
       Plotly's layout. False if it cannot be done this way. */
    function slide(gd, j, x, y) {
        var line = y === undefined;
        var shape = gd.layout.shapes[j], full = gd._fullLayout && gd._fullLayout.shapes && gd._fullLayout.shapes[j];
        var drawn = gd.querySelector('.shapelayer path[data-index="' + j + '"]');
        var xa = axisOf(gd, shape.xref), ya = line ? null : axisOf(gd, shape.yref);
        if (!drawn || !full || !xa || (!line && !ya)) { return false; }
        // where Plotly drew this element: noted on it the first time it is slid; a redraw
        // by Plotly makes a new element without the note, drawn where the layout says
        var note = drawn.getAttribute("data-drawn-at");
        var from = note ? note.split(",").map(Number) : (line ? [shape.x0, 0] : [shape.xanchor, shape.yanchor]);
        var dx = xa.l2p(x) - xa.l2p(from[0]);
        var dy = line ? 0 : ya.l2p(y) - ya.l2p(from[1]);
        if (!isFinite(dx) || !isFinite(dy)) { return false; }
        if (!note) { drawn.setAttribute("data-drawn-at", from.join(",")); }
        drawn.setAttribute("transform", "translate(" + dx + "," + dy + ")");
        // a slid element takes its clipping with it, so keep it out of sight when it leaves the axes
        drawn.style.display = within(xa, x) && (line || within(ya, y)) ? "" : "none";
        if (line) { shape.x0 = shape.x1 = full.x0 = full.x1 = x; }
        else { shape.xanchor = full.xanchor = x; shape.yanchor = full.yanchor = y; }
        return true;
    }

    /* The playhead of a plot: a ring on the point xy (hidden when xy holds nothing), or,
       where xy is undefined, the vertical line at time t. */
    function moveShape(gd, xy, t, token) {
        var j = playheadShape(gd);
        if (j < 0 || gd.layout.shapes[j]._playhead === token) { return null; }
        var shape = gd.layout.shapes[j], line = xy === undefined, change = {}, how = "plotly";
        var there = line || (!!xy && xy[0] !== null && xy[1] !== null);
        var visible = shape.visible !== false;
        var slid = false;
        if (there && visible && state.direct) {
            try { slid = line ? slide(gd, j, t) : slide(gd, j, xy[0], xy[1]); } catch (error) { slid = false; }
        }
        if (slid) {
            how = "direct";
        } else {
            if (line) {
                change["shapes[" + j + "].x0"] = t;
                change["shapes[" + j + "].x1"] = t;
            } else if (there) {
                change["shapes[" + j + "].xanchor"] = xy[0];
                change["shapes[" + j + "].yanchor"] = xy[1];
            }
            if (visible !== there) { change["shapes[" + j + "].visible"] = there; }
            if (Object.keys(change).length) { window.Plotly.relayout(gd, change); }
        }
        gd.layout.shapes[j]._playhead = token;
        return how;
    }

    // ------------------------------------------------------------------ all figures

    /* Put every figure on step k. Each moving part carries a note of the step it shows, so
       a figure already there costs nothing, and one the server has just redrawn (new
       objects, no note) is moved again. Figures scrolled out of sight are left alone and
       caught up when they come back into view. */
    function apply(k, everything) {
        var data = state.data;
        if (!data || !window.Plotly || !(k >= 0) || k >= data.n) { return; }
        var token = data.stamp + ":" + k + ":" + (state.direct ? "d" : "p");
        var ids = ["fig-scene"].concat(Object.keys(data.points), data.lines);
        ids.forEach(function (id) {
            var gd = plot(id), how;
            if (!gd || (!everything && !inView(gd))) { return; }
            try {
                if (id === "fig-scene") { how = moveScene(gd, data, k, token); }
                else if (data.points[id]) { how = moveShape(gd, data.points[id][k], undefined, token); }
                else { how = moveShape(gd, undefined, data.t[k], token); }
                if (how) { state.paths[id] = how; }
            } catch (error) {
                /* the figure is being redrawn by the server's answer; the next call puts it right */
            }
        });
    }

    // ------------------------------------------------------------------ the controls

    /* The slider is made here, not by Dash, so that dragging it and moving it during
       playback cost nothing but a redraw of the slider itself. */
    function slider() {
        var input = document.getElementById("playhead");
        if (input) { return input; }
        var holder = document.getElementById("playhead-holder");
        if (!holder) { return null; }
        input = document.createElement("input");
        input.type = "range";
        input.id = "playhead";
        input.min = 0; input.max = 1; input.step = 1; input.value = 0;
        input.setAttribute("aria-label", "Playhead: position in the selected frames");
        input.addEventListener("input", function () { state.pos = Number(input.value); show(Number(input.value)); });
        input.addEventListener("change", report);
        holder.appendChild(input);
        return input;
    }

    function readout(k) {
        var data = state.data, box = document.getElementById("playhead-info");
        if (!box || !data) { return; }
        box.textContent = "t = " + data.t[k].toFixed(2) + " s · frame " + data.frame[k] +
                          " · joint rotation " + data.rotation[k].toFixed(1) + "°";
    }

    /* The playhead goes to step k: slider, text, figures. */
    function show(k) {
        var data = state.data;
        if (!data) { return; }
        k = Math.max(0, Math.min(data.n - 1, Math.round(k)));
        state.step = k;
        var input = slider();
        if (input && Number(input.value) !== k) { input.value = k; }
        readout(k);
        apply(k, false);
    }

    /* Tell Dash where the playhead is (a click on a hidden button: see `sync` below). */
    function report() {
        var button = document.getElementById("playhead-sync");
        if (button) { button.click(); }
    }

    function mark(playing) {
        state.playing = playing;
        var bar = document.getElementById("transport");
        if (bar) { bar.setAttribute("data-playing", playing ? "true" : "false"); }
    }

    function frame(now) {
        var data = state.data;
        if (!state.playing || !data) { return; }
        // The playhead follows the clock, not the count of frames drawn, so a slow machine
        // shows fewer frames at the right speed. (Capped, so that a tab left in the
        // background does not leap ahead when it comes back.)
        var elapsed = Math.min((now - state.time) / 1000, 1.0);
        state.time = now;
        state.pos += elapsed * data.rate * state.speed;
        var last = data.n - 1;
        if (state.pos >= last) {
            if (state.loop) {
                state.pos = 0;
            } else {
                state.pos = last;
                show(last);
                mark(false);
                report();
                return;
            }
        }
        show(Math.floor(state.pos));
        window.requestAnimationFrame(frame);
    }

    function play() {
        var data = state.data;
        if (!data || state.playing) { return; }
        state.pos = state.step >= data.n - 1 ? 0 : state.step;      // at the end: start over
        state.time = window.performance.now();
        mark(true);
        window.requestAnimationFrame(frame);
    }

    function pause() {
        if (!state.playing) { return; }
        mark(false);
        report();
    }

    document.addEventListener("click", function (event) {
        if (event.target && event.target.closest && event.target.closest("#play")) {
            if (state.playing) { pause(); } else { play(); }
        }
    });

    var waiting = null;
    window.addEventListener("scroll", function () {
        if (waiting) { return; }
        waiting = window.setTimeout(function () { waiting = null; apply(state.step, false); }, 120);
    }, {passive: true});

    /* New playback data from the server, drawn with the playhead at data.step. */
    function load(data) {
        var fresh = !state.data || state.data.key !== data.key;     // another recording or frame range
        state.data = data;
        if (fresh) {
            mark(false);
            state.step = state.pos = 0;
        }
        state.step = Math.min(state.step, data.n - 1);
        var input = slider();
        if (input) { input.max = data.n - 1; input.value = state.step; }
        mark(state.playing);
        readout(state.step);
        // The figures arrive, and are drawn, a moment after this call; look again a few times.
        [0, 80, 250, 600, 1500].forEach(function (ms) {
            window.setTimeout(function () { if (state.data === data) { apply(state.step, false); } }, ms);
        });
    }

    // ------------------------------------------------------------------ what Dash calls

    var api = (window.dash_clientside = window.dash_clientside || {});
    api.playback = {

        /* Dash's one line to this script. It hands over new playback data and the speed and
           loop settings, and takes back where the playhead is: {key, step}, which the server
           reads when it next draws the figures. Also runs when the hidden "playhead-sync"
           button is clicked, which is how this script reports a new position. */
        sync: function (data, speed, loop) {
            state.speed = Number(speed) || 1;
            state.loop = !!(loop && loop.length);
            if (data && data.n && (!state.data || data.stamp !== state.data.stamp)) { load(data); }
            return state.data ? {key: state.data.key, step: state.step} : window.dash_clientside.no_update;
        },

        /* For tests and the console. */
        _show: show,
        _apply: function (k) { apply(k, true); return state; },
        _state: function () { return state; }
    };
})();
