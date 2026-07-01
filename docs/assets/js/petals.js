/* Flower-bed dynamic background: procedurally drawn petals drift and
 * gently scatter away from the cursor, echoing a bed of flowers stirred
 * by wind. Pure canvas 2D — no external image assets. */
(function () {
  "use strict";

  var canvas = document.getElementById("petal-canvas");
  if (!canvas || !canvas.getContext) return;

  var ctx = canvas.getContext("2d");
  var hero = canvas.parentElement;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var width = 0, height = 0;
  var petals = [];
  var pointer = { x: -9999, y: -9999, active: false };
  var rafId = null;

  var COLORS = [
    "rgba(201, 145, 90, ALPHA)",  /* gold */
    "rgba(217, 140, 147, ALPHA)", /* rose */
    "rgba(230, 205, 170, ALPHA)", /* ivory */
    "rgba(167, 120, 100, ALPHA)"  /* clay */
  ];

  function density() {
    var area = width * height;
    var base = Math.round(area / 22000);
    return Math.max(24, Math.min(base, 90));
  }

  function makePetal() {
    var size = 6 + Math.random() * 10;
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      size: size,
      baseSize: size,
      angle: Math.random() * Math.PI * 2,
      spin: (Math.random() - 0.5) * 0.02,
      drift: 0.25 + Math.random() * 0.55,
      sway: Math.random() * Math.PI * 2,
      swaySpeed: 0.004 + Math.random() * 0.006,
      swayAmp: 10 + Math.random() * 24,
      color: COLORS[Math.floor(Math.random() * COLORS.length)].replace(
        "ALPHA",
        (0.25 + Math.random() * 0.35).toFixed(2)
      ),
      vx: 0,
      vy: 0
    };
  }

  function resize() {
    var rect = hero.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * DPR;
    canvas.height = height * DPR;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

    var target = reduceMotion ? Math.round(density() / 3) : density();
    if (petals.length === 0) {
      for (var i = 0; i < target; i++) petals.push(makePetal());
    } else {
      for (var j = 0; j < petals.length; j++) {
        petals[j].x = petals[j].x % width;
        if (petals[j].x < 0) petals[j].x += width;
        petals[j].y = petals[j].y % height;
        if (petals[j].y < 0) petals[j].y += height;
      }
      if (petals.length < target) {
        while (petals.length < target) petals.push(makePetal());
      } else if (petals.length > target) {
        petals.length = target;
      }
    }
  }

  function drawPetal(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.angle);
    ctx.beginPath();
    ctx.moveTo(0, -p.size);
    ctx.bezierCurveTo(p.size * 0.8, -p.size * 0.6, p.size * 0.8, p.size * 0.6, 0, p.size);
    ctx.bezierCurveTo(-p.size * 0.8, p.size * 0.6, -p.size * 0.8, -p.size * 0.6, 0, -p.size);
    ctx.closePath();
    ctx.fillStyle = p.color;
    ctx.fill();
    ctx.restore();
  }

  function step(t) {
    ctx.clearRect(0, 0, width, height);

    for (var i = 0; i < petals.length; i++) {
      var p = petals[i];

      p.sway += p.swaySpeed;
      var swayX = Math.cos(p.sway) * 0.15;

      p.vy += (p.drift - p.vy) * 0.02;
      p.vx += (swayX - p.vx) * 0.02;

      if (pointer.active) {
        var dx = p.x - pointer.x;
        var dy = p.y - pointer.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        var radius = 130;
        if (dist < radius && dist > 0.001) {
          var force = (1 - dist / radius) * 2.4;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        }
      }

      p.x += p.vx;
      p.y += p.vy;
      p.angle += p.spin + p.vx * 0.01;

      if (p.y - p.size > height) {
        p.y = -p.size;
        p.x = Math.random() * width;
      }
      if (p.x < -p.size * 2) p.x = width + p.size;
      if (p.x > width + p.size * 2) p.x = -p.size;

      drawPetal(p);
    }

    rafId = requestAnimationFrame(step);
  }

  function onPointerMove(e) {
    var rect = canvas.getBoundingClientRect();
    pointer.x = e.clientX - rect.left;
    pointer.y = e.clientY - rect.top;
    pointer.active = true;
  }

  function onPointerLeave() {
    pointer.active = false;
    pointer.x = -9999;
    pointer.y = -9999;
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  });

  hero.addEventListener("mousemove", onPointerMove, { passive: true });
  hero.addEventListener("mouseleave", onPointerLeave, { passive: true });
  hero.addEventListener(
    "touchmove",
    function (e) {
      if (e.touches && e.touches[0]) {
        onPointerMove(e.touches[0]);
      }
    },
    { passive: true }
  );
  hero.addEventListener("touchend", onPointerLeave, { passive: true });

  resize();

  if (reduceMotion) {
    step(0);
    cancelAnimationFrame(rafId);
  } else {
    rafId = requestAnimationFrame(step);
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      cancelAnimationFrame(rafId);
    } else if (!reduceMotion) {
      rafId = requestAnimationFrame(step);
    }
  });
})();
