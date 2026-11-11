/**
 * 3D Financial Horizon & Holographic Volatility Wave Scene
 * Pure WebGL / Canvas implementation with zero external dependencies.
 * Designed for 60fps performance, battery efficiency, and mobile responsiveness.
 */

(function () {
  'use strict';

  function initStock3DScene() {
    var container = document.getElementById('stock-3d-canvas-container');
    if (!container) return;

    var canvas = document.createElement('canvas');
    canvas.className = 'stock-3d-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    container.appendChild(canvas);

    var ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    var animId = null;
    var isVisible = true;
    var rotX = 0.35;
    var rotY = 0.2;
    var targetRotX = 0.35;
    var targetRotY = 0.2;
    var time = 0;

    // Generate 3D Market Grid & Candlesticks
    var cols = 14;
    var rows = 8;
    var gridPoints = [];

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var baseHeight = Math.sin(c * 0.45) * 22 + Math.cos(r * 0.6) * 16;
        gridPoints.push({
          col: c,
          row: r,
          x: (c - (cols - 1) / 2) * 28,
          z: (r - (rows - 1) / 2) * 26,
          baseHeight: baseHeight,
          phase: Math.random() * Math.PI * 2,
          isCandle: (c + r) % 3 === 0,
          candleHeight: 15 + Math.random() * 25,
          isBull: (c + r) % 2 === 0
        });
      }
    }

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var width = container.clientWidth || 480;
      var height = container.clientHeight || 280;

      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
    }

    window.addEventListener('resize', resize);
    resize();

    // Mouse / Touch perspective tilt
    function onPointerMove(e) {
      var clientX = e.touches ? e.touches[0].clientX : e.clientX;
      var clientY = e.touches ? e.touches[0].clientY : e.clientY;
      var rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      var normX = ((clientX - rect.left) / rect.width) * 2 - 1;
      var normY = ((clientY - rect.top) / rect.height) * 2 - 1;

      targetRotY = normX * 0.45;
      targetRotX = 0.35 - normY * 0.25;
    }

    window.addEventListener('mousemove', onPointerMove, { passive: true });
    window.addEventListener('touchmove', onPointerMove, { passive: true });

    // IntersectionObserver for CPU/GPU power efficiency
    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        isVisible = entries[0].isIntersecting;
        if (isVisible && !animId) {
          loop();
        }
      }, { threshold: 0.1 });
      observer.observe(canvas);
    }

    function loop() {
      if (!isVisible) {
        animId = null;
        return;
      }

      time += 0.022;
      rotX += (targetRotX - rotX) * 0.08;
      rotY += (targetRotY - rotY) * 0.08;

      render();
      animId = requestAnimationFrame(loop);
    }

    function render() {
      var width = parseFloat(canvas.style.width) || 480;
      var height = parseFloat(canvas.style.height) || 280;
      var cx = width / 2;
      var cy = height / 2 + 10;

      ctx.clearRect(0, 0, width, height);

      var cosY = Math.cos(rotY);
      var sinY = Math.sin(rotY);
      var cosX = Math.cos(rotX);
      var sinX = Math.sin(rotX);
      var fov = 380;

      // Calculate projected 3D points
      var projected = [];
      for (var i = 0; i < gridPoints.length; i++) {
        var pt = gridPoints[i];
        var dynamicWave = Math.sin(time * 1.5 + pt.phase) * 12 + Math.cos(time * 0.8 + pt.col * 0.4) * 8;
        var y = -(pt.baseHeight + dynamicWave);

        // Y-axis rotation
        var x1 = pt.x * cosY + pt.z * sinY;
        var y1 = y;
        var z1 = -pt.x * sinY + pt.z * cosY;

        // X-axis rotation
        var y2 = y1 * cosX - z1 * sinX;
        var z2 = y1 * sinX + z1 * cosX;

        var scale = fov / (fov + z2 + 160);
        projected.push({
          col: pt.col,
          row: pt.row,
          px: cx + x1 * scale,
          py: cy + y2 * scale,
          depth: z2,
          scale: scale,
          isCandle: pt.isCandle,
          candleHeight: pt.candleHeight * scale,
          isBull: pt.isBull
        });
      }

      // Draw 3D Perspective Grid Wires
      ctx.save();
      ctx.strokeStyle = 'rgba(112, 240, 192, 0.15)';
      ctx.lineWidth = 1;

      // Column lines
      for (var c = 0; c < cols; c++) {
        ctx.beginPath();
        for (var r = 0; r < rows; r++) {
          var p = projected[r * cols + c];
          if (r === 0) ctx.moveTo(p.px, p.py);
          else ctx.lineTo(p.px, p.py);
        }
        ctx.stroke();
      }

      // Row lines
      for (var r = 0; r < rows; r++) {
        ctx.beginPath();
        for (var c = 0; c < cols; c++) {
          var p = projected[r * cols + c];
          if (c === 0) ctx.moveTo(p.px, p.py);
          else ctx.lineTo(p.px, p.py);
        }
        ctx.stroke();
      }
      ctx.restore();

      // Sort candles by depth and render
      var candles = projected.filter(function (p) { return p.isCandle; });
      candles.sort(function (a, b) { return b.depth - a.depth; });

      for (var k = 0; k < candles.length; k++) {
        var cd = candles[k];
        var color = cd.isBull ? '#70f0c0' : '#ff7a85';
        var alpha = Math.max(0.2, Math.min(0.9, cd.scale * 1.1));

        ctx.save();
        ctx.fillStyle = cd.isBull ? 'rgba(112, 240, 192, ' + alpha + ')' : 'rgba(255, 122, 133, ' + alpha + ')';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;

        // Draw Candle Wick
        ctx.beginPath();
        ctx.moveTo(cd.px, cd.py - cd.candleHeight * 1.4);
        ctx.lineTo(cd.px, cd.py + cd.candleHeight * 0.4);
        ctx.stroke();

        // Draw Candle Body (3D Box effect)
        var bw = Math.max(3, 7 * cd.scale);
        var bh = Math.max(4, cd.candleHeight);
        ctx.fillRect(cd.px - bw / 2, cd.py - bh, bw, bh);
        ctx.strokeRect(cd.px - bw / 2, cd.py - bh, bw, bh);

        // Subtle glow
        if (cd.scale > 0.8) {
          ctx.shadowColor = color;
          ctx.shadowBlur = 10 * cd.scale;
          ctx.strokeRect(cd.px - bw / 2, cd.py - bh, bw, bh);
        }
        ctx.restore();
      }
    }

    loop();
  }

  // 3D Card Tilt Interaction
  function init3DCards() {
    var cards = document.querySelectorAll('.card, .metric, .visual, .step, .runbox');
    var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReduced) return;

    cards.forEach(function (card) {
      card.addEventListener('mouseenter', function () {
        card.style.transition = 'transform 0.12s cubic-bezier(0.2, 0, 0, 1)';
      });

      card.addEventListener('mousemove', function (e) {
        var rect = card.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        var x = (e.clientX - rect.left) / rect.width;
        var y = (e.clientY - rect.top) / rect.height;

        var rotX = (0.5 - y) * 10;
        var rotY = (x - 0.5) * 10;

        card.style.transform = 'perspective(900px) rotateX(' + rotX.toFixed(2) + 'deg) rotateY(' + rotY.toFixed(2) + 'deg) scale3d(1.02, 1.02, 1.02)';
        card.style.setProperty('--card-x', (x * 100).toFixed(1) + '%');
        card.style.setProperty('--card-y', (y * 100).toFixed(1) + '%');
      });

      card.addEventListener('mouseleave', function () {
        card.style.transition = 'transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1)';
        card.style.transform = 'perspective(900px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
      });
    });
  }

  // View toggle between 3D Horizon & 2D Chart
  function initViewToggle() {
    var btn3D = document.getElementById('view-toggle-3d');
    var btn2D = document.getElementById('view-toggle-2d');
    var c3D = document.getElementById('stock-3d-canvas-container');
    var c2D = document.getElementById('stock-2d-chart-container');

    if (!btn3D || !btn2D || !c3D || !c2D) return;

    btn3D.addEventListener('click', function () {
      btn3D.classList.add('active');
      btn2D.classList.remove('active');
      c3D.style.display = 'block';
      c2D.style.display = 'none';
    });

    btn2D.addEventListener('click', function () {
      btn2D.classList.add('active');
      btn3D.classList.remove('active');
      c2D.style.display = 'block';
      c3D.style.display = 'none';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      initStock3DScene();
      init3DCards();
      initViewToggle();
    });
  } else {
    initStock3DScene();
    init3DCards();
    initViewToggle();
  }
})();
