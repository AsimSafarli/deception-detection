// Canvas renderer for the 3D face mesh (painter's algorithm), coloured by muscle-region values.
// Standalone copy of the renderer in index.html, used by analyze.html.
(function () {
  const css = (el, name) => getComputedStyle(el).getPropertyValue(name).trim();
  const rgb = (c) => { const x = d3.color(c); return x ? [x.r, x.g, x.b] : [200, 200, 200]; };

  window.FaceView = function (container, { onHover } = {}) {
    const canvas = document.createElement('canvas');
    canvas.className = 'fx-canvas';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', '3D face model coloured by muscle activity');
    container.appendChild(canvas);
    const v = { yaw: -0.45, pitch: 0.12, verts: [], tris: [], w: [], anchors: [], vals: null, top: [], scale: 2.5, proj: null };

    let drag = null;
    canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: v.yaw, pitch: v.pitch }; canvas.setPointerCapture(e.pointerId); onHover && onHover(null); });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('pointerleave', () => onHover && onHover(null));
    canvas.addEventListener('pointermove', (e) => {
      if (drag) {
        v.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
        v.pitch = Math.max(-0.9, Math.min(0.9, drag.pitch + (e.clientY - drag.y) * 0.006));
        render();
        return;
      }
      if (!onHover || !v.proj || !v.vals) return;
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = -1, bd = 22 * 22;
      v.proj.forEach((p, i) => { if (p.front) { const d = (p.x - px) ** 2 + (p.y - py) ** 2; if (d < bd) { bd = d; best = i; } } });
      const ws = best >= 0 ? v.w[best] : [];
      if (!ws.length) { onHover(null); return; }
      const mi = ws.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
      onHover({ muscle: v.anchors[mi], value: v.vals[mi], x: px, y: py });
    });

    function setMesh(verts, tris, anchors) {
      const idx = new Map(anchors.map((a, i) => [a.muscle, i]));
      v.anchors = anchors;
      v.verts = verts.map((r) => [r.x, r.y, r.z]);
      v.w = verts.map((r) => Object.entries(r.w || {}).map(([m, w]) => [idx.get(m), w]).filter((p) => p[0] != null));
      v.tris = tris.map((t) => [t.a, t.b, t.c]);
    }

    function setValues(vals, scale) {
      v.vals = vals;
      if (scale) v.scale = scale;
      v.top = vals.map((z, i) => [i, z]).filter((p) => p[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 3).map((p) => p[0]);
      render();
      return v.top;
    }

    function render() {
      if (!v.verts.length) return;
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(240, container.clientWidth || 500);
      const H = Math.round(Math.min(520, Math.max(300, W * 0.9)));
      canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const cy = Math.cos(v.yaw), sy = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
      const s = Math.min(W, H) * 0.74, cx = W / 2, cyy = H / 2;
      const rot = (p) => { const x1 = p[0] * cy + p[2] * sy, z1 = -p[0] * sy + p[2] * cy; return [x1, p[1] * cp - z1 * sp, p[1] * sp + z1 * cp]; };
      const R = v.verts.map(rot);
      v.proj = R.map((r) => ({ x: cx + r[0] * s, y: cyy - r[1] * s, z: r[2], front: true }));

      const vv = v.w.map((ws) => {
        if (!v.vals || !ws.length) return 0;
        let num = 0, den = 0.35;
        ws.forEach(([m, w]) => { num += w * (v.vals[m] || 0); den += w; });
        return num / den;
      });
      const base = rgb(css(container, '--fx-face')), hot = rgb(css(container, '--fx-hot')), cold = rgb(css(container, '--fx-cold'));
      const order = v.tris.map((t, i) => [i, (R[t[0]][2] + R[t[1]][2] + R[t[2]][2]) / 3]).sort((a, b) => a[1] - b[1]);
      ctx.lineJoin = 'round';
      for (const [i] of order) {
        const [a, b, c] = v.tris[i];
        const A = R[a], B = R[b], C = R[c];
        const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2], wx = C[0] - A[0], wy = C[1] - A[1], wz = C[2] - A[2];
        const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        const len = Math.hypot(nx, ny, nz) || 1;
        const light = 0.5 + 0.5 * Math.abs((nz * 0.85 + ny * 0.35 - nx * 0.2) / len);
        const val = (vv[a] + vv[b] + vv[c]) / 3;
        const t = Math.min(1, Math.abs(val) / v.scale) * 0.92;
        const tgt = val >= 0 ? hot : cold;
        const col = base.map((bc, k) => Math.round((bc + (tgt[k] - bc) * t) * light));
        const fill = `rgb(${col[0]},${col[1]},${col[2]})`;
        const P = v.proj;
        ctx.beginPath();
        ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.lineTo(P[c].x, P[c].y); ctx.closePath();
        ctx.fillStyle = fill; ctx.strokeStyle = fill; ctx.lineWidth = 0.7;
        ctx.fill(); ctx.stroke();
      }
      const zMax = Math.max(...v.proj.map((p) => p.z));
      v.proj.forEach((p) => { p.front = p.z > zMax - 0.45; });

      ctx.font = '700 11px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      v.top.forEach((mi, k) => {
        const p = rot([v.anchors[mi].x, v.anchors[mi].y, v.anchors[mi].z]);
        const x = cx + p[0] * s, y = cyy - p[1] * s;
        ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fillStyle = css(container, '--fx-hot'); ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.fillText(String(k + 1), x, y + 0.5);
      });
    }

    return { setMesh, setValues, render };
  };
})();
