// In-browser SmallCNN (BatchNorm folded) + preprocessing + occlusion, mirroring
// medxai/infer.py and medxai/xai.py. Exposes a global `Engine`.
(function (root) {
  "use strict";

  function b64bytes(s) {
    if (typeof atob === "function") {
      const bin = atob(s), u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      return u;
    }
    return new Uint8Array(Buffer.from(s, "base64"));
  }
  function f32(s) { const u = b64bytes(s); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4); }

  function load(json) {
    return {
      meta: json,
      layers: json.layers.map(l => ({ cout: l.shape[0], cin: l.shape[1], w: f32(l.w), b: f32(l.b) })),
      lw: f32(json.linear.w), lb: f32(json.linear.b), nout: json.linear.shape[0], nin: json.linear.shape[1],
    };
  }

  // 3x3 conv, padding 1, then ReLU. x: cin*h*w. Works on a zero-padded copy so each
  // tap is one long contiguous loop (rows of width w+2; the 2 extra columns are dropped).
  function convRelu(x, cin, h, w, L) {
    const W2 = w + 2, P = (h + 2) * W2, pad = new Float32Array(cin * P);
    for (let c = 0; c < cin; c++) for (let y = 0; y < h; y++) pad.set(x.subarray((c * h + y) * w, (c * h + y + 1) * w), c * P + (y + 1) * W2 + 1);
    const span = h * W2 - 2, acc = new Float32Array(span), n = h * w, out = new Float32Array(L.cout * n);
    for (let o = 0; o < L.cout; o++) {
      acc.fill(L.b[o]);
      for (let c = 0; c < cin; c++) {
        const wb = (o * cin + c) * 9, pb = c * P;
        for (let t = 0; t < 9; t++) {
          const k = L.w[wb + t], off = pb + ((t / 3) | 0) * W2 + (t % 3);
          for (let q = 0; q < span; q++) acc[q] += k * pad[off + q];
        }
      }
      const ob = o * n;
      for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) { const v = acc[y * W2 + xx]; out[ob + y * w + xx] = v > 0 ? v : 0; }
    }
    return out;
  }

  function pool2(x, c, h, w) {
    const h2 = h >> 1, w2 = w >> 1, out = new Float32Array(c * h2 * w2);
    for (let k = 0; k < c; k++)
      for (let y = 0; y < h2; y++)
        for (let xx = 0; xx < w2; xx++) {
          const i = k * h * w + 2 * y * w + 2 * xx;
          out[(k * h2 + y) * w2 + xx] = Math.max(x[i], x[i + 1], x[i + w], x[i + w + 1]);
        }
    return out;
  }

  // x: normalised Float32Array(size*size) -> logits
  function forward(m, x, size) {
    let h = size, w = size, c = 1, t = x;
    m.layers.forEach((L, i) => {
      t = convRelu(t, c, h, w, L); c = L.cout;
      if (i === 1 || i === 3) { t = pool2(t, c, h, w); h >>= 1; w >>= 1; }
    });
    const n = h * w, feat = new Float64Array(c);
    for (let k = 0; k < c; k++) { let s = 0; for (let p = 0; p < n; p++) s += t[k * n + p]; feat[k] = s / n; }
    const logits = [];
    for (let o = 0; o < m.nout; o++) {
      let s = m.lb[o];
      for (let k = 0; k < m.nin; k++) s += m.lw[o * m.nin + k] * feat[k];
      logits.push(s);
    }
    return logits;
  }

  function softmax(z) { const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); }

  // Pillow's bicubic resampling (a = -0.5, support widened when downscaling), 8-bit, horizontal then vertical.
  function bicubic(v) { const a = -0.5; v = Math.abs(v); if (v < 1) return ((a + 2) * v - (a + 3)) * v * v + 1; if (v < 2) return (((v - 5) * v + 8) * v - 4) * a; return 0; }
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, fs = Math.max(scale, 1), support = 2 * fs, res = [];
    for (let o = 0; o < outSize; o++) {
      const center = (o + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0), xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
      const k = []; let tot = 0;
      for (let x = xmin; x < xmax; x++) { const wv = bicubic((x - center + 0.5) / fs); k.push(wv); tot += wv; }
      res.push({ xmin, k: k.map(v => (tot ? v / tot : 0)) });
    }
    return res;
  }
  const clip8 = v => Math.min(255, Math.max(0, Math.floor(v + 0.5)));
  function resize(gray, w, h, ow, oh) {
    let src = gray, sw = w;
    if (ow !== w) {
      const cs = coeffs(w, ow), tmp = new Uint8Array(ow * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < ow; x++) {
        const { xmin, k } = cs[x]; let s = 0;
        for (let i = 0; i < k.length; i++) s += k[i] * src[y * sw + xmin + i];
        tmp[y * ow + x] = clip8(s);
      }
      src = tmp; sw = ow;
    }
    if (oh !== h) {
      const cs = coeffs(h, oh), tmp = new Uint8Array(sw * oh);
      for (let y = 0; y < oh; y++) { const { xmin, k } = cs[y];
        for (let x = 0; x < sw; x++) { let s = 0;
          for (let i = 0; i < k.length; i++) s += k[i] * src[(xmin + i) * sw + x];
          tmp[y * sw + x] = clip8(s); } }
      src = tmp;
    }
    return src;
  }

  // RGBA pixels -> {gray} or {error}, like validate_and_load (colour check, then ITU-R 601 luma).
  function toGray(rgba, w, h) {
    if (Math.min(w, h) < 16 || Math.max(w, h) > 8192) return { error: `image size ${w}x${h} is outside the allowed range 16 to 8192 px` };
    const g = new Uint8Array(w * h);
    for (let i = 0, p = 0; p < g.length; p++, i += 4) {
      const r = rgba[i], gg = rgba[i + 1], b = rgba[i + 2];
      if (Math.abs(r - gg) > 8 || Math.abs(gg - b) > 8) return { error: "This looks like a colour image. The model only accepts grayscale chest X-rays." };
      g[p] = (r * 19595 + gg * 38470 + b * 7471 + 0x8000) >> 16;
    }
    return { gray: g };
  }

  // centre crop to square, resize, /255, normalise. Returns {u8, x}.
  function preprocess(gray, w, h, size, norm) {
    const s = Math.min(w, h), top = (h - s) >> 1, left = (w - s) >> 1, sq = new Uint8Array(s * s);
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) sq[y * s + x] = gray[(top + y) * w + left + x];
    const u8 = resize(sq, s, s, size, size), x = new Float32Array(size * size);
    for (let i = 0; i < x.length; i++) x[i] = (u8[i] / 255 - norm.mean) / norm.std;
    return { u8, x };
  }

  // Occlusion sensitivity, patch 4, stride 2, fill 0 (= training mean). Async so the page stays responsive.
  async function occlusion(m, x, size, onProgress, patch = 4, stride = 2) {
    const p0 = softmax(forward(m, x, size)), t = p0[1] > p0[0] ? 1 : 0;
    const heat = new Float64Array(size * size), cnt = new Float64Array(size * size), coords = [];
    for (let i = 0; i + patch <= size; i += stride) for (let j = 0; j + patch <= size; j += stride) coords.push([i, j]);
    for (let n = 0; n < coords.length; n++) {
      const [i, j] = coords[n], xo = x.slice();
      for (let y = i; y < i + patch; y++) for (let xx = j; xx < j + patch; xx++) xo[y * size + xx] = 0;
      const d = p0[t] - softmax(forward(m, xo, size))[t];
      for (let y = i; y < i + patch; y++) for (let xx = j; xx < j + patch; xx++) { heat[y * size + xx] += d; cnt[y * size + xx] += 1; }
      if (n % 12 === 11) { onProgress && onProgress((n + 1) / coords.length); await new Promise(r => setTimeout(r, 0)); }
    }
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < heat.length; k++) { heat[k] = Math.max(0, heat[k] / Math.max(cnt[k], 1)); lo = Math.min(lo, heat[k]); hi = Math.max(hi, heat[k]); }
    for (let k = 0; k < heat.length; k++) heat[k] = hi > lo ? (heat[k] - lo) / (hi - lo) : 0;
    return { heat, target: t, p0 };
  }

  const api = { load, forward, softmax, resize, toGray, preprocess, occlusion, b64bytes };
  if (typeof module !== "undefined") module.exports = api; else root.Engine = api;
})(this);
