import * as ort from "onnxruntime-web/wasm";

// The WASM binary is emitted by Vite (resolved via import.meta.url): no CDN, no manual copy.
// Threads need cross-origin isolation (COOP/COEP); GitHub Pages does not send those headers,
// so we stay single-threaded and say so in the HUD.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export const CONTEXT = 512;
export const LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

export class Student {
  private constructor(private sess: ort.InferenceSession, public readonly file: string, public readonly sizeMb: number) {}

  static async load(file: string): Promise<Student> {
    const url = `${import.meta.env.BASE_URL}models/${file}`;
    const buf = await (await fetch(url)).arrayBuffer();
    const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    return new Student(sess, file, buf.byteLength / 1e6);
  }

  /** context: raw values, NaN = missing, length <= 512 (left-padded with NaN). Returns 13 x 64 + ms. */
  async forecast(context: number[]): Promise<{ q: number[][]; ms: number }> {
    const x = new Float32Array(CONTEXT).fill(NaN);
    const c = context.slice(-CONTEXT);
    x.set(c.map((v) => (v === null || !Number.isFinite(v) ? NaN : v)), CONTEXT - c.length);
    const t0 = performance.now();
    const out = await this.sess.run({ context: new ort.Tensor("float32", x, [1, CONTEXT]) });
    const ms = performance.now() - t0;
    const d = out.quantiles.data as Float32Array;
    const H = d.length / LEVELS.length;
    const q = LEVELS.map((_, k) => Array.from(d.subarray(k * H, (k + 1) * H)));
    return { q, ms };
  }

  get threads(): number { return ort.env.wasm.numThreads ?? 1; }
}
