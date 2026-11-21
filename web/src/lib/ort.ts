import * as ort from "onnxruntime-web/wasm";

// The WASM binary is emitted by Vite (resolved via import.meta.url): no CDN, no manual copy.
// Threads need cross-origin isolation (COOP/COEP headers). GitHub Pages does not send them, so we run one thread.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export const CONTEXT = 512;
export const LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

export class Student {
  private constructor(private sess: ort.InferenceSession, public readonly sizeMb: number) {}

  static async load(file: string): Promise<Student> {
    const buf = await (await fetch(`${import.meta.env.BASE_URL}models/${file}`)).arrayBuffer();
    const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    const s = new Student(sess, buf.byteLength / 1e6);
    await s.forecast([1, 2, 3]); // warm-up: the first run compiles kernels and is much slower; never time it
    return s;
  }

  /** context: raw values (NaN = missing), any length; the model reads the last 512. Returns 13 x 64 quantiles + ms. */
  async forecast(context: number[]): Promise<{ q: number[][]; ms: number }> {
    const x = new Float32Array(CONTEXT).fill(NaN); // left-pad with NaN: the scaler inside the graph ignores it
    const c = context.slice(-CONTEXT);
    x.set(c.map((v) => (Number.isFinite(v) ? v : NaN)), CONTEXT - c.length);
    const t0 = performance.now();
    const out = await this.sess.run({ context: new ort.Tensor("float32", x, [1, CONTEXT]) });
    const ms = performance.now() - t0; // session.run only: no drawing, no data copies
    const d = out.quantiles.data as Float32Array; // (1, 13, 64), row-major
    const H = d.length / LEVELS.length;
    return { q: LEVELS.map((_, k) => Array.from(d.subarray(k * H, (k + 1) * H))), ms };
  }

  get threads(): number {
    return ort.env.wasm.numThreads ?? 1;
  }
}
