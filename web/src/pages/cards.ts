// Study cards: question on the front, answer + what we measured + where to go deeper on the back.
// Every number on a card comes from a results file or journal entry named on the card.
import { header } from "../lib/nav";

header();

type Topic = "Evaluation" | "Leakage & data" | "Distillation" | "Architecture" | "Long horizons & context" | "Shipping";
interface Card { topic: Topic; q: string; a: string; here: string; src: string; more?: [string, string] }

const CARDS: Card[] = [
  // ---------------------------------------------------------------- evaluation
  { topic: "Evaluation", q: "Why is the pinball loss the right loss for a quantile forecast?",
    a: "ρ<sub>τ</sub>(y − q) charges τ·(y − q) when you under-predict and (1 − τ)·(q − y) when you over-predict. Its expected value is minimised exactly at the τ-quantile of the true distribution, so it is a proper scoring rule: a model cannot lower it by lying.",
    here: "Every student is trained and scored with it: 13 levels, weighted by the probability mass each level stands for.",
    src: "tinycast/eval/metrics.py · pinball_loss", more: ["post-evaluation.html", "Post 1, section 1"] },
  { topic: "Evaluation", q: "CRPS is the standard score for probabilistic forecasts. How do you compute it from 13 quantiles, and what can go wrong?",
    a: "CRPS equals twice the integral of the pinball loss over all levels τ. With a finite set of levels you get a Riemann sum, and its bias depends on the grid. Comparing models on the same grid is fine; comparing numbers computed on different grids is not.",
    here: "For a calibrated Gaussian forecast, the 9 levels 0.1–0.9 overestimate CRPS by about 9%, our 13 levels underestimate it by about 17%. We report on the 9 levels every model shares.",
    src: "docs/decisions/0002-metrics.md · tests/test_metrics.py" },
  { topic: "Evaluation", q: "Why divide every score by seasonal naive, and why average with a geometric mean?",
    a: "Raw errors are not comparable across series: a flat series is easy, a spiky one hard. The ratio to seasonal naive on the same window is unit-free. The geometric mean treats ×2 and ÷2 symmetrically, so one exploding series cannot dominate. This is also the GIFT-Eval leaderboard rule.",
    here: "A score of 0.655 (the published student on fresh electricity) reads: 34.5% less error than repeating the last day.",
    src: "results/fresh_fr_power/summary.json" },
  { topic: "Evaluation", q: "Two models' 95% intervals overlap. Are they tied?",
    a: "Not necessarily. Most of each interval is the difficulty of the test windows, which both models share. Compare on the same windows: the geometric mean of per-window ratios, with its own interval. Shared difficulty cancels out.",
    here: "Student 0.655 [0.608, 0.702] vs Toto-2.0-4m 0.709 [0.653, 0.764]: the intervals overlap, but the paired ratio is 0.924 [0.894, 0.951], a clear win.",
    src: "results/study/stage_c3/table.json", more: ["post-evaluation.html", "Post 1, section 3"] },
  { topic: "Evaluation", q: "Your test set has 312 windows. What is your sample size for a confidence interval?",
    a: "Closer to the number of series than the number of windows. Consecutive windows of one series overlap and share the same weather, so they are not independent. Resample whole series (a cluster bootstrap), or the intervals come out far too narrow.",
    here: "Fresh electricity: 312 windows from 13 series. Every interval on this site resamples series.",
    src: "tinycast/eval/metrics.py · bootstrap_ci, paired_ratio_ci" },
  { topic: "Evaluation", q: "What does interval coverage tell you that WQL does not?",
    a: "Calibration. The 80% interval should contain the outcome 80% of the time. A model can have a good WQL with intervals that are too narrow; coverage exposes it. Check coverage per level, not only the average score.",
    here: "On fresh electricity the published student's 80% interval covers 62% of outcomes, Toto-2.0-4m's 70%: the student is over-confident, despite its lower error.",
    src: "results/fresh_fr_power/summary.json (mean_cov80)" },

  // ---------------------------------------------------------------- leakage & data
  { topic: "Leakage & data", q: "How do you assign a forecasting window to train, validation or test without leaking?",
    a: "By its target. A window belongs to a split only if its whole forecast target lies inside that split. The context may reach back into earlier periods: that is just the past. Enforce the rule with a unit test, not with care.",
    here: "Public series: split by position (70 / 15 / 15%). Electricity: by calendar, with the test starting 2026-10-15, after every model's release.",
    src: "tinycast/data/windows.py · tests/test_leakage.py", more: ["post-data.html", "Post 2, section 3"] },
  { topic: "Leakage & data", q: "Foundation models were pretrained on most public datasets. How do you get an honest test?",
    a: "Two defences. (1) Data published after the models were released. (2) For training data: hash every chunk of every candidate series (z-normalised) and drop any series that shares a chunk with a benchmark.",
    here: "302 of 18,416 candidate training series were dropped by the hash check. The headline test is electricity published after 2026-10-14.",
    src: "tinycast/data/sources/data_v2.py · data/processed/data_v2_index.json" },
  { topic: "Leakage & data", q: "Your synthetic test suite says the student beats a bigger model. Why might that be optimistic?",
    a: "If the training generator produces the same families as the test suite, the test is partly in-distribution. Synthetic suites are good for diagnosing behaviour, bad for claiming generalisation. Use a broad public benchmark for that.",
    here: "Our training synthetics include randomised versions of the 20 test shapes. On shapes, a student trained on the real future beats the distilled one (it learns the generator); on GIFT-Eval the distilled one wins by 13%.",
    src: "docs/notes/gap-analysis.md, section 3" },
  { topic: "Leakage & data", q: "Validation ranks model A first, the test ranks model B first. Which do you publish?",
    a: "Whatever validation says, if you decided the protocol in advance. Picking on the test makes the test a validation set. If you do pick after seeing the test, say so: the winner's margin is then a selected, optimistic number.",
    here: "Validation ranked the 3-teacher ensemble first (0.1342 vs 0.1369). We later published the student with the best GIFT-Eval score, which is a test set, and say so on the Results page.",
    src: "docs/journal.md, 2026-11-16 (Stage C3)" },

  // ---------------------------------------------------------------- distillation
  { topic: "Distillation", q: "Classic distillation matches softmax outputs with a KL divergence. How do you distil a model that outputs quantiles?",
    a: "Treat the teacher's quantiles as observations weighted by the probability mass each stands for, and train every student level with the pinball loss against them. The loss is minimised where the student's τ-quantile equals the teacher's. No softmax, no temperature.",
    here: "That is the loss of every distilled student here (kd_pinball_teacher, α = 1 for the published one).",
    src: "tinycast/distill/losses.py · kd_pinball_teacher", more: ["post-distillation.html", "Post 4, section 2"] },
  { topic: "Distillation", q: "Your teacher only outputs the levels 0.1 to 0.9. What do you do with the 0.01 and 0.99 levels during distillation?",
    a: "Mask them out of the distillation targets. If you extrapolate the teacher's tails and train on them, the student learns your extrapolation rule, not the teacher. Let the student's own tails come from the ground-truth term or from teachers that output them.",
    here: "teacher_level_mask keeps only Toto's 9 native levels as pseudo-observations.",
    src: "tinycast/distill/losses.py" },
  { topic: "Distillation", q: "Why compute the distillation loss after the student's own scaler rather than in raw units?",
    a: "Quantiles commute with increasing transforms, so the minimiser does not move. But in raw units a 50 GW series produces gradients 10,000 times larger than a 5 MW one, and dominates training. In the scaled space every window weighs the same.",
    here: "scaled_targets applies the student's (loc, scale, asinh) to the teacher's quantiles and the real future before the loss.",
    src: "tinycast/distill/train.py · scaled_targets" },
  { topic: "Distillation", q: "Should the student learn from the teacher's forecast or from the real future?",
    a: "On real, noisy data: from the teacher. The observed future is one noisy draw; the teacher's distribution is a smoother, richer target. On synthetic data where the generator knows the exact continuation, the truth is better.",
    here: "Same multi-token network and windows, GIFT-Eval: teacher labels beat the real future by 8% at 300k windows (0.919 [0.894, 0.945]), 9% at 1M and 13% at 3M (0.868 [0.819, 0.916]). On our synthetic shapes, whose generator gives the exact future, the real future wins (teacher/real 1.23 at 3M).",
    src: "results/study/stage_c13_gt_vs_kd · stage_c15_3m · journal 2026-11-22" },
  { topic: "Distillation", q: "You have a fixed compute budget for distillation. Where do you spend it?",
    a: "On more distinct training windows labelled by the teacher, before a bigger student, a better teacher or a fancier loss. Expect diminishing, overlapping gains from everything else.",
    here: "GIFT-Eval with teacher labels: 0.624 at 300k windows, 0.603 at 1M, 0.582 at 3M. Trained on the real future instead, the same student stops improving after 1M (0.679, 0.664, 0.671). Size, context, decoder and teacher choice each gave 1–2%, and those gains do not add up.",
    src: "results/study/stage_c15_3m · journal 2026-11-21 and 2026-11-22" },
  { topic: "Distillation", q: "Is a bigger teacher always a better teacher?",
    a: "No. A small student cannot absorb everything a huge teacher knows (the capacity gap), and a smaller teacher's function can be easier to copy. A smaller teacher is also much cheaper to run on millions of windows, which buys more data, the lever that matters most.",
    here: "Same 0.87M student, 1M windows: Timestral-5M labels vs Timestral-28M on GIFT-Eval 0.984 [0.965, 1.003]. Timestral-209M vs 28M gave about 0.5% for the 0.95M student.",
    src: "results/study/stage_c14_teacher5m · journal 2026-11-20 and 2026-11-22" },
  { topic: "Distillation", q: "Should you mix the two targets: the real future where it is exact, the teacher where it is noisy?",
    a: "It sounds like the best of both, but check it on the benchmark you care about. Each target wins on the data it suits, and the mix can pull the model toward the wrong one for your use.",
    here: "Real future on synthetic windows, Timestral-5M labels on real ones: 22% better on shapes than pure teacher labels (0.779 [0.724, 0.832]), 2% worse on public data, and 0.607 vs 0.593 on GIFT-Eval.",
    src: "results/study/stage_c16_hybrid · results/gift_eval/tinycast-hyb5m_1m" },
  { topic: "Distillation", q: "Can you distil with no real data at all, on random inputs?",
    a: "Only with structured inputs. On white noise the teacher answers with the same flat forecast every time, so there is nothing to copy. Gaussian-process series with periodic and trend kernels (KernelSynth) make the teacher show its skill.",
    here: "A student distilled on noise-labelled windows had 48% higher error on fresh electricity, and 5.7× the error on shapes, than one distilled on KernelSynth.",
    src: "results/study/stage_a/table.json" },
  { topic: "Distillation", q: "You have one year of your own data. Distilled student, or a student trained only on your data?",
    a: "With plenty of in-domain history, training on it directly is a strong baseline: a specialist can match a distilled generalist on its own domain. Distillation pays when your history is short or you need several domains.",
    here: "A student trained only on electricity history ties the distilled ones on fresh electricity (1M-window student 0.998, 3M-window student 0.993 [0.959, 1.025]) but is 20% worse on shapes. With a week of history, fine-tuning from distilled weights wins clearly.",
    src: "results/site_pairs.json · results/study/stage_d_final" },

  // ---------------------------------------------------------------- architecture
  { topic: "Architecture", q: "What in a patch transformer fixes how far back it can read and how far ahead it predicts?",
    a: "Input: the number of positional embeddings times the patch length. Output: the size of the output layer (horizon × number of quantile levels). Changing either means a new tensor shape, so retraining.",
    here: "33 position rows (32 patches + 1 query token) × 16 steps = 512 steps. Output layer 128 → 832 = 13 levels × 64 steps.",
    src: "tinycast/models/student", more: ["post-student.html", "Post 3, section 2"] },
  { topic: "Architecture", q: "Why normalise each window inside the model, and why with median, mean absolute deviation and asinh?",
    a: "Instance normalisation makes the network amplitude-free: one model for megawatts and passenger counts. Median and absolute deviation resist spikes; asinh compresses large excursions while staying linear near 0. Inside the model, it ships with the weights.",
    here: "Multiply the input by 1,000 and add 50,000: the forecast moves identically to within 0.002% of the scale.",
    src: "tinycast/models/student · RobustScaler · web/public/data/explain.json" },
  { topic: "Architecture", q: "Our student reads its whole 64-step forecast from one 128-number token. Why can that be a bottleneck, and what is the fix?",
    a: "One summary vector has to encode every step and every quantile. A multi-token decoder gives each future patch its own query token, as large models do, so the output is decoded patch by patch in one pass.",
    here: "Multi-token vs single token (size m, 1M windows): GIFT-Eval 0.979 [0.959, 0.997], but mixed on our suites (electricity 1.045, worse). Best GIFT score: multi-token, 2.9M, Timestral-209M labels, 0.598.",
    src: "docs/decisions/0008-multi-token-decoder.md · results/study/stage_c9_mt_scale" },
  { topic: "Architecture", q: "How do you guarantee quantile forecasts never cross?",
    a: "By construction: predict the median, then positive gaps (softplus) added outward from it, cumulatively. Sorting outputs after the fact also works but hides the problem from the loss.",
    here: "MonotoneQuantileHead: every student here outputs valid quantiles whatever its weights.",
    src: "tinycast/models/student · MonotoneQuantileHead" },

  // ---------------------------------------------------------------- long horizons & context
  { topic: "Long horizons & context", q: "Your model predicts 64 steps; you need 512. Feeding back the median forecast: what goes wrong?",
    a: "The fed-back median is smoother than any real path, so uncertainty collapses and later blocks are over-confident. Feed back sampled paths instead (correlated over time, e.g. an AR(1) Gaussian copula) and average the forecasts.",
    here: "At 512 steps on shapes, 80% coverage is 0.30 with median feedback and 0.72 with sampled paths. On real seasonal data rollout stays close to the teacher; on shapes it falls far behind (1.47).",
    src: "results/long_horizon/table.md" },
  { topic: "Long horizons & context", q: "Why does context length matter so much on GIFT-Eval?",
    a: "Because the benchmark mixes sampling rates. 512 steps is 3 weeks of hourly data but 85 minutes of 10-second data: daily cycles become invisible. Compare models at equal context before blaming anything else.",
    here: "Toto-2.0-4m at 512 steps instead of 4096 is 6.8% worse. raws-labs TinyCast goes from 0.570 to 0.612 when cut to 512 steps. Our students score 0.603 at both 512 and 2048 steps: they do not yet use long history.",
    src: "results/gift_eval/pairs.json · journal 2026-11-21" },

  // ---------------------------------------------------------------- shipping
  { topic: "Shipping", q: "After int8 quantisation the outputs differ by 4% (99th percentile). Is the model 4% worse?",
    a: "Not necessarily: parity error measures how much the outputs moved, not how much the score moved. Re-run the benchmark on the quantised model; it is the only way to know.",
    here: "int8 vs fp32: 1.001 [1.000, 1.002] on electricity, 1.032 [1.023, 1.042] on shapes for the 1M-window student; 0.997 on electricity for the 3M-window site student. The file is 3× smaller.",
    src: "results/site_pairs.json · results/export/c3_kd_toto_1m.json", more: ["post-compression.html", "Post 5, section 4"] },
  { topic: "Shipping", q: "What cheap inference-time tricks improve a forecaster without retraining, and how do you check they are real?",
    a: "Sign symmetrisation: average f(x) and −f(−x). Downsampling long series to fit the context. Treat each as an experiment: paired comparison against the plain model, on the same benchmark.",
    here: "Flip on GIFT-Eval: 0.989 [0.973, 1.004] on the 1M-window student, nothing on the 3M one (0.998 [0.988, 1.009]): with more data the student seems to learn the symmetry itself. Downsampling at 512 steps: 1.031, worse.",
    src: "docs/journal.md, 2026-11-21 (inference tricks)" },
];

const TOPICS: ("All" | Topic)[] = ["All", "Evaluation", "Leakage & data", "Distillation", "Architecture", "Long horizons & context", "Shipping"];
const KEY = "tinycast-cards-known";
const known = new Set<string>();
try { for (const k of JSON.parse(localStorage.getItem(KEY) ?? "[]")) known.add(k); } catch { /* storage unavailable: start empty */ }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify([...known])); } catch { /* ignore */ } };

let topic: (typeof TOPICS)[number] = "All";
let order = CARDS.map((_, i) => i);
const flipped = new Set<number>();
const $ = (id: string) => document.getElementById(id)!;
const hide = $("c-hide") as HTMLInputElement;
const id = (c: Card) => c.q.slice(0, 48);

function render(): void {
  $("c-topics").innerHTML = TOPICS.map((t) => `<button type="button" data-t="${t}" aria-pressed="${t === topic}">${t}</button>`).join("");
  $("c-topics").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => { topic = b.dataset.t as typeof topic; render(); }));
  const shown = order.filter((i) => (topic === "All" || CARDS[i].topic === topic) && !(hide.checked && known.has(id(CARDS[i]))));
  const inTopic = CARDS.filter((c) => topic === "All" || c.topic === topic);
  $("c-progress").textContent = `${inTopic.filter((c) => known.has(id(c))).length} of ${inTopic.length} marked as known${topic === "All" ? "" : ` in ${topic}`}.`;
  $("c-grid").innerHTML = shown.map((i) => {
    const c = CARDS[i], f = flipped.has(i), k = known.has(id(c));
    return `<article class="card${f ? " flipped" : ""}${k ? " known" : ""}" data-i="${i}">
      <div class="card-top"><span class="card-topic">${c.topic}</span>${k ? `<span class="card-known">known</span>` : ""}</div>
      <p class="card-q">${c.q}</p>
      ${f ? `<div class="card-back"><p>${c.a}</p><p class="card-here"><b>In this project</b> ${c.here}</p>
        <p class="src">${c.src}</p>${c.more ? `<p class="small"><a href="${c.more[0]}">${c.more[1]} →</a></p>` : ""}</div>` : ""}
      <div class="card-actions"><button type="button" data-act="flip">${f ? "Hide answer" : "Show answer"}</button>
        ${f ? `<button type="button" data-act="known">${k ? "Forget" : "I knew it"}</button>` : ""}</div>
    </article>`;
  }).join("") || `<p class="muted">Nothing left here. Untick “hide cards I know” to review them.</p>`;
  $("c-grid").querySelectorAll<HTMLElement>(".card").forEach((el) => {
    const i = +el.dataset.i!;
    el.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = (e) => {
      e.stopPropagation();
      if (b.dataset.act === "flip") { if (flipped.has(i)) flipped.delete(i); else flipped.add(i); }
      else { const k = id(CARDS[i]); if (known.has(k)) known.delete(k); else known.add(k); save(); }
      render();
    }));
  });
}
hide.onchange = render;
$("c-shuffle").onclick = () => { order = order.map((i) => [Math.random(), i] as const).sort((a, b) => a[0] - b[0]).map(([, i]) => i); flipped.clear(); render(); };
$("c-flipall").onclick = () => {
  const vis = [...document.querySelectorAll<HTMLElement>(".card")].map((el) => +el.dataset.i!);
  const allOpen = vis.every((i) => flipped.has(i));
  for (const i of vis) { if (allOpen) flipped.delete(i); else flipped.add(i); }
  render();
};
render();
