const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const html = readFileSync(
  require("node:path").join(__dirname, "../local_llm/web/index.html"),
  "utf8",
);
const renderer = html.slice(
  html.indexOf("      function renderOptimization()"),
  html.indexOf("      async function refreshAccelerator()"),
);
// A small DOM substitute exercises the actual renderer without a browser/server.
function element(tag, cls = "", text = "") {
  return {
    tag,
    className: cls,
    textContent: text,
    children: [],
    dataset: {},
    classList: { toggle() {} },
    append(...children) {
      this.children.push(...children);
    },
    replaceChildren(...children) {
      this.children = children;
    },
    querySelector(tag) {
      return this.children.find((c) => c.tag === tag) || null;
    },
  };
}
function render(
  profile,
  features = ["validated_calibration", "gpu_runtime", "usage_profiles"],
  usage = "balanced",
  pending = false,
) {
  const elements = new Map();
  const sandbox = {
    $: (id) => {
      if (!elements.has(id)) elements.set(id, element("div"));
      return elements.get(id);
    },
    node: element,
    fmt: (n) => (Number.isFinite(n) ? n.toFixed(1) : "—"),
    duration: (n) => (Number.isFinite(n) ? Math.round(n * 1000) + " ms" : "—"),
    bytes: (n) => (Number.isFinite(n) ? n + " bytes" : "—"),
    info: { features },
    busy: false,
    calibrationRunning: false,
    activeGPU: { loaded: true, model_id: "target", model_name: "Target" },
    gpuInfo: {
      available: true,
      loaded: true,
      model_id: "target",
      profile,
      usage_profile: usage,
      config: {
        context: 8192,
        slots: 3,
        threads: 0,
        batch: 2048,
        kv_type: "f16",
      },
    },
  };
  vm.runInNewContext(renderer + "\nrenderOptimization();", sandbox);
  if (pending)
    vm.runInNewContext(
      '$("usageProfile").value = "code"; renderOptimization();',
      sandbox,
    );
  const flatten = (e) => [e.textContent, ...e.children.map(flatten)].join(" ");
  return {
    text: flatten(elements.get("optimizationResult")),
    status: elements.get("optimizationStatus").textContent,
    disabled: elements.get("optimizeModel").disabled,
    selected: elements.get("usageProfile").value,
  };
}
function profile() {
  const summary = {
    decode_tps: 100,
    seconds: 4,
    process_rss_bytes: null,
    categories: { discussion: { decode_tps: 100, seconds: 1 } },
  };
  return {
    protocol: 5,
    winner: "standard",
    candidate: "motifs-4",
    config: { speculative: "none" },
    summaries: { standard: summary },
    training_summaries: {
      standard: summary,
      "motifs-4": { ...summary, decode_tps: 200 },
    },
    trials: { standard: {}, "motifs-4": {} },
    decisions: {
      "motifs-4": {
        reason: "Sorties différentes sur les prompts indépendants.",
        stage: "validation indépendante",
      },
    },
    gain_percent: 0,
    decode_gain_percent: 0,
    validation: {
      decision: { reason: "Sorties différentes sur les prompts indépendants." },
    },
    cache_benchmark: {
      cold_first_token_seconds: 0.229,
      warm_first_token_seconds: 0.014,
      prefill_seconds_saved: null,
      cache_verified: false,
    },
    measured_at: "2026-10-02T12:00:00Z",
  };
}
test("failed holdout reports zero gain and explains rejection despite faster selection", () => {
  const view = render(profile());
  assert.match(view.text, /Gain vérifié \+0\.0 tok\/s/);
  assert.match(view.text, /Sorties différentes sur les prompts indépendants/);
  assert.match(view.text, /229 ms/);
  assert.match(view.text, /14 ms/);
  assert.match(view.text, /Préparation économisée par le cache Non vérifiée/);
  assert.match(view.text, /RSS après essai —/);
  assert.equal(view.disabled, false);
});
test("old server and old profiles cannot advertise the new validation", () => {
  const view = render({ protocol: 3, winner: "old", config: {} }, [
    "gpu_runtime",
  ]);
  assert.equal(view.disabled, true);
  assert.match(view.status, /Relancez le serveur/);
  assert.match(view.text, /Ancien profil/);
  assert.doesNotMatch(view.text, /Gain vérifié/);
});
test("quantized KV is presented as a precision change with limited quality evidence", () => {
  const report = profile();
  report.kv_precision_changed = true;
  const view = render(report);
  assert.match(view.text, /cache KV passe de F16 à Q8/);
  assert.match(view.text, /sans garantie générale de qualité/);
});
test("all embedded production scripts are syntactically valid", () => {
  for (const match of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g))
    new vm.Script(match[1]);
});

test("specialized profile shows only its scoped verified gain and uses its own decision", () => {
  const report = profile();
  const baseline = report.summaries.standard;
  const retained = { ...baseline, decode_tps: 150, seconds: 0.7 };
  report.summaries["code-fast"] = retained;
  report.profiles = {
    balanced: {
      winner: "standard",
      config: report.config,
      baseline,
      retained: baseline,
      decision: { reason: "Standard conservé" },
      gain_percent: 0,
      decode_gain_percent: 0,
    },
    code: {
      winner: "code-fast",
      config: { speculative: "ngram-simple", draft_tokens: 48, kv_type: "f16" },
      category: "code",
      baseline,
      retained,
      decision: { reason: "Code validé indépendamment" },
      gain_percent: 42.9,
      decode_gain_percent: 50,
    },
  };
  report.speculation = {
    "code-fast": {
      proposed_tokens: 60,
      accepted_tokens: 42,
      acceptance_percent: 70,
    },
  };
  const view = render(
    report,
    ["usage_profiles", "validated_calibration", "gpu_runtime"],
    "code",
  );
  assert.match(view.text, /Gain vérifié \+50\.0 tok\/s/);
  assert.match(view.text, /type « code » uniquement/);
  assert.match(view.text, /Code validé indépendamment/);
  assert.match(view.text, /42 \/ 60 tokens anticipés acceptés/);
  assert.doesNotMatch(view.text, /Premier token · contexte froid/);
  assert.equal(view.selected, "code");
});

test("polling preserves a pending profile selection", () => {
  const report = profile();
  const summary = report.summaries.standard;
  report.profiles = {
    balanced: {
      winner: "standard",
      config: report.config,
      baseline: summary,
      retained: summary,
      decision: { reason: "Standard conservé" },
      gain_percent: 0,
      decode_gain_percent: 0,
    },
  };
  assert.equal(
    render(
      report,
      ["usage_profiles", "validated_calibration"],
      "balanced",
      true,
    ).selected,
    "code",
  );
});
