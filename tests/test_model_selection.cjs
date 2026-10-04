const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const html = readFileSync(
  require("node:path").join(__dirname, "../local_llm/web/index.html"),
  "utf8",
);
const between = (start, end) =>
  html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
const functions = [
  between("      const currentModelKey =", "      function updateTheme()"),
  between("      function setBusy(value)", "      function showInfo(data)"),
  between(
    "      async function loadModel(",
    "      function renderModelPicker()",
  ),
  between(
    "      async function refreshAccelerator()",
    '      $("optimizeModel").onclick',
  ),
  between(
    "      async function activateChat(id)",
    "      function newConversation()",
  ),
].join("\n");
const smol = {
  model_id: "smol",
  model_name: "Smol",
  loaded: true,
  available: true,
};
const ling = { ...smol, model_id: "ling", model_name: "Ling Tiny" };
function fixture(request) {
  const nodes = new Map();
  const $ = (id) => {
    if (!nodes.has(id))
      nodes.set(id, {
        textContent: "",
        disabled: false,
        hidden: false,
        classList: {
          toggle(name, value) {
            if (name === "hidden") nodes.get(id).hidden = value;
          },
        },
        setAttribute() {},
        close() {},
      });
    return nodes.get(id);
  };
  const sandbox = {
    $,
    api: request,
    epoch: 0,
    saveDraft() {},
    setWorkspace() {},
    renderChat() {},
    closeSidebar() {},
    info: { features: ["gpu_runtime", "model_unload"], loaded: false },
    gpuInfo: { ...smol },
    activeGPU: { ...smol },
    activeLM: null,
    lmCatalog: [],
    catalog: [
      { id: "smol", name: "Smol", accelerator_candidate: true },
      { id: "ling", name: "Ling Tiny", accelerator_candidate: true },
    ],
    catalogData: {},
    busy: false,
    controller: null,
    calibrationRunning: false,
    gpuPolling: false,
    modelSelection: null,
    modelSelectionEpoch: 0,
    draftChoices: [],
    completions: [],
    conversations: {
      active: { id: "chat", modelKey: "llamacpp:smol", modelName: "Smol" },
      save() {},
    },
    directModelKey: (key) => key,
    modelChoices: (models) =>
      models.map((m) => ({ key: "llamacpp:" + m.id, name: m.name })),
    document: { querySelectorAll: () => [] },
    updateMeasureButton() {},
    renderOptimization() {},
    refreshContext() {},
    syncLength() {},
    renderCatalog() {},
    updateStorageNotice() {},
    renderResponsePerformance() {},
    refreshModelAdvice() {},
    loadDraftChoices: async () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(functions + "\nglobalThis.canSend = () => ready();", sandbox);
  return { sandbox, $ };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

test("an explicitly requested engine is confirmed as well as the model identity", async () => {
  const { sandbox: s, $ } = fixture(async (path) => {
    if (path === "/v1/accelerator/load" || path === "/v1/accelerator")
      return { ...smol, engine: "mlx" };
    throw new Error("Unexpected request " + path);
  });
  assert.equal(await s.loadModel("llamacpp:smol", null, false, "mtplx"), false);
  assert.equal(s.conversations.active.enginePreference, "mtplx");
  assert.match($("modelLoadFailureMessage").textContent, /n’a pas confirmé/);
  assert.equal(s.canSend(), false);
});

test("returning to a chat restores its explicit engine on the same model", async () => {
  const requests = [];
  const { sandbox: s } = fixture(async (path, body) => {
    if (path === "/v1/accelerator/load") {
      requests.push(body);
      return { ...smol, engine: "mtplx" };
    }
    if (path === "/health") return { features: ["gpu_runtime"], loaded: false };
    throw new Error("Unexpected request " + path);
  });
  s.gpuInfo.engine = s.activeGPU.engine = "mlx";
  s.conversations.select = function (id) {
    this.active = {
      id,
      modelKey: "llamacpp:smol",
      modelName: "Smol",
      enginePreference: "mtplx",
    };
    return this.active;
  };
  await s.activateChat("mtplx-chat");
  assert.equal(requests[0].engine, "mtplx");
  assert.equal(s.canSend(), true);
});

test("failed Ling load keeps Ling selected, persists intent, shows reason and blocks Smol sending", async () => {
  const { sandbox: s, $ } = fixture(async (path, body) => {
    if (path === "/v1/accelerator/load")
      throw new Error("RAM disponible insuffisante : 8,7 Gio disponibles");
    if (path === "/v1/accelerator") return { ...smol };
    throw new Error("Unexpected request " + path);
  });
  assert.equal(await s.loadModel("llamacpp:ling"), false);
  assert.equal($("modelSelectName").textContent, "Ling Tiny");
  assert.equal($("modelStatus").textContent, "Chargement échoué");
  assert.equal($("modelLoadFailure").hidden, false);
  assert.match(
    $("modelLoadFailureMessage").textContent,
    /RAM disponible insuffisante/,
  );
  assert.equal(s.conversations.active.modelKey, "llamacpp:ling");
  assert.equal(s.activeGPU.model_id, "smol"); // Actual runtime remains truthful.
  assert.equal(s.canSend(), false);
  assert.equal($("send").disabled, true);
  await s.refreshAccelerator();
  assert.equal($("modelSelectName").textContent, "Ling Tiny");
  assert.equal(s.canSend(), false);
});

test("an in-flight Smol poll cannot overwrite a successful Ling selection", async () => {
  const stale = deferred(),
    load = deferred();
  const { sandbox: s, $ } = fixture(async (path) => {
    if (path === "/v1/accelerator") return stale.promise;
    if (path === "/v1/accelerator/load") return load.promise;
    if (path === "/health") return { features: ["gpu_runtime"], loaded: false };
    throw new Error("Unexpected request " + path);
  });
  const poll = s.refreshAccelerator();
  const selection = s.loadModel("llamacpp:ling");
  assert.equal($("modelSelectName").textContent, "Ling Tiny");
  assert.equal($("modelStatus").textContent, "Chargement…");
  assert.equal(s.canSend(), false);
  load.resolve({ ...ling });
  assert.equal(await selection, true);
  stale.resolve({ ...smol });
  await poll;
  assert.equal(s.gpuInfo.model_id, "ling");
  assert.equal(s.activeGPU.loaded, true);
  assert.equal($("modelSelectName").textContent, "Ling Tiny");
  assert.equal(s.canSend(), true);
});

test("retry loads the requested model and clears the error only after confirmation", async () => {
  let attempts = 0;
  const { sandbox: s, $ } = fixture(async (path) => {
    if (path === "/v1/accelerator/load") {
      if (++attempts === 1) throw new Error("Not enough RAM");
      return { ...ling };
    }
    if (path === "/v1/accelerator") return { ...smol };
    if (path === "/health") return { features: ["gpu_runtime"], loaded: false };
    throw new Error("Unexpected request " + path);
  });
  await s.loadModel("llamacpp:ling");
  assert.equal(await s.loadModel(s.modelSelection.key), true);
  assert.equal($("modelLoadFailure").hidden, true);
  assert.equal($("modelSelectName").textContent, "Ling Tiny");
  assert.equal(s.canSend(), true);
});

test("a load response for another model is rejected rather than sending to Smol", async () => {
  const { sandbox: s, $ } = fixture(async (path) => ({ ...smol }));
  assert.equal(await s.loadModel("llamacpp:ling"), false);
  assert.match($("modelLoadFailureMessage").textContent, /n’a pas confirmé/);
  assert.equal(s.canSend(), false);
});

test("choosing another conversation reloads its model after a failed switch stopped the worker", async () => {
  const requests = [];
  const { sandbox: s, $ } = fixture(async (path, body) => {
    if (path === "/v1/accelerator/load") {
      requests.push(body.id);
      if (body.id === "ling") throw new Error("Worker could not load Ling");
      return { ...smol };
    }
    if (path === "/v1/accelerator") return { ...smol, loaded: false };
    if (path === "/health") return { features: ["gpu_runtime"], loaded: false };
    throw new Error("Unexpected request " + path);
  });
  await s.loadModel("llamacpp:ling");
  s.conversations.select = function (id) {
    this.active = { id, modelKey: "llamacpp:smol", modelName: "Smol" };
    return this.active;
  };
  await s.activateChat("smol-chat");
  assert.deepEqual(requests, ["ling", "smol"]);
  assert.equal($("modelLoadFailure").hidden, true);
  assert.equal($("modelSelectName").textContent, "Smol");
  assert.equal(s.canSend(), true);
});
