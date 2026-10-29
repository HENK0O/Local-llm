const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const html = readFileSync(
  require("node:path").join(__dirname, "../local_llm/web/index.html"),
  "utf8",
);
const extract = (start, end) =>
  html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));

test("navigation selects exactly one workspace and preserves loaded model state", () => {
  const nodes = new Map();
  const $ = (id) => {
    if (!nodes.has(id))
      nodes.set(id, {
        hidden: false,
        attrs: {},
        classList: {
          toggle(name, value) {
            if (name === "hidden") nodes.get(id).hidden = value;
          },
        },
        setAttribute(name, value) {
          this.attrs[name] = value;
        },
      });
    return nodes.get(id);
  };
  const context = {
    $,
    document: { title: "" },
    gpuInfo: { model_id: "ling", loaded: true },
  };
  vm.createContext(context);
  vm.runInContext(
    extract("      function setWorkspace(name)", "      function syncLength()"),
    context,
  );
  for (const selected of ["library", "performance", "chat"]) {
    context.setWorkspace(selected);
    for (const [key, view, tab] of [
      ["library", "libraryView", "libraryTab"],
      ["performance", "performanceView", "performanceTab"],
      ["chat", "chatView", "chatTab"],
    ]) {
      assert.equal($(view).hidden, key !== selected);
      assert.equal($(tab).attrs["aria-selected"], String(key === selected));
      assert.equal($(tab).tabIndex, key === selected ? 0 : -1);
    }
    assert.equal(context.gpuInfo.model_id, "ling");
  }
});

test("catalog fallback never treats unsupported quantization as a load candidate", () => {
  const context = {};
  vm.createContext(context);
  vm.runInContext(
    extract(
      "      function libraryAvailability(model)",
      "      function renderCatalog(data)",
    ),
    context,
  );
  for (const format of ["EXL3", "EXL2", "GPTQ", "AWQ", "BitsAndBytes"]) {
    assert.equal(
      context.libraryAvailability({ format, accelerator_candidate: true }).code,
      "unsupported_format",
    );
  }
  assert.equal(
    context.libraryAvailability({ role: "auxiliary" }).code,
    "auxiliary",
  );
  assert.equal(
    context.libraryAvailability({ format: "GGUF", accelerator_candidate: true })
      .code,
    "candidate",
  );
});
