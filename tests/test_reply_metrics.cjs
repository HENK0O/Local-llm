const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const html = fs.readFileSync(
  require("node:path").join(__dirname, "../local_llm/web/index.html"),
  "utf8",
);
const start = html.indexOf("      function contextReusePresentation(stats)");
const end = html.indexOf("      function appendReplyFooter(", start);
const context = vm.createContext({});
vm.runInContext(html.slice(start, end), context);
test("prefix reuse is labeled as reused input and never advertised as app speedup", () => {
  for (const backend of ["llamacpp", "native", "lmstudio"]) {
    const data = context.contextReusePresentation({
      backend,
      reused_prompt_tokens: 123,
    });
    assert.equal(data.text, "123 tok réutilisés");
    assert.equal(data.known, true);
    assert.match(data.tooltip, /ne mesure pas une accélération/);
    assert.doesNotMatch(
      data.text + data.tooltip,
      /tokens gagnés|grâce à|\+123/,
    );
  }
});
test("zero reuse and unavailable metrics are distinct and never inferred", () => {
  assert.equal(
    context.contextReusePresentation({ reused_prompt_tokens: 0 }).text,
    "0 tok réutilisés",
  );
  for (const count of [undefined, null, -1, 1.2, NaN, "12"]) {
    const data = context.contextReusePresentation({
      reused_prompt_tokens: count,
    });
    assert.equal(data.known, false);
    assert.equal(data.text, "Cache non mesuré");
  }
});
