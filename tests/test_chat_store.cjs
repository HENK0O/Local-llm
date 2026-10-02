const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { randomUUID } = require("node:crypto");
const vm = require("node:vm");

const html = readFileSync(
  require("node:path").join(__dirname, "../local_llm/web/index.html"),
  "utf8",
);
const source = html.match(
  /<script id="conversation-store">([\s\S]*?)<\/script>/,
)[1];
const sandbox = { module: { exports: {} }, crypto: { randomUUID } };
vm.runInNewContext(source, sandbox);
const { ConversationStore } = sandbox.module.exports;
function storage() {
  const data = new Map();
  return {
    getItem: (key) => data.get(key) || null,
    setItem: (key, value) => data.set(key, value),
  };
}

test("separate contexts and drafts survive reload and selection", () => {
  const disk = storage();
  const store = new ConversationStore(disk);
  const first = store.create("one", "Model One");
  first.messages.push(
    { role: "user", content: "Context A" },
    { role: "assistant", content: "Answer A" },
  );
  first.draft = "Draft A";
  store.touch(first);
  const second = store.create("two", "Model Two");
  second.messages.push({ role: "user", content: "Context B" });
  store.touch(second);
  store.select(first.id);
  const reopened = new ConversationStore(disk);
  assert.equal(reopened.active.modelKey, "one");
  assert.equal(reopened.active.messages[1].content, "Answer A");
  assert.equal(reopened.active.draft, "Draft A");
  reopened.select(second.id);
  assert.equal(reopened.active.messages.length, 1);
  assert.equal(reopened.active.messages[0].content, "Context B");
});

test("archiving is recoverable and keeps the complete history", () => {
  const disk = storage();
  const store = new ConversationStore(disk);
  const first = store.create("one", "One");
  first.messages.push({ role: "user", content: "Keep this" });
  store.touch(first);
  const second = store.create("two", "Two");
  store.archive(first.id);
  const reopened = new ConversationStore(disk);
  assert.equal(reopened.visible().length, 1);
  assert.equal(reopened.activeId, second.id);
  reopened.restore(first.id);
  reopened.select(first.id);
  assert.equal(reopened.active.messages[0].content, "Keep this");
});

test("storage failure leaves the current conversation intact in memory", () => {
  const store = new ConversationStore({
    getItem: () => null,
    setItem: () => {
      throw new Error("Quota");
    },
  });
  const chat = store.create("one", "One");
  chat.messages.push({ role: "user", content: "Keep this session" });
  assert.equal(store.save(), false);
  assert.ok(store.error);
  assert.equal(store.active.messages[0].content, "Keep this session");
});

test("invalid or duplicate saved conversations are rejected without executing contents", () => {
  const disk = storage();
  disk.setItem(
    "local-llm-conversations-v1",
    JSON.stringify({
      version: 1,
      chats: [
        {
          id: "same",
          title: "<script>never execute</script>",
          updatedAt: 1,
          messages: [{ role: "user", content: "<script>text only</script>" }],
        },
        { id: "same", title: "Duplicate", messages: [] },
        {
          id: "bad",
          title: "Bad",
          messages: [{ role: "system", content: "Invalid role" }],
        },
      ],
    }),
  );
  const store = new ConversationStore(disk);
  assert.equal(store.visible().length, 1);
  assert.equal(store.active.title, "<script>never execute</script>");
});
