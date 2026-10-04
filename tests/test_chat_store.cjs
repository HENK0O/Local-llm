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
const sandbox = { module: { exports: {} }, crypto: { randomUUID }, URL };
vm.runInNewContext(source, sandbox);
const {
  ConversationStore,
  contextMessages,
  captureContextRequest,
  modelChoices,
  selectedLMInstance,
  directModelKey,
  verifiedHubLink,
  streamDelta,
  completionNote,
} = sandbox.module.exports;
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

test("context updates after replies and excludes drafts and failed exchanges", () => {
  const store = new ConversationStore(storage());
  const chat = store.create("one", "One");
  chat.draft = "Not sent";
  chat.messages.push({ role: "user", content: "Remember A" });
  assert.equal(contextMessages(chat).length, 1);
  chat.messages.push({ role: "assistant", content: "Answer A" });
  chat.messages.push({ role: "user", content: "Failed", inContext: false });
  chat.messages.push({ role: "assistant", content: "", inContext: false });
  assert.equal(
    JSON.stringify(contextMessages(chat)),
    JSON.stringify([
      { role: "user", content: "Remember A" },
      { role: "assistant", content: "Answer A" },
    ]),
  );
  const second = store.create("two", "Two");
  assert.equal(contextMessages(second).length, 0);
  store.select(chat.id);
  assert.equal(contextMessages(store.active).length, 2);
});

test("last request stays exact after a failure, reply and reload", () => {
  const disk = storage();
  const store = new ConversationStore(disk);
  const chat = store.create("one", "One");
  chat.messages.push(
    { role: "user", content: "Old failure", inContext: false },
    { role: "user", content: "Sent question" },
  );
  chat.lastRequest = captureContextRequest(chat, "local", "one");
  // A failure excludes the question from future context, not from what was sent.
  chat.messages[1].inContext = false;
  chat.messages.push({ role: "assistant", content: "Later response" });
  store.touch(chat);
  const restored = new ConversationStore(disk).active;
  assert.equal(
    contextMessages(restored, restored.lastRequest)[0].content,
    "Sent question",
  );
  assert.equal(contextMessages(restored, restored.lastRequest).length, 1);
  assert.equal(contextMessages(restored)[0].content, "Later response");
  assert.equal(
    contextMessages(restored, { messageIndices: [-1, "0", 999] }).length,
    0,
  );
});

test("picker contains usable models without grey external duplicates", () => {
  const choices = modelChoices(
    [
      { id: "native", name: "Local", compatible: true },
      { id: "ling-file", name: "Ling Q8", compatible: false },
      { id: "qwen-file", name: "Qwen IQ3", compatible: false },
    ],
    [
      {
        id: "ling",
        name: "Ling",
        instances: [{ id: "ling-one" }, { id: "ling-two" }],
      },
      { id: "qwen", name: "Qwen", loaded: false, instances: [] },
    ],
  );
  assert.equal(
    JSON.stringify(choices.map((m) => m.key)),
    JSON.stringify(["native", "lmstudio:ling", "lmstudio:ling"]),
  );
});

test("an unloaded or ambiguous LM selection never picks another instance", () => {
  assert.equal(selectedLMInstance({ instances: [] }), null);
  assert.equal(selectedLMInstance({ instances: [{ id: "one" }] }), "one");
  const multiple = { instances: [{ id: "one" }, { id: "two" }] };
  assert.equal(selectedLMInstance(multiple), null);
  assert.equal(selectedLMInstance(multiple, "two"), "two");
  assert.equal(selectedLMInstance(multiple, "previously-loaded"), null);
});

test("direct GGUF choices include unsupported native quantizations without duplicate CPU entries", () => {
  const choices = modelChoices(
    [
      { id: "q8", name: "Q8", compatible: true, accelerator_candidate: true },
      {
        id: "iq3",
        name: "IQ3",
        compatible: false,
        accelerator_candidate: true,
      },
      {
        id: "aux",
        name: "DFlash",
        compatible: false,
        accelerator_candidate: false,
      },
      { id: "directory", name: "Native directory", compatible: true },
    ],
    [],
    true,
  );
  assert.equal(
    JSON.stringify(choices.map((m) => m.key)),
    JSON.stringify(["llamacpp:q8", "llamacpp:iq3", "directory"]),
  );
  assert.equal(choices[0].source, "Sur cet appareil · moteur direct");
});

test("GPU request identity survives conversation reload without mixing other contexts", () => {
  const disk = storage(),
    store = new ConversationStore(disk);
  const first = store.create("llamacpp:gguf", "GPU GGUF");
  first.messages.push({ role: "user", content: "Context A" });
  first.lastRequest = captureContextRequest(first, "llamacpp", "gguf");
  store.touch(first);
  const second = store.create("llamacpp:gguf", "GPU GGUF");
  second.messages.push({ role: "user", content: "Context B" });
  store.touch(second);
  store.select(first.id);
  const reloaded = new ConversationStore(disk);
  assert.equal(reloaded.active.lastRequest.backend, "llamacpp");
  assert.equal(reloaded.active.lastRequest.model, "gguf");
  assert.equal(contextMessages(reloaded.active)[0].content, "Context A");
  assert.notEqual(first.id, second.id);
});

test("LM Studio reasoning channels stay separate from the answer", () => {
  assert.equal(
    streamDelta({
      choices: [{ delta: { reasoning_content: "Thinking", content: null } }],
    }).reasoning,
    "Thinking",
  );
  const alternate = streamDelta({
    choices: [{ delta: { reasoning: "Thinking too", content: "Answer" } }],
  });
  assert.equal(alternate.content, "Answer");
  assert.equal(alternate.reasoning, "Thinking too");
  assert.equal(
    streamDelta({
      choices: [
        { delta: { reasoning: "Alias", reasoning_content: "Primary" } },
      ],
    }).reasoning,
    "Primary",
  );
  assert.equal(streamDelta({ usage: {} }).content, "");
  assert.equal(
    streamDelta({ choices: [{ delta: { content: 123, reasoning: {} } }] })
      .content,
    "",
  );
});

test("reasoning-only or empty completions explain the missing answer", () => {
  assert.match(
    completionNote("", "Thought", "length", 256),
    /256 tokens pendant sa réflexion/,
  );
  assert.match(
    completionNote("", "Thought", "stop", 256),
    /sans produire de réponse/,
  );
  assert.match(completionNote("", "", "stop", 256), /aucun texte/);
  assert.equal(completionNote("Answer", "Thought", "length", 256), "");
  const chat = {
    messages: [
      { role: "user", content: "Question" },
      {
        role: "assistant",
        content: "",
        reasoning: "Thought",
        inContext: false,
      },
    ],
  };
  assert.equal(contextMessages(chat).length, 1);
  assert.equal(contextMessages(chat)[0].content, "Question");
});

test("legacy LM chats map only an exact installed file key to direct execution", () => {
  const local = [
    {
      id: "exact",
      accelerator_candidate: true,
      lmstudio_key: "owner/repo/file.gguf",
    },
  ];
  assert.equal(
    directModelKey("lmstudio:owner/repo/file.gguf", local, true),
    "llamacpp:exact",
  );
  assert.equal(directModelKey("lmstudio:file", local, true), "lmstudio:file");
  assert.equal(
    directModelKey("lmstudio:owner/repo/file.gguf", local, false),
    "lmstudio:owner/repo/file.gguf",
  );
  assert.equal(
    directModelKey(
      "lmstudio:owner/repo/file.gguf",
      [...local, { ...local[0], id: "duplicate" }],
      true,
    ),
    "lmstudio:owner/repo/file.gguf",
  );
});
test("variant links allow only direct revision pinned Hugging Face pages", () => {
  const good =
    "https://huggingface.co/owner/repo/blob/" + "a".repeat(40) + "/model.gguf";
  assert.equal(verifiedHubLink(good), good);
  for (const bad of [
    "javascript:alert(1)",
    "https://huggingface.co.evil.test/a",
    good.replace("huggingface.co", "user:token@huggingface.co"),
    good.replace("https:", "http:"),
    good.replace("a".repeat(40), "main"),
  ])
    assert.equal(verifiedHubLink(bad), null);
});
