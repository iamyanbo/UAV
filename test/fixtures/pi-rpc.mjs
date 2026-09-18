import { createInterface } from "node:readline";
import { writeFileSync } from "node:fs";
if (process.env.FAKE_COMMAND_PATH) writeFileSync(process.env.FAKE_COMMAND_PATH, JSON.stringify({
  argv: process.argv, childCapacity: process.env.CURI_SUBAGENT_CONCURRENCY,
}));
const output = value => process.stdout.write(JSON.stringify(value) + "\n");
createInterface({ input: process.stdin }).on("line", line => {
  const command = JSON.parse(line);
  const respond = (success = true, data) => output({ type: "response", id: command.id,
    command: command.type, success, data, error: success ? undefined : "preflight rejected" });
  if (command.type === "get_state") {
    setTimeout(() => respond(true, { model: { id: "test-model", provider: "test-provider" } }), Number(process.env.FAKE_READY_DELAY ?? 0));
  } else if (command.type === "prompt") {
    if (command.message === "reject") { respond(false); return; }
    if (command.message === "exit") { process.exit(3); }
    if (command.message === "hang") return;
    if (command.message === "cancel" && process.env.FAKE_CANCEL_FILE) {
      respond();
      writeFileSync(process.env.FAKE_CANCEL_FILE, "operator requested model switch");
      return;
    }
    if (command.message === "retry") {
      const error = "429 Resource exhausted";
      setTimeout(() => {
        respond();
        output({ type: "agent_end", willRetry: true, messages: [{ role: "assistant", stopReason: "error", errorMessage: error, content: [] }] });
        output({ type: "auto_retry_start", attempt: 1, maxAttempts: 3, delayMs: 80, errorMessage: error });
        setTimeout(() => output({ type: "agent_end", willRetry: false,
          messages: [{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "RECOVERED" }] }] }), 80);
      }, 20);
      return;
    }
    setTimeout(() => {
      respond(); output({ type: "agent_end", messages: [{ role: "assistant", content: [{ type: "text", text: "CURRENT" }] }] });
    }, 60);
  } else if (command.type === "abort") {
    respond(); output({ type: "agent_end", messages: [] });
  } else respond();
});
