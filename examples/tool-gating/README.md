# tool-gating: SYNTHETIC tool calls for a support agent

The state is the JSON of a tool call the agent proposes. The question is whether it may run: `allow`, `deny` or `ask_human`. All 36 calls (24 train, 12 test) are synthetic.

`rules.json` shows what belongs in code:

- destructive SQL and a deny-list of tools: `deny`, final;
- any amount above 10,000: `ask_human`, final (a threshold is a rule, not a model opinion);
- a read-only allowlist: `allow`;
- side-effecting tools: `ask_human`.

The rules abstain (confidence 0) on `send_email` and `send_reply`, where the risk depends on content. A gate escalates those to the local model, an LLM or a person. With only 12 test examples, this folder is a demonstration, not a benchmark.
