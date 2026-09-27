# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For implementation tasks, record the user's requirements, promised work, affected implementations
and routes, and the quality dimensions to be evaluated before judging results. For comparative
claims, also identify the baseline, workloads, metrics, aggregation method, and acceptance criteria.
Keep this accounting current as the task develops. Explain justified changes and preserve the
earlier findings; never narrow the scope or change the criteria after seeing unfavorable results
to make the task appear successful. This accounting may live in the working conversation or
existing task artifacts; it does not require a new permanent planning document.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

Requests for analysis or proposals for review do not authorize applying them;
implementation authorization must cover that same scope.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. The implementation targets `sm_120a` and
is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## Codex CLI (Responses API) compatibility

NInfer's `openai_responses_*` serve layer is the integration point for OpenAI Codex CLI
(native Rust binary, `wire_api = "responses"`). Codex is stricter than the OpenAI HTTP
contract in a few ways that NInfer's parser must honor. The Mac-side Codex config
(`~/.codex/config.toml`, `auth.json`, `local-models.json`) lives outside this repo; see the
`codex-ninfer` skill.

### Field-by-field contract (Codex wire → NInfer change → why)

- **`additional_tools` input items** — Codex declares its tool set as `{"type":"additional_tools","tools":[...]}`
  items inside `input`, not a top-level `tools` array. `parse_input` skips them (`continue`) and
  `parse_tools` folds them in. **Ordering gotcha:** `parse_input` runs *before* `parse_tools`
  (openai_responses_http.cpp), so a `custom_tool_call`/`function_call` in history is lowered
  before its tool declaration is seen. `lower_function_identity` therefore emplaces identities
  lazily and keeps the `freeform` flag sticky (`position->second.freeform ||= identity.freeform`).
- **Namespaced tools** — Codex nests tools under `{"type":"namespace","name":"...","tools":[...]}`
  (e.g. `collaboration.spawn_agent`, `functions.exec_command`). Nested tools may be `function`
  *or* `custom`; the namespace loop must accept both. Engine names are flattened to
  `namespace__name` (`lower_function_identity`); the response side (`add_wire_function_identity`)
  splits them back into `name` + `namespace` for the wire, so Codex recognizes the call.
- **Freeform (`custom`) tools** — A `custom` tool's schema is synthesized as
  `{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}`
  (`make_freeform_tool`). The model emits `{"input":"<raw text>"}`; `emit_tool_arguments`
  unwraps the single string param to raw text for the `custom_tool_call.input` field.
  **History gotcha (was a 500):** `parse_custom_tool_call_item` must store
  `arguments_json` as the JSON object `{"input":"..."}` (`.dump()`), *not* the bare string —
  the Qwen3.5 chat template (`chat_template.cpp`) does `Json::parse(arguments_json)`, which
  throws on a bare shell string → uncaught `parse_error` → HTTP 500 "failed during prepare".
- **`custom_tool_call_output.output`** — Codex sends `output` as a string *or* a JSON value.
  `parse_custom_tool_call_output_item` must accept both (`is_string() ? get : dump()`); a
  hard `is_string()` check 400s on JSON-valued output.
- **`agent_message` items** (multi-agent mode) — Codex with the `multi_agent` feature on spawns sub-agents and emits `agent_message` items (`author`/`recipient` + `content` parts of `input_text`/`encrypted_content`). `parse_input` carries them as a user message (author/recipient header + the parts), and `encrypted_content` (plain text, not really encrypted) is surfaced as a text part. So multi-agent Codex works against the single NInfer model.
- **`parallel_tool_calls`** — Codex sends `parallel_tool_calls: false` alongside tools. NInfer
  must not reject it (the field is echoed in the response, not enforced).
- **`reasoning.summary` / `include:["reasoning.encrypted_content"]`** — handled by
  `reasoning_summary` / `add_reasoning_encrypted_content` (the latter is an opaque mirror,
  not real encryption).

### Build & deploy (Docker)

- Build via `DOCKER_BUILDKIT=1 docker build -t ninfer:latest .` in `~/Servers/ninfer_build`.
  The build stage uses a Ninja cache mount (`RUN --mount=type=cache,target=/build ...`), so
  incremental rebuilds are ~30 s once the cache is warm (first build ~5 min).
- **Cache-mount + COPY gotcha:** a `COPY --from=build /build/apps/ninfer ...` reads from the
  cache-mounted `/build`, and BuildKit's cache-key checksum of that path can fail with
  `"/build/apps/ninfer": not found` after a failed link poisons the cache ref. The Dockerfile
  works around it by `cp`-ing the binaries out of the cache mount into `/src` (a normal layer)
  and `COPY --from=build /src/ninfer ...`. Don't "simplify" this back to copying from `/build`.
- Deploy with `deploy_ninfer.sh` (docker stop/rm `ninfer-qwen38-27b`, `docker run` `ninfer:latest`,
  nvidia runtime, port 8020, SELinux `:z` mounts, `--model-id local --api-key rjman`).

### Verifying end-to-end

`codex exec --disable multi_agent --sandbox workspace-write "create /tmp/x with hello, read it back"`
must complete with the file created. The `multi_agent` feature flag (on by default) makes Codex
spawn sub-agents and emit `agent_message` items, which NInfer does not parse — multi-agent Codex is supported (see `agent_message` above). Watch the container log for `openai-responses ... done` (not
`failed during prepare | HTTP 500`).
