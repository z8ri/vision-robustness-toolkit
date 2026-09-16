# Engineering notes

Deeper write-ups of the non-obvious decisions and real bugs behind this repo. Written for someone deciding whether the six-line README summary is worth trusting — here's the evidence.

## 1. The ONNX export bug (Component 2 → discovered in Component 6)

**Symptom.** `torch.onnx.export` on any model containing `SWFA` (the S-WFA module from `models/wfa.py`) failed with:

```
torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of convolution for kernel of unknown shape.
```

The first stack trace shown in the terminal was truncated to "Torch IR graph at exception" with no visible error message above it — the actual exception only showed up after capturing full stderr to a file and grepping for `SymbolicValueError`.

**Root cause.** The original `models/wavelet.py` built the Haar DWT/IDWT depthwise convolution kernel *inline*, inside `forward()`, via something like `.expand(channels, 1, 2, 2).contiguous()`, recomputed on every call. That's harmless for eager-mode PyTorch — the tensor is fully concrete at runtime. But `torch.onnx.export` uses the legacy TorchScript-based tracer, which needs to resolve tensor shapes *statically* at trace time to bake a convolution op into the graph. A kernel built by expanding a scalar constant at trace time doesn't carry the same static-shape guarantee a `Parameter` or `register_buffer` does, and the tracer refused to proceed.

**Fix.** Converted `HaarDWT` and `HaarIDWT` from free functions into `nn.Module`s that call `self.register_buffer(f"kernel_{band}", ...)` once, in `__init__`, when `channels` is already a concrete Python int. `WFACore` now holds `self.dwt = HaarDWT(channels)` / `self.idwt = HaarIDWT(channels)` as submodules instead of calling free functions per forward pass. Three lines changed the module's structure; zero lines changed its math.

**Verification, not assumption.** After the fix:
- Re-ran the full pre-existing 158-test suite (Components 1–5) — all still passed, confirming the refactor didn't silently change any numerical behavior.
- Manual export smoke test: exported, then compared PyTorch vs. ONNX Runtime outputs on fresh random input — max abs diff ≈ 2.98e-8, argmax agreement 100%.
- Ran inference at a **different batch size than the model was exported at**, to confirm the dynamic batch axis actually works, not just that the file loads.

**Why this is worth mentioning at all:** a conv-based DWT/IDWT *looks* export-safe on paper — convolutions are one of the best-supported ONNX ops. The failure only shows up when you actually attempt the integration, which is exactly why Component 6 exists as a separate, later component rather than being declared "done" the moment Component 2's own unit tests passed.

## 2. The "split guard" pattern, used three times on purpose

Three unrelated components independently need the same protection: a piece of code must never be run on the wrong data split, because doing so silently produces a number that looks legitimate but doesn't generalize.

| Component | Guarded call | What leaking here would mean |
|---|---|---|
| A-PhysDeg | `DiagnosticBatchRunner.run(difficulty_fn, split="train")` | The adaptive sampler starts prioritizing whatever the model is currently weak on *at test time* — the curriculum would be shaped by exactly the numbers it's supposed to be evaluated against. |
| Robustness Cube | `FinalizeGuard.finalize("test"/"ood")` | Re-running "the final test report" after tweaking something turns a held-out evaluation into a second round of hyperparameter search with extra steps. |
| Selective Prediction | `SelectivePredictor.fit(..., split="calibration")` | Picking a rejection threshold using the same data it gets reported against inflates the reported coverage/risk trade-off. |

Each guard is deliberately implemented as "pass an explicit string, or raise" rather than inferring the split from context. It's documented (in `eval/split_guard.py`'s own docstring) as a **process discipline check, not a cryptographic guarantee** — it can't stop someone from constructing a fresh guard object to route around it. The value is in making the leak-shaped mistake require an active, visible decision instead of an easy accident.

## 3. Design claims as test cases, not comments

Three claims in the design docs this project draws from were treated as falsifiable, not accepted on faith:

- **"A-PhysDeg reduces to static PhysDeg when fully unlocked and difficulty is uniform."** Verified by drawing 4000 chains from both samplers under those conditions and comparing the empirical (type, severity) marginal distributions — not just eyeballing that the code "looks like" it should reduce correctly.
- **"Original WFA is S-WFA with the sample gate forced to 1."** `SWFA(use_gate=False)` and `SWFA(use_gate=True, force_gate=1.0)` are asserted to produce bit-for-bit identical output on the same input, for the same underlying `WFACore` weights.
- **"Temperature scaling never changes the predicted class."** `softmax(z/T)` and `softmax(z)` share the same argmax for any `T > 0` — this is mathematically obvious, but it's still asserted directly on real LBFGS-fit temperatures rather than left as an unverified footnote, since a bug in `apply_temperature`'s division axis would silently violate it.

## 4. What's explicitly *not* proven here (and why)

- **GPU execution-provider fallback detection** (`deploy/verify.py::check_no_fallback`) is only exercised against `CPUExecutionProvider` on this machine, because there's no GPU to request `CUDAExecutionProvider` against and observe a real fallback. The test suite says this directly instead of silently only covering the boring case.
- **`SpectralGate`'s claim of "not learning a background/material shortcut"** is a real concern named in the source design doc's own validation checklist, and it requires real data and real training to check meaningfully. The tests here only confirm the gate responds differently to two hand-built synthetic images with different spectral content — a much weaker property than "doesn't shortcut on real defect imagery."
- **Every accuracy/latency figure describing an idealized version of this project** is a theoretical projection from the design document, not something measured by running this code. Nothing in this repository reproduces or validates those numbers, and nothing here should be read as claiming to.
