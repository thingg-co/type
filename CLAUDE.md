# Type

Android keyboard (IME) with on-device LLM spell correction via llama.cpp. English only for
now; Thai is planned. arm64-v8a only. Licensed PolyForm Noncommercial 1.0.0.

## Build and test

- Build: `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew assembleDebug`
  (first build compiles llama.cpp, a few minutes; needs NDK 28.2 and CMake 3.31.6 from SDK Manager).
- Release builds are signed only when BOARD_KEYSTORE/BOARD_KEYSTORE_PASS/BOARD_KEY_PASS (historic names, unchanged) are set
  (local keystore, not in the repo).
- Install + activate on a device/emulator: `tools/sideload.sh [debug [model.gguf]]`.
- Model/prompt evaluation: `tools/eval.py --server <llama-server> model.gguf ...` replays the
  app's exact prompts; keep it in sync with `llm/Prompts.kt` whenever prompts change.
- Python tooling tests: `python -m pytest -q tools` from the repo root with a venv that has torch
  (on the Spark `~/lab/.venv/bin/python`). They cover the TNW asset format round trip, the batched
  evaluator, the sweep runner, staging, the corpus preparer and the bigram builder.
- `./gradlew test` runs the JVM suites (DictionaryTest, TypingPolicyTest, CorrectionFilterTest);
  verify UI changes by driving the emulator (adb `input tap`, `screencap`,
  logcat tags TypeIME, SpellLlm, TypeLLM, Dictionary). Pace scripted taps ~250 ms apart.
  Never `am force-stop` the app: Android silently falls back to Gboard.

## Structure

- `app/src/main/cpp/llm_jni.cpp` - JNI bridge; owns the llama.cpp context. The fixed prompt
  prefix is decoded once (`setPrefix`) and rolled back per request (`llama_memory_seq_rm`),
  which is what makes corrections fast. Single-word output is GBNF-constrained.
- `llm/` - `SpellLlm` (modes, prefix split via chat template, output validation), `Prompts`.
- `dict/` - 126k-word frequency list + trie (assets/en_words.txt): known-word gate,
  keyboard-weighted (KeyNeighbors) bounded edit-distance suggestions, slip-tolerant prefix
  completions (`slipPredictions`: a bounded-distance walk of the whole trie, so a slip inside a
  longer word still completes to "better" or "beyond"), adaptive-key weights;
  Contractions auto-apostrophizes bare forms before the known-word gate; Confusables lists
  swappable words (then/than, there/their/they're) whose margins the prediction network
  decides — thresholds calibrated by tools/confusables_calibrate.py + ConfusableCalibrationHarness,
  pinned by ConfusableMarginProbeTest. `Lexer` finds previous tokens;
  `NeuralLm` (assets/en_nextword.bin; since 0.8.0 a recurrent TNW5 net from tools/nn/train_gru.py,
  golden-vector tested) predicts and ranks from the sentence so far; the dense nets it replaced
  used a K-word window through a trunk of one or more layers (TNW3 layout: header V, K, E, L; per layer out, in, W, b;
  TNW1/TNW2 one-layer assets still load; TNW5 is a recurrent trunk: header V, E, H, L, then per
  GRU layer W_ih, W_hh, b_ih, b_hh in PyTorch gate order and a linear map back to E, trained by
  tools/nn/train_gru.py on whole sentences). A recurrent net gets the whole sentence as context
  (`contextWords`, 40) and keeps the state after every position of the last prefix, so a word
  costs one GRU step; the dense nets keep their five-word context. The dense sweep ended at two
  layers of 1024 on the mixed corpus (0.7.4): two layers beat one, three do not, and width past
  1024 or more dialogue data buys nothing at that size; the recurrent trunk is where the next
  gains came from. Train on the Spark (CUDA, ~17 min per 60k steps) rather than the Mac (MPS corrupts the
  126k-wide top-k in eval). The format lives in tools/nn/tnw.py (export, reader, and the exact
  quantized forward the app mirrors; TNW4 adds an untied output table, which trained worse under
  the sampled softmax and is not used). A sweep is a json of runs for tools/nn/sweep.py, which
  writes a results table; tools/nn/stage.py checks a run's golden vector and copies its asset
  into the app. Corpora on the Spark: ~/type-data/data126k (news, wiki, Tatoeba; its val.bin is
  the headline metric) and data_mix (the same plus 5.9M OpenSubtitles lines; data_subs/val.bin
  is the conversational metric, where every net scores about ten points lower). Fallback is
  `Bigrams` (assets/en_bigrams.bin, tools/build_bigrams.py); `Personalizer` learns sparse per-user deltas over the
  frozen network (state in files/personal.bin, shape-checked); `TypingPolicy` combines them.
  Word ids in BOTH binary
  assets are en_words.txt line numbers and expansion is append-only (ids never move,
  tools/expand_vocab.py / tools/curated_words.py). The network is trained on the full
  126k list; the bigram table packs 16-bit ids, so words past line 65536 simply have
  no bigrams (Bigrams.kt guards the range). Reordering or shrinking the list still
  means rebuilding both assets together.
- `ime/` - `TypeInputMethodService` (word tracking, boundary autocorrect with undo, sentence
  fix), `KeyboardView` (Canvas-drawn, adaptive keys), `SuggestionStripView`, `KeyboardLayouts`.
- `model/` - catalog (chosen from eval results in tools/eval.py; don't swap models without
  rerunning it), resumable downloader, SAF import.
- `third_party/llama.cpp` - submodule; the JNI code tracks the current llama.h API.

## Conventions

- Layouts and dictionaries are data; adding a language must not touch input logic.
- All llama.cpp calls stay on SpellLlm's single thread; UI work stays off it.
- Correction safety order: dictionary gate -> model -> edit-distance/dictionary validation ->
  user-visible undo. Anything that weakens a step needs a stronger one elsewhere.
