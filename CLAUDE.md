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
- `./gradlew test` runs the JVM suites (DictionaryTest, TypingPolicyTest, CorrectionFilterTest);
  verify UI changes by driving the emulator (adb `input tap`, `screencap`,
  logcat tags TypeIME, SpellLlm, TypeLLM, Dictionary). Pace scripted taps ~250 ms apart.
  Never `am force-stop` the app: Android silently falls back to Gboard.

## Structure

- `app/src/main/cpp/llm_jni.cpp` - JNI bridge; owns the llama.cpp context. The fixed prompt
  prefix is decoded once (`setPrefix`) and rolled back per request (`llama_memory_seq_rm`),
  which is what makes corrections fast. Single-word output is GBNF-constrained.
- `llm/` - `SpellLlm` (modes, prefix split via chat template, output validation), `Prompts`.
- `dict/` - 56k-word frequency list + trie (assets/en_words.txt): known-word gate, bounded
  Damerau-Levenshtein suggestions, adaptive-key weights.
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
