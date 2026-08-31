# Type

An Android keyboard that fixes typos with a small language model running on the phone.
Everything happens on-device: the model is loaded into the keyboard process through
llama.cpp, and nothing you type is sent anywhere.

Site: https://thingg-co.github.io/type/

## What it does

- Standard QWERTY keyboard with shift, symbols layers, key repeat and haptics.
- When a word you finish is not in the dictionary, the model is asked what you meant. If the
  answer is close enough to what you typed (edit distance and dictionary checks), the word is
  replaced. An undo chip appears in the suggestion bar; backspace straight after a correction
  restores the original too.
- While you type, the bar is context-aware: a small lexer reads the previous token and a
  packed bigram table (260k pairs from Norvig's web corpus) ranks completions, orders the
  word keys, and suggests the next word right after a space. If what you typed cannot start
  any dictionary word, dictionary candidates appear immediately and the model is asked after
  a short pause.
- The ✨ button sends the current sentence to the model for a whole-sentence pass.
- Adaptive keys: letters that cannot continue any dictionary word shrink and fade as you type.
  Hit areas do not change, so unusual words stay typable.
- Word keys: when what you have typed can only end a few ways, the suggestion bar becomes
  whole-word keys (type "restau" and tap "restaurant"). Plural and possessive variants count
  as one family.
- Passwords, email and URL fields, and fields that opt out of suggestions are left alone.

## Layout of the code

```
app/src/main/cpp/llm_jni.cpp        JNI bridge to llama.cpp: load, prefix caching, greedy decode with GBNF grammar
app/src/main/java/.../llm/          LlamaNative (JNI), Prompts (few-shot prompt), SpellLlm (modes, validation)
app/src/main/java/.../dict/         Dictionary: frequency list + trie, edit-distance suggestions
app/src/main/java/.../ime/          InputMethodService, KeyboardView, SuggestionStripView, layouts
app/src/main/java/.../model/        Model catalog, storage, resumable downloader
app/src/main/java/.../SettingsActivity.kt   setup steps, model download, options, test field
app/src/main/assets/en_words.txt    52k English words, most frequent first
tools/                              desktop eval harness used to choose the model and prompt
public/                             the GitHub Pages site
third_party/llama.cpp               git submodule
```

The fixed part of the prompt (instructions plus a few examples) is decoded once and kept in the
KV cache; a correction only decodes the request's own tokens and generates a handful more. Output
for single words is constrained by a grammar to letters, apostrophes and hyphens.

## Building

Requirements: Android SDK with platform 36, NDK 28.2, CMake 3.31.6 (install both through
SDK Manager), JDK 17 or newer. The repo uses the Gradle wrapper.

```
git clone --recursive git@github.com:thingg-co/type.git
cd type
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Only arm64-v8a is built. The first build compiles llama.cpp and takes a few minutes.

To try it on a device or emulator:

```
adb shell ime enable com.aosmith.type/.ime.TypeInputMethodService
adb shell ime set com.aosmith.type/.ime.TypeInputMethodService
```

Then open Type, download a model (or import a `.gguf` you already have), and type in the
test field. For a debuggable build you can also push a model directly:

```
adb push model.gguf /data/local/tmp/model.gguf
adb shell run-as com.aosmith.type sh -c 'mkdir -p files/models && cp /data/local/tmp/model.gguf files/models/'
```

## Models

The settings screen offers SmolLM2 360M (fast), Llama 3.2 1B (balanced) and Qwen2.5 1.5B
(recommended; it fixed every typo and sentence in the eval, see tools/eval.py). Any
instruct-tuned GGUF that llama.cpp can run will load through the import button; the chat
template is read from the file.

## Languages

English only for now. Layouts live in `KeyboardLayouts.kt` as plain data and the dictionary is
an asset file, so a second language needs a layout, a word list and a subtype entry in
`res/xml/method.xml`. Thai is the planned next language.

## License

PolyForm Noncommercial 1.0.0: free for personal, hobby, research and other noncommercial use.
Commercial use needs a separate license. See LICENSE and NOTICE.
