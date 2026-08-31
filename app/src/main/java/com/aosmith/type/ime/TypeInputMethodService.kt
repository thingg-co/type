package com.aosmith.type.ime

import android.content.SharedPreferences
import android.inputmethodservice.InputMethodService
import android.text.InputType
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.widget.LinearLayout
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.aosmith.type.Prefs
import com.aosmith.type.R
import com.aosmith.type.dict.Bigrams
import com.aosmith.type.dict.Dictionary
import com.aosmith.type.dict.Lexer
import com.aosmith.type.dict.NeuralLm
import com.aosmith.type.dict.Personalizer
import com.aosmith.type.dict.MidWordAction
import com.aosmith.type.dict.TypingPolicy
import com.aosmith.type.llm.SpellLlm
import com.aosmith.type.model.ModelStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TypeInputMethodService : InputMethodService(), KeyboardView.Listener, SuggestionStripView.Listener {

    private lateinit var prefs: Prefs
    private lateinit var store: ModelStore
    private lateinit var llm: SpellLlm

    @Volatile private var dictionary: Dictionary? = null
    @Volatile private var bigrams: Bigrams? = null
    @Volatile private var neural: NeuralLm? = null
    @Volatile private var personalizer: Personalizer? = null
    private var noLearning = false

    private val mainScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private var keyboardView: KeyboardView? = null
    private var strip: SuggestionStripView? = null

    /** Letters of the word under construction (to the left of the cursor). */
    private val word = StringBuilder()
    private var correctionAllowed = true
    private var suggestionsAllowed = true
    private var liveJob: Job? = null
    private var sentenceJob: Job? = null
    private var pendingUndo: Undo? = null
    private var undoArmed = false
    private var suppressedTakeoverPrefix: String? = null
    private val sessionAccepted = HashSet<String>()

    private class Undo(val original: String, val replacement: String, val separator: String)

    private val prefListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == "model_id") maybeLoadModel()
        if (key == "learn_typing") neural?.personal = if (prefs.learnFromTyping) personalizer else null
        if (key == "personal_cleared") {
            personalizer?.clear()
            personalFile().delete()
        }
        applyPrefsToViews()
    }

    private fun personalFile() = java.io.File(filesDir, "personal.bin")

    override fun onCreate() {
        super.onCreate()
        prefs = Prefs(this)
        store = ModelStore(this)
        llm = SpellLlm { dictionary }
        prefs.registerListener(prefListener)
        mainScope.launch {
            dictionary = withContext(Dispatchers.IO) { Dictionary.load(this@TypeInputMethodService) }
            bigrams = withContext(Dispatchers.IO) {
                runCatching { Bigrams.load(this@TypeInputMethodService) }
                    .onFailure { Log.e(TAG, "bigram load failed", it) }
                    .getOrNull()
            }
            neural = withContext(Dispatchers.IO) {
                runCatching { loadNeural() }
                    .onFailure { Log.i(TAG, "next-word network unavailable: ${it.message}") }
                    .getOrNull()
            }
            neural?.let { n ->
                personalizer = withContext(Dispatchers.IO) {
                    Personalizer(n).also { p ->
                        runCatching {
                            personalFile().takeIf { it.exists() }?.inputStream()?.use(p::load)
                        }.onFailure {
                            Log.w(TAG, "personalization state discarded: ${it.message}")
                            p.clear()
                            personalFile().delete()
                        }
                    }
                }
                if (prefs.learnFromTyping) n.personal = personalizer
                Log.i(TAG, "personalization ready: ${personalizer?.lifetimeSamples ?: -1} lifetime samples")
            }
        }
        maybeLoadModel()
    }

    override fun onDestroy() {
        prefs.unregisterListener(prefListener)
        mainScope.cancel()
        llm.scope.launch { llm.unload() }
        super.onDestroy()
    }

    override fun onCreateInputView(): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(ContextCompat.getColor(context, R.color.kb_background))
        }
        val s = SuggestionStripView(this).apply { listener = this@TypeInputMethodService }
        val k = KeyboardView(this).apply { listener = this@TypeInputMethodService }
        container.addView(s, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, resources.getDimensionPixelSize(R.dimen.kb_strip_height)))
        container.addView(k, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        // With targetSdk 35+ the IME window is edge-to-edge, so the bottom row would sit under the
        // navigation bar unless we pad for it ourselves. Below 35 the inset arrives as zero.
        // The bottom uses the stable (ignoring-visibility) navigation inset too: devices with a
        // transient taskbar (Android 16) report zero while the bar floats over the app, which
        // left it covering the spacebar until it auto-hid.
        ViewCompat.setOnApplyWindowInsetsListener(container) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val stableNav = insets.getInsetsIgnoringVisibility(WindowInsetsCompat.Type.navigationBars())
            val gestures = insets.getInsets(WindowInsetsCompat.Type.mandatorySystemGestures())
            val tappable = insets.getInsetsIgnoringVisibility(WindowInsetsCompat.Type.tappableElement())
            // Transient taskbars are inconsistent about which inset carries their height
            // (some report it stable, some only as a gesture region), so take them all.
            val bottom = maxOf(bars.bottom, stableNav.bottom, gestures.bottom, tappable.bottom)
            if (com.aosmith.type.BuildConfig.DEBUG) {
                Log.d(TAG, "insets: bars=${bars.bottom} stableNav=${stableNav.bottom} gestures=${gestures.bottom} tappable=${tappable.bottom} -> $bottom")
            }
            v.updatePadding(left = bars.left, right = bars.right, bottom = bottom)
            insets
        }
        ViewCompat.requestApplyInsets(container)
        strip = s
        keyboardView = k
        applyPrefsToViews()
        return container
    }

    override fun onEvaluateFullscreenMode(): Boolean = false

    override fun onStartInputView(info: EditorInfo, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        val inputType = info.inputType
        val cls = inputType and InputType.TYPE_MASK_CLASS
        val variation = inputType and InputType.TYPE_MASK_VARIATION
        val isText = cls == InputType.TYPE_CLASS_TEXT
        val sensitive = variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
            variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
            variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD ||
            variation == InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS ||
            variation == InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS ||
            variation == InputType.TYPE_TEXT_VARIATION_URI
        // TYPE_TEXT_FLAG_NO_SUGGESTIONS is treated as advisory: Meta's apps stamp it on
        // ordinary chat fields (Instagram DMs declare 0xac001), which would silence the
        // keyboard exactly where people type most. Sensitive variations and the per-field
        // no-personalized-learning flag remain fully honored.
        noLearning = info.imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING != 0
        suggestionsAllowed = isText && !sensitive
        // A screen recording captures the keyboard as its own window even when the editor
        // blacks itself out; while a sensitive field is focused, keep this window out of
        // captures too.
        window?.window?.let { w ->
            if (sensitive) {
                w.addFlags(android.view.WindowManager.LayoutParams.FLAG_SECURE)
            } else {
                w.clearFlags(android.view.WindowManager.LayoutParams.FLAG_SECURE)
            }
        }
        correctionAllowed = suggestionsAllowed && prefs.autocorrect

        // Taskbar/nav state can change between shows without a new inset dispatch; ask again.
        (strip?.parent as? View)?.let { ViewCompat.requestApplyInsets(it) }
        keyboardView?.layer = if (cls == InputType.TYPE_CLASS_NUMBER || cls == InputType.TYPE_CLASS_PHONE) Layer.SYMBOLS else Layer.LETTERS
        strip?.fixEnabled = suggestionsAllowed
        strip?.clear()
        pendingUndo = null
        undoArmed = false
        suppressedTakeoverPrefix = null
        keyboardView?.wordTakeover = null
        cancelLive()
        syncWordFromEditor()
        updateShift()
        applyPrefsToViews()
        maybeLoadModel()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        personalizer?.let { p ->
            if (prefs.learnFromTyping && p.pendingSamples > 0) {
                mainScope.launch(Dispatchers.Default) {
                    val n = p.pendingSamples
                    val t0 = System.currentTimeMillis()
                    p.trainAndMaybeSave(personalFile())
                    Log.d(TAG, "personal train burst over $n samples (${System.currentTimeMillis() - t0} ms)")
                }
            }
        }
        cancelLive()
        sentenceJob?.cancel()
        word.clear()
        keyboardView?.keyWeights = null
        keyboardView?.wordTakeover = null
    }

    override fun onUpdateSelection(oldSelStart: Int, oldSelEnd: Int, newSelStart: Int, newSelEnd: Int, candidatesStart: Int, candidatesEnd: Int) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        if (newSelStart != newSelEnd) {
            if (word.isNotEmpty()) {
                word.clear()
                onWordChanged()
            }
            return
        }
        // Our own commits also land here; only resync when the editor disagrees with us.
        val trailing = trailingWordInEditor() ?: return
        if (trailing != word.toString()) {
            word.setLength(0)
            word.append(trailing)
            onWordChanged()
        }
    }

    // ---- key events ------------------------------------------------------------------

    override fun onText(text: String) {
        val ic = currentInputConnection ?: return
        undoArmed = false
        ic.commitText(text, 1)
        if (text.length == 1 && isWordChar(text[0])) {
            word.append(text)
            onWordChanged()
        } else {
            val finished = word.toString()
            word.clear()
            onWordBoundary(finished, text)
            // The tracker and editor already agree on the empty word, so onUpdateSelection
            // will not resync; evaluate the empty prefix here for next-word suggestions.
            onWordChanged()
        }
        updateShift()
    }

    override fun onSpace() = onText(" ")

    override fun onBackspace() {
        val ic = currentInputConnection ?: return
        if (undoArmed && pendingUndo != null) {
            revertCorrection()
            return
        }
        val selected = ic.getSelectedText(0)
        if (!selected.isNullOrEmpty()) {
            ic.commitText("", 1)
            word.clear()
        } else {
            ic.deleteSurroundingTextInCodePoints(1, 0)
            if (word.isNotEmpty()) word.setLength(word.length - 1)
        }
        onWordChanged()
        updateShift()
    }

    override fun onEnter() {
        if (currentInputConnection == null) return
        val finished = word.toString()
        word.clear()
        undoArmed = false
        // sendDefaultEditorAction applies the imeOptions rules (action set, no IME_FLAG_NO_ENTER_ACTION);
        // multi-line fields carry that flag, so they get a real newline instead.
        if (sendDefaultEditorAction(true)) {
            cancelLive()
            keyboardView?.keyWeights = null
            strip?.clear()
        } else {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
            onWordBoundary(finished, "\n")
        }
        updateShift()
    }

    // ---- word tracking -----------------------------------------------------------------

    private fun isWordChar(c: Char): Boolean = c.isLetter() || c == '\''

    private fun onWordChanged() {
        cancelLive()
        val dict = dictionary
        val current = word.toString()
        // No adaptive dimming in sensitive fields: the fading keys would broadcast the
        // structure of a password to anyone watching the screen.
        keyboardView?.keyWeights =
            if (current.isEmpty() || dict == null || !suggestionsAllowed) null else dict.nextLetters(current)
        if (current.isEmpty()) suppressedTakeoverPrefix = null
        if (!suggestionsAllowed || dict == null) {
            keyboardView?.wordTakeover = null
            if (pendingUndo == null || !undoArmed) strip?.clear()
            return
        }
        liveJob = mainScope.launch {
            val beforeText = textBeforeWord(current) ?: ""
            val prevWords = Lexer.previousWords(beforeText, 5)
            val action = withContext(Dispatchers.Default) { TypingPolicy.midWord(dict, bigrams, neural, prevWords, current) }
            if (com.aosmith.type.BuildConfig.DEBUG) {
                Log.d(TAG, "midWord '$current' prev=$prevWords neural=${neural != null} -> ${action.javaClass.simpleName}")
            }
            if (word.toString() != current) return@launch
            if (action !is MidWordAction.WordKeys) keyboardView?.wordTakeover = null
            when (action) {
                is MidWordAction.WordKeys -> {
                    val suppressed = suppressedTakeoverPrefix?.let { current.startsWith(it) } == true
                    val minAdded = action.words.minOf { it.length } - current.length
                    if (!suppressed && minAdded >= 3) {
                        // So few ways to finish, and enough typing saved, that keys are not
                        // needed: the words themselves become the keyboard.
                        keyboardView?.wordTakeover = action.words
                        strip?.clear()
                    } else {
                        keyboardView?.wordTakeover = null
                        strip?.showWordKeys(action.words)
                    }
                }
                is MidWordAction.Predictions ->
                    strip?.showSuggestions(action.words.map { SuggestionStripView.Suggestion(it) })
                is MidWordAction.NextWords ->
                    if (pendingUndo == null) strip?.showSuggestions(action.words.map { SuggestionStripView.Suggestion(it) })
                is MidWordAction.Typo -> {
                    val typed = SuggestionStripView.Suggestion(current, isTypedWord = true)
                    val dictSuggestions = action.dictSuggestions.map { SuggestionStripView.Suggestion(it) }
                    strip?.showSuggestions(listOf(typed) + dictSuggestions)
                    if (action.askModel && prefs.liveSuggestions && llm.isReady && current.lowercase() !in sessionAccepted) {
                        delay(LIVE_DEBOUNCE_MS)
                        val corrected = llm.correctWord(beforeText, current)
                        if (corrected != null && word.toString() == current) {
                            val rest = dictSuggestions.filter { !it.text.equals(corrected, ignoreCase = true) }.take(1)
                            strip?.showSuggestions(listOf(SuggestionStripView.Suggestion(corrected, fromModel = true), typed) + rest)
                        }
                    }
                }
                MidWordAction.None -> if (pendingUndo == null || !undoArmed) strip?.clear()
            }
        }
    }

    private fun cancelLive() {
        liveJob?.let {
            if (it.isActive) {
                it.cancel()
                // Only interrupt the native side for a request this job actually started;
                // a finished job must not cancel someone else's decode (e.g. a boundary fix).
                llm.cancelCurrent()
            }
        }
        liveJob = null
    }

    private fun onWordBoundary(finished: String, separator: String) {
        keyboardView?.keyWeights = null
        keyboardView?.wordTakeover = null
        suppressedTakeoverPrefix = null
        if (finished.isNotEmpty()) recordTyped(finished, separator)
        if (pendingUndo == null || !undoArmed) strip?.clear()
        if (!correctionAllowed || finished.isEmpty() || finished.length > 24) return
        if (!finished.all { it.isLetter() || it == '\'' }) return
        if (finished.lowercase() in sessionAccepted) return
        // Auto-apostrophe runs before the known-word gate: the word list knows "dont" as a
        // word, which is precisely why the normal path can never fix it.
        com.aosmith.type.dict.Contractions.fix(finished)?.let { fixed ->
            if (fixed != finished) applyCorrection(finished, separator, fixed)
            return
        }
        if (finished.length < 2) return
        val dict = dictionary ?: return
        if (dict.isKnown(finished)) return
        val before = textBeforeWord(finished, separator) ?: ""
        mainScope.launch {
            val corrected = if (llm.isReady) {
                llm.correctWord(before, finished)
            } else {
                dictionaryOnlyCorrection(dict, finished)
            }
            if (corrected != null && !corrected.equals(finished, ignoreCase = true)) {
                applyCorrection(finished, separator, corrected)
            }
        }
    }

    /**
     * Feeds one finished in-vocabulary word into personalization. Guarded by the setting,
     * the field type, and the editor's no-personalized-learning flag; only word ids are
     * stored, never text, and nothing is captured from sensitive fields.
     */
    private fun recordTyped(finished: String, separator: String) {
        if (!prefs.learnFromTyping || !suggestionsAllowed || noLearning) return
        val dict = dictionary ?: return
        val n = neural ?: return
        val p = personalizer ?: return
        val target = dict.idOf(finished)
        if (target < 0) return
        val before = textBeforeWord(finished, separator) ?: return
        val ctx = Lexer.previousWords(before, 5).map { dict.idOf(it).let { id -> if (id >= 0) id else n.unk } }
        p.record(ctx, target)
    }

    /** Without a model only very safe fixes are applied: one edit away from a common word. */
    private fun dictionaryOnlyCorrection(dict: Dictionary, typed: String): String? {
        if (typed.length < 4) return null
        val best = dict.suggest(typed, 1).firstOrNull() ?: return null
        if (Dictionary.editDistance(best.lowercase(), typed.lowercase()) != 1) return null
        if (dict.rankOf(best) > 15_000) return null
        return best
    }

    private fun applyCorrection(original: String, separator: String, corrected: String) {
        val ic = currentInputConnection ?: return
        val before = ic.getTextBeforeCursor(original.length + separator.length + 64, 0)?.toString() ?: return
        val idx = before.lastIndexOf(original + separator)
        if (idx < 0) return
        val typedSince = before.substring(idx + original.length + separator.length)
        if (typedSince.length > 40) return
        ic.beginBatchEdit()
        ic.deleteSurroundingText(before.length - idx, 0)
        ic.commitText(corrected + separator + typedSince, 1)
        ic.endBatchEdit()
        pendingUndo = Undo(original, corrected, separator)
        undoArmed = typedSince.isEmpty()
        strip?.showUndo(original)
        if (com.aosmith.type.BuildConfig.DEBUG) {
            Log.i(TAG, "corrected '$original' -> '$corrected' (${llm.lastLatencyMs} ms)")
        }
    }

    private fun revertCorrection() {
        val undo = pendingUndo ?: return
        val ic = currentInputConnection ?: return
        val needle = undo.replacement + undo.separator
        val before = ic.getTextBeforeCursor(needle.length + 64, 0)?.toString() ?: return
        val idx = before.lastIndexOf(needle)
        pendingUndo = null
        undoArmed = false
        strip?.clear()
        if (idx < 0) return
        val typedSince = before.substring(idx + needle.length)
        ic.beginBatchEdit()
        ic.deleteSurroundingText(before.length - idx, 0)
        ic.commitText(undo.original + undo.separator + typedSince, 1)
        ic.endBatchEdit()
        sessionAccepted += undo.original.lowercase()
        syncWordFromEditor()
    }

    // ---- suggestion strip --------------------------------------------------------------

    override fun onWordKey(word: String) = onSuggestionPicked(word, false)

    override fun onEscapeWordMode() {
        suppressedTakeoverPrefix = word.toString()
        keyboardView?.wordTakeover = null
        onWordChanged()
    }

    override fun onSuggestionPicked(text: String, isTypedWord: Boolean) {
        val ic = currentInputConnection ?: return
        cancelLive()
        val current = word.toString()
        if (isTypedWord) sessionAccepted += current.lowercase()
        ic.beginBatchEdit()
        if (current.isNotEmpty()) ic.deleteSurroundingText(current.length, 0)
        ic.commitText(Dictionary.matchCase(current.ifEmpty { text }, text) + " ", 1)
        ic.endBatchEdit()
        recordTyped(text, " ")
        word.clear()
        keyboardView?.keyWeights = null
        keyboardView?.wordTakeover = null
        suppressedTakeoverPrefix = null
        strip?.clear()
        updateShift()
    }

    override fun onUndo() = revertCorrection()

    override fun onFixSentence() {
        val ic = currentInputConnection ?: return
        if (!suggestionsAllowed) return
        if (!llm.isReady) {
            strip?.showStatus(if (llm.state is SpellLlm.State.Loading) "Model is still loading…" else "No model installed. Open Type to download one.")
            return
        }
        val before = ic.getTextBeforeCursor(400, 0)?.toString() ?: return
        val start = sentenceStart(before)
        val span = before.substring(start)          // what gets replaced, trailing whitespace included
        val segment = span.trimEnd()                // what the model sees
        val trailing = span.substring(segment.length)
        if (segment.isBlank()) return
        cancelLive()
        sentenceJob?.cancel()
        strip?.showStatus("…")
        sentenceJob = mainScope.launch {
            val corrected = llm.correctSentence(segment)
            val ic2 = currentInputConnection ?: return@launch
            if (corrected == null) {
                strip?.showStatus("Looks fine")
                delay(1200)
                strip?.clear()
                return@launch
            }
            val now = ic2.getTextBeforeCursor(400, 0)?.toString() ?: return@launch
            if (!now.endsWith(span)) return@launch
            ic2.beginBatchEdit()
            ic2.deleteSurroundingText(span.length, 0)
            ic2.commitText(corrected + trailing, 1)
            ic2.endBatchEdit()
            pendingUndo = Undo(segment, corrected, trailing)
            undoArmed = false
            strip?.showUndo("undo")
            syncWordFromEditor()
        }
    }

    /**
     * Index where the last sentence starts: after the last terminator, unless that leaves
     * fewer than three words, in which case the whole last line is used.
     */
    private fun sentenceStart(text: String): Int {
        val trimmed = text.trimEnd()
        if (trimmed.isEmpty()) return text.length
        val cut = trimmed.indexOfLast { it == '.' || it == '!' || it == '?' || it == '\n' }
        var start = cut + 1
        if (trimmed.substring(start).trim().split(Regex("\\s+")).size < 3) {
            start = trimmed.lastIndexOf('\n') + 1
        }
        while (start < trimmed.length && trimmed[start].isWhitespace()) start++
        return start
    }

    // ---- editor helpers ----------------------------------------------------------------

    private fun trailingWordInEditor(): String? {
        val ic = currentInputConnection ?: return null
        val after = ic.getTextAfterCursor(1, 0)
        if (after != null && after.isNotEmpty() && isWordChar(after[0])) return ""
        val before = ic.getTextBeforeCursor(32, 0) ?: return null
        var i = before.length
        while (i > 0 && isWordChar(before[i - 1])) i--
        return before.substring(i)
    }

    private fun syncWordFromEditor() {
        val trailing = trailingWordInEditor() ?: ""
        word.setLength(0)
        word.append(trailing)
        onWordChanged()
    }

    /** Text before the word in progress (or before `word + separator` once committed). */
    private fun textBeforeWord(current: String, separator: String = ""): String? {
        val ic = currentInputConnection ?: return null
        val before = ic.getTextBeforeCursor(160, 0)?.toString() ?: return null
        val tail = current + separator
        return if (before.endsWith(tail)) before.dropLast(tail.length) else before
    }

    private fun updateShift() {
        val kv = keyboardView ?: return
        if (kv.shift == KeyboardView.Shift.LOCKED || kv.layer != Layer.LETTERS) return
        val ic = currentInputConnection ?: return
        val inputType = currentInputEditorInfo?.inputType ?: return
        val caps = if (inputType and InputType.TYPE_TEXT_FLAG_CAP_SENTENCES != 0 ||
            inputType and InputType.TYPE_TEXT_FLAG_CAP_WORDS != 0 ||
            inputType and InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS != 0
        ) ic.getCursorCapsMode(inputType) != 0 else false
        kv.shift = if (caps) KeyboardView.Shift.ON else KeyboardView.Shift.OFF
    }

    private fun applyPrefsToViews() {
        keyboardView?.adaptiveEnabled = prefs.adaptiveKeys
        keyboardView?.hapticsEnabled = prefs.haptics
        correctionAllowed = suggestionsAllowed && prefs.autocorrect
    }

    private fun maybeLoadModel() {
        val file = store.activeFile(prefs) ?: return
        val threads = if (prefs.threads > 0) prefs.threads else defaultThreads()
        llm.scope.launch {
            val wasReady = llm.isReady
            val ok = llm.ensureLoaded(file, threads)
            if (ok && !wasReady) Log.i(TAG, "model ready: ${file.name}")
        }
    }

    /**
     * The network's dense arrays load from the asset; the native top-k needs a real file,
     * so the asset is copied into filesDir once. Absent asset (model not shipped) means
     * null, and everything falls back to bigrams.
     */
    private fun loadNeural(): NeuralLm {
        val f = java.io.File(filesDir, "en_nextword.bin")
        val assetSize = assets.open("en_nextword.bin").use { it.available().toLong() }
        if (!f.exists() || f.length() != assetSize) {
            assets.open("en_nextword.bin").use { src ->
                f.outputStream().use { dst -> src.copyTo(dst, 1 shl 20) }
            }
        }
        if (!com.aosmith.type.llm.LlamaNative.nnLoad(f.absolutePath)) {
            Log.w(TAG, "native next-word matvec unavailable; using the Kotlin fallback")
        }
        return NeuralLm.load(this)
    }

    private fun defaultThreads(): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        return cores.coerceIn(2, 4)
    }

    companion object {
        private const val TAG = "TypeIME"
        private const val LIVE_DEBOUNCE_MS = 350L
    }
}
