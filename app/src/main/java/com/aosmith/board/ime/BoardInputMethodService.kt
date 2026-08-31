package com.aosmith.board.ime

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
import com.aosmith.board.Prefs
import com.aosmith.board.R
import com.aosmith.board.dict.Dictionary
import com.aosmith.board.llm.SpellLlm
import com.aosmith.board.model.ModelStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BoardInputMethodService : InputMethodService(), KeyboardView.Listener, SuggestionStripView.Listener {

    private lateinit var prefs: Prefs
    private lateinit var store: ModelStore
    private lateinit var llm: SpellLlm

    @Volatile private var dictionary: Dictionary? = null

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
    private val sessionAccepted = HashSet<String>()

    private class Undo(val original: String, val replacement: String, val separator: String)

    private val prefListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == "model_id") maybeLoadModel()
        applyPrefsToViews()
    }

    override fun onCreate() {
        super.onCreate()
        prefs = Prefs(this)
        store = ModelStore(this)
        llm = SpellLlm { dictionary }
        prefs.registerListener(prefListener)
        mainScope.launch {
            dictionary = withContext(Dispatchers.IO) { Dictionary.load(this@BoardInputMethodService) }
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
        val s = SuggestionStripView(this).apply { listener = this@BoardInputMethodService }
        val k = KeyboardView(this).apply { listener = this@BoardInputMethodService }
        container.addView(s, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, resources.getDimensionPixelSize(R.dimen.kb_strip_height)))
        container.addView(k, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        // With targetSdk 35+ the IME window is edge-to-edge, so the bottom row would sit under the
        // navigation bar unless we pad for it ourselves. Below 35 the inset arrives as zero.
        ViewCompat.setOnApplyWindowInsetsListener(container) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.updatePadding(left = bars.left, right = bars.right, bottom = bars.bottom)
            insets
        }
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
        val noSuggestions = inputType and InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS != 0
        suggestionsAllowed = isText && !sensitive && !noSuggestions
        correctionAllowed = suggestionsAllowed && prefs.autocorrect

        keyboardView?.layer = if (cls == InputType.TYPE_CLASS_NUMBER || cls == InputType.TYPE_CLASS_PHONE) Layer.SYMBOLS else Layer.LETTERS
        strip?.fixEnabled = suggestionsAllowed
        strip?.clear()
        pendingUndo = null
        undoArmed = false
        cancelLive()
        syncWordFromEditor()
        updateShift()
        applyPrefsToViews()
        maybeLoadModel()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        cancelLive()
        sentenceJob?.cancel()
        word.clear()
        keyboardView?.keyWeights = null
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
        keyboardView?.keyWeights = if (current.isEmpty() || dict == null) null else dict.nextLetters(current)
        if (!suggestionsAllowed || current.isEmpty() || dict == null) {
            if (pendingUndo == null || !undoArmed) strip?.clear()
            return
        }
        if (current.length < 2 || dict.isKnown(current) || dict.hasPrefix(current) && current.length < 4) {
            strip?.clear()
            return
        }
        val typed = SuggestionStripView.Suggestion(current, isTypedWord = true)
        val wantModel = prefs.liveSuggestions && llm.isReady && current.length >= 3 && current.lowercase() !in sessionAccepted
        liveJob = mainScope.launch {
            // Dictionary candidates first (a few ms on a background thread), then the model.
            val dictSuggestions = withContext(Dispatchers.Default) { dict.suggest(current, 2) }
                .map { SuggestionStripView.Suggestion(it) }
            if (word.toString() != current) return@launch
            strip?.showSuggestions(listOf(typed) + dictSuggestions)
            if (!wantModel) return@launch
            delay(LIVE_DEBOUNCE_MS)
            val before = textBeforeWord(current) ?: ""
            val corrected = llm.correctWord(before, current)
            if (corrected != null && word.toString() == current) {
                val rest = dictSuggestions.filter { !it.text.equals(corrected, ignoreCase = true) }.take(1)
                strip?.showSuggestions(listOf(SuggestionStripView.Suggestion(corrected, fromModel = true), typed) + rest)
            }
        }
    }

    private fun cancelLive() {
        liveJob?.let {
            it.cancel()
            llm.cancelCurrent()
        }
        liveJob = null
    }

    private fun onWordBoundary(finished: String, separator: String) {
        keyboardView?.keyWeights = null
        if (pendingUndo == null || !undoArmed) strip?.clear()
        if (!correctionAllowed || finished.length < 2 || finished.length > 24) return
        if (!finished.all { it.isLetter() || it == '\'' }) return
        val dict = dictionary ?: return
        if (dict.isKnown(finished) || finished.lowercase() in sessionAccepted) return
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
        Log.i(TAG, "corrected '$original' -> '$corrected' (${llm.lastLatencyMs} ms)")
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

    override fun onSuggestionPicked(text: String, isTypedWord: Boolean) {
        val ic = currentInputConnection ?: return
        cancelLive()
        val current = word.toString()
        if (isTypedWord) sessionAccepted += current.lowercase()
        ic.beginBatchEdit()
        if (current.isNotEmpty()) ic.deleteSurroundingText(current.length, 0)
        ic.commitText(Dictionary.matchCase(current.ifEmpty { text }, text) + " ", 1)
        ic.endBatchEdit()
        word.clear()
        keyboardView?.keyWeights = null
        strip?.clear()
        updateShift()
    }

    override fun onUndo() = revertCorrection()

    override fun onFixSentence() {
        val ic = currentInputConnection ?: return
        if (!suggestionsAllowed) return
        if (!llm.isReady) {
            strip?.showStatus(if (llm.state is SpellLlm.State.Loading) "Model is still loading…" else "No model installed. Open Board to download one.")
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

    private fun defaultThreads(): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        return cores.coerceIn(2, 4)
    }

    companion object {
        private const val TAG = "BoardIME"
        private const val LIVE_DEBOUNCE_MS = 350L
    }
}
