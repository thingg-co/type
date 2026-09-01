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
import com.aosmith.type.dict.Casing
import com.aosmith.type.dict.Dictionary
import com.aosmith.type.dict.Lexer
import com.aosmith.type.dict.NeuralLm
import com.aosmith.type.dict.Personalizer
import com.aosmith.type.dict.SlipProfile
import com.aosmith.type.dict.KeyNeighbors
import com.aosmith.type.dict.TypoTable
import com.aosmith.type.dict.Confusables
import com.aosmith.type.dict.Contractions
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

class TypeInputMethodService : InputMethodService(), KeyboardView.Listener, SuggestionStripView.Listener, EmojiPanelView.Listener {

    private lateinit var prefs: Prefs
    private lateinit var store: ModelStore
    private lateinit var llm: SpellLlm

    @Volatile private var dictionary: Dictionary? = null
    @Volatile private var emojiWords: EmojiSuggestions? = null
    @Volatile private var bigrams: Bigrams? = null
    @Volatile private var neural: NeuralLm? = null
    @Volatile private var personalizer: Personalizer? = null
    @Volatile private var slipProfile: SlipProfile? = null
    @Volatile private var slipDirty = false
    private var noLearning = false

    private val mainScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private var keyboardView: KeyboardView? = null
    private var strip: SuggestionStripView? = null
    private var emojiPanel: EmojiPanelView? = null

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
    private var lastSpaceAt = 0L

    private class Undo(val original: String, val replacement: String, val separator: String)

    private val prefListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (key == "model_id") maybeLoadModel()
        if (key == "learn_typing") {
            neural?.personal = if (prefs.learnFromTyping) personalizer else null
            KeyNeighbors.personal = if (prefs.learnFromTyping) slipProfile else null
        }
        if (key == "personal_cleared") {
            personalizer?.clear()
            personalFile().delete()
            slipProfile?.clear()
            slipFile().delete()
        }
        applyPrefsToViews()
    }

    private fun personalFile() = java.io.File(filesDir, "personal.bin")

    private fun slipFile() = java.io.File(filesDir, "slips.bin")

    /** True when this field and the settings allow learning from what the user does. */
    private fun learningAllowed(): Boolean = prefs.learnFromTyping && suggestionsAllowed && !noLearning

    override fun onCreate() {
        super.onCreate()
        lastOrientation = resources.configuration.orientation
        prefs = Prefs(this)
        store = ModelStore(this)
        llm = SpellLlm { dictionary }
        prefs.registerListener(prefListener)
        mainScope.launch {
            dictionary = withContext(Dispatchers.IO) {
                Dictionary.load(this@TypeInputMethodService).also { d ->
                    d.misspellings = runCatching { TypoTable.load(this@TypeInputMethodService) }
                        .onFailure { Log.e(TAG, "typo table load failed", it) }
                        .getOrNull()
                    d.casing = runCatching { Casing.load(this@TypeInputMethodService) }
                        .onFailure { Log.e(TAG, "casing table load failed", it) }
                        .getOrNull()
                }
            }
            emojiWords = withContext(Dispatchers.IO) {
                runCatching { EmojiSuggestions.load(this@TypeInputMethodService) }
                    .onFailure { Log.e(TAG, "emoji suggestions load failed", it) }
                    .getOrNull()
            }
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
            slipProfile = withContext(Dispatchers.IO) {
                SlipProfile().also { p ->
                    runCatching { slipFile().takeIf { it.exists() }?.inputStream()?.use(p::load) }
                        .onFailure {
                            Log.w(TAG, "slip profile discarded: ${it.message}")
                            p.clear()
                            slipFile().delete()
                        }
                }
            }
            KeyNeighbors.personal = if (prefs.learnFromTyping) slipProfile else null
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
        val e = EmojiPanelView(this).apply {
            listener = this@TypeInputMethodService
            categories = runCatching { EmojiData.load(this@TypeInputMethodService) }
                .onFailure { Log.e(TAG, "emoji data load failed", it) }
                .getOrDefault(emptyList())
            visibility = View.GONE
        }
        container.addView(s, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, resources.getDimensionPixelSize(R.dimen.kb_strip_height)))
        container.addView(k, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        container.addView(e, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
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
        emojiPanel = e
        applyPrefsToViews()
        return container
    }

    override fun onEvaluateFullscreenMode(): Boolean = false

    private var lastOrientation = android.content.res.Configuration.ORIENTATION_UNDEFINED

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        val wasShown = isInputViewShown
        val rotated = newConfig.orientation != lastOrientation
        lastOrientation = newConfig.orientation
        if (com.aosmith.type.BuildConfig.DEBUG) {
            Log.d(TAG, "onConfigurationChanged orientation=${newConfig.orientation} wasShown=$wasShown rotated=$rotated")
        }
        super.onConfigurationChanged(newConfig)
        // Hide instead of riding the rotation out. A keyboard that stays shown through
        // reorientation latches the host app's layout on the T807D — the app relayouts
        // as if no keyboard exists while the keyboard keeps painting over it, and every
        // keyboard tested does it (Type, Gboard, FUTO), so it is not avoidable from in
        // here. Hidden, the app's full-height layout is simply correct, and the next
        // tap on the field is a fresh show into settled geometry: a clean insets
        // negotiation every time. One extra tap after rotating is a better deal than
        // typing into an input pinned off-screen. No auto-reshow — racing the rotation
        // animation with a show is how the 0.5.7 bounce re-latched what it had cured.
        // The one automatic intervention that survives review, because it cannot fail
        // into anything: after the rotation settles, grow the window by one pixel for
        // one frame and shrink it back. An app goes stale by missing the mid-rotation
        // insets dispatch and then never hearing another (steady state = no dispatch);
        // a frame change is the trigger the system always delivers, so the nudge hands
        // the app two fresh dispatches with true geometry. No show/hide state machine,
        // no window rebuild — the previous attempts down those roads each stranded the
        // keyboard or corrupted layouts worse than the latch (see git history around
        // 0.6.2-0.6.6). Worst case here is an invisible 1 px resize that changes
        // nothing. Manual recovery for anything residual: long-press ✨.
        if (rotated && wasShown) {
            nudgeJob?.cancel()
            nudgeJob = mainScope.launch {
                // A ramp instead of one fixed wait: the app's own post-rotation
                // relayout lands anywhere from ~300 to ~900 ms, and a nudge is free
                // (invisible 1 px, no state-machine contact), so fire early for fast
                // settles and again after slow ones rather than guessing a single
                // moment. Cumulative: 350 ms, 750 ms, 1250 ms.
                for (wait in longArrayOf(350, 400, 500)) {
                    delay(wait)
                    val root = strip?.parent as? View ?: return@launch
                    if (!isInputViewShown) return@launch
                    if (com.aosmith.type.BuildConfig.DEBUG) Log.d(TAG, "post-rotation insets nudge")
                    // The root's bottom padding carries the navigation-bar inset:
                    // nudge relative to it and restore it exactly, never overwrite it
                    // (0.6.8 zeroed it and dropped the keyboard under the taskbar).
                    val l = root.paddingLeft
                    val t = root.paddingTop
                    val r = root.paddingRight
                    val b = root.paddingBottom
                    root.setPadding(l, t, r, b + 1)
                    try {
                        delay(50) // one relayout and insets dispatch at the nudged frame
                    } finally {
                        root.setPadding(l, t, r, b)
                        // If a real insets dispatch landed mid-nudge, the captured
                        // padding is stale: let the listener reassert the live value.
                        androidx.core.view.ViewCompat.requestApplyInsets(root)
                    }
                }
            }
        }
    }

    private var nudgeJob: kotlinx.coroutines.Job? = null

    // ---- window lifecycle instrumentation (debug builds only) ------------------------
    // Chasing the stuck-surface bug: system marks the IME hidden while the window stays
    // painted (insets-animation leash abandoned). These logs turn one real occurrence
    // into a timeline: what we were asked, when, and what insets we reported.

    private var lastLoggedInsets = ""

    override fun onWindowShown() {
        super.onWindowShown()
        if (com.aosmith.type.BuildConfig.DEBUG) Log.d(TAG, "onWindowShown")
    }

    /**
     * Recovery for the OEM stuck-surface state, on the user's explicit long-press of ✨.
     *
     * The failure (T807D/TCL, Signal, Gboard reproduces it): WindowManager holds the IME
     * window with mHasSurface=true but isReadyForDisplay()=false and the given insets
     * dropped to zero, so the app is told there is no keyboard while the stale surface
     * keeps painting over it. That state survives the framework hide/show path
     * (requestHideSelf/requestShowSelf reuse the same window), which is why the
     * automatic bounce shims failed — one even re-latched the app it had just cured.
     * Dismissing the dialog removes the window from WM entirely; the following show
     * re-adds it as a brand-new window with fresh readiness and insets registration.
     */
    override fun onUnstick() {
        if (com.aosmith.type.BuildConfig.DEBUG) Log.d(TAG, "onUnstick: rebuilding IME window")
        mainScope.launch { rebuildImeWindow() }
    }

    private suspend fun rebuildImeWindow() {
        val w = window ?: return
        try {
            w.dismiss()
        } catch (e: Exception) {
            Log.w(TAG, "rebuild dismiss failed", e)
        }
        delay(250) // let WM finish removing the old window
        try {
            w.show()
        } catch (e: Exception) {
            Log.w(TAG, "rebuild show failed", e)
        }
        requestShowSelf(0) // resync the framework's show state
    }

    override fun onWindowHidden() {
        super.onWindowHidden()
        if (com.aosmith.type.BuildConfig.DEBUG) Log.d(TAG, "onWindowHidden")
    }

    /**
     * Stock framework computation, untouched. Both directions of cleverness here are
     * proven regressions: a remembered-value guard self-poisoned (0.5.5–0.5.10), and
     * the full-height-window explicit reporting emitted a pre-layout transient that
     * apps latched as "keyboard covers the screen" (0.6.0–0.6.6, the white void). The
     * keyboard-sized default window plus default insets is the configuration that
     * always typed reliably; the rotate-under-keyboard latch it shares with Gboard and
     * FUTO on the T807D is recovered manually (✨ long-press).
     */
    override fun onComputeInsets(outInsets: Insets) {
        super.onComputeInsets(outInsets)
        val root = strip?.parent as? View
        if (com.aosmith.type.BuildConfig.DEBUG) {
            val s = "content=${outInsets.contentTopInsets} visible=${outInsets.visibleTopInsets} " +
                "touchable=${outInsets.touchableInsets} rootH=${root?.height ?: -1} shown=$isInputViewShown"
            if (s != lastLoggedInsets) {
                lastLoggedInsets = s
                Log.d(TAG, "onComputeInsets $s")
            }
        }
    }

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
        showEmojiPanel(false)
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
        if (slipDirty) {
            slipDirty = false
            slipProfile?.let { p ->
                mainScope.launch(Dispatchers.IO) {
                    runCatching { slipFile().outputStream().use(p::save) }
                        .onFailure { Log.w(TAG, "slip profile save failed: ${it.message}") }
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

    override fun onSpace() {
        val now = android.os.SystemClock.uptimeMillis()
        if (now - lastSpaceAt < DOUBLE_SPACE_MS && canDoubleSpacePeriod()) {
            lastSpaceAt = 0
            val ic = currentInputConnection ?: return
            ic.beginBatchEdit()
            ic.deleteSurroundingText(1, 0)
            ic.commitText(". ", 1)
            ic.endBatchEdit()
            undoArmed = false
            word.clear()
            onWordChanged()
            updateShift() // sentence caps: the editor's caps mode now asks for shift
            return
        }
        lastSpaceAt = now
        onText(" ")
    }

    /**
     * Double-tapped space becomes ". " only after something sentence-like: a word, digit,
     * emoji or closing quote/bracket followed by the first space. Never in fields where
     * suggestions are off (passwords, URLs, email addresses).
     */
    private fun canDoubleSpacePeriod(): Boolean {
        if (!suggestionsAllowed) return false
        val ic = currentInputConnection ?: return false
        val t = ic.getTextBeforeCursor(3, 0) ?: return false
        if (t.length < 2 || t[t.length - 1] != ' ') return false
        val prev = t[t.length - 2]
        return prev.isLetterOrDigit() || prev == '\'' || prev == '"' ||
            prev == ')' || prev == ']' || Lexer.isEmojiChar(prev)
    }

    // ---- emoji panel -----------------------------------------------------------------

    override fun onEmoji() = showEmojiPanel(true)

    override fun onBackToLetters() = showEmojiPanel(false)

    override fun onEmojiPicked(emoji: String) {
        onText(emoji)
        prefs.emojiRecents = listOf(emoji) + prefs.emojiRecents.filterNot { it == emoji }
        emojiPanel?.recents = prefs.emojiRecents
    }

    private fun showEmojiPanel(show: Boolean) {
        val panel = emojiPanel ?: return
        if (show) {
            panel.recents = prefs.emojiRecents
            cancelLive()
            keyboardView?.wordTakeover = null
            keyboardView?.keyWeights = null
        }
        panel.visibility = if (show) View.VISIBLE else View.GONE
        keyboardView?.visibility = if (show) View.GONE else View.VISIBLE
        if (!show) onWordChanged()
    }

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
        } else if (word.isNotEmpty()) {
            ic.deleteSurroundingTextInCodePoints(1, 0)
            word.setLength(word.length - 1)
        } else {
            // Outside a word the last thing may be an emoji: delete the whole grapheme
            // cluster (flag pair, keycap, ZWJ family), not one code point of it.
            val before = ic.getTextBeforeCursor(24, 0)?.toString()
            if (before.isNullOrEmpty()) {
                ic.deleteSurroundingTextInCodePoints(1, 0)
            } else {
                val it = android.icu.text.BreakIterator.getCharacterInstance()
                it.setText(before)
                val start = it.preceding(before.length)
                val n = if (start == android.icu.text.BreakIterator.DONE) 1 else before.length - start
                ic.deleteSurroundingText(n, 0)
            }
        }
        onWordChanged()
        updateShift()
    }

    override fun onEnter() {
        if (currentInputConnection == null) return
        val finished = word.toString()
        word.clear()
        undoArmed = false
        // Enter may send the message, and a sent message is beyond correction: the
        // synchronous net-only pass runs first. In multi-line fields the later \n
        // boundary re-checks are no-ops (the corrected text no longer matches).
        preSendFix(finished)
        // sendDefaultEditorAction applies the imeOptions rules (action set, no IME_FLAG_NO_ENTER_ACTION);
        // multi-line fields carry that flag, so they get a real newline instead.
        if (sendDefaultEditorAction(true)) {
            cancelLive()
            keyboardView?.keyWeights = null
            strip?.clear()
            // Deliberately no InputConnection round-trips past this point: the app is
            // busy dispatching the send and may be about to hide this window, and a
            // main thread blocked in a synchronous IC call right then is how a hide
            // animation gets abandoned mid-flight, leaving the keyboard painted over
            // the expanded app (seen on T807D/Signal). The field restarts anyway,
            // and onStartInputView recomputes shift.
        } else {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
            onWordBoundary(finished, "\n")
            updateShift()
        }
    }

    /**
     * The last look before an editor action that may send: fixes a confusable in the
     * final one or two words, synchronously and from the net alone (no model call, no
     * undo once sent). With a TNW2 asset the scores include the end-of-message term,
     * which is what finally separates "Your welcome" sent alone from "your welcome"
     * mid-phrase.
     */
    private fun preSendFix(wordAtCursor: String) {
        if (!correctionAllowed) return
        val dict = dictionary ?: return
        if (neural == null) return
        var w = wordAtCursor
        var sep = ""
        if (w.isEmpty()) {
            // Enter right after a space: the last committed word is the one to check.
            val before = currentInputConnection?.getTextBeforeCursor(96, 0)?.toString() ?: return
            var i = before.length
            while (i > 0 && before[i - 1] == ' ') i--
            val end = i
            while (i > 0 && isWordChar(before[i - 1])) i--
            if (i == end) return
            w = before.substring(i, end)
            sep = before.substring(end)
        }
        if (w.length > 24 || !w.all { it.isLetter() || it == '\'' }) return
        if (w.lowercase() in sessionAccepted) return
        if (maybeFixConfusablePrevious(dict, w, sep, endOfMessage = true)) return
        Confusables.alternativesOf(w).takeIf { it.isNotEmpty() }?.let { alts ->
            maybeFixConfusable(dict, w, sep, alts, endOfMessage = true)
        }
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
            // Emoji chips outrank the usual strip content: "lol" and a trailing "555" run
            // offer their emoji first, with the best word suggestion keeping the last slot.
            val emojiSugs = emojiWords?.let {
                if (current.isNotEmpty()) it.forWord(current) else it.forTextTail(beforeText)
            } ?: emptyList()
            if (emojiSugs.isNotEmpty() && (pendingUndo == null || !undoArmed)) {
                keyboardView?.wordTakeover = null
                val words = when (action) {
                    is MidWordAction.WordKeys -> action.words
                    is MidWordAction.Predictions -> action.words
                    is MidWordAction.NextWords -> action.words
                    is MidWordAction.Typo -> action.dictSuggestions
                    MidWordAction.None -> emptyList()
                }
                val chips = emojiSugs.take(2) + words.take(1)
                strip?.showSuggestions(chips.map { SuggestionStripView.Suggestion(it) })
                return@launch
            }
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
        // Laugh variants ("hahah", "loool") are interjections, not typos; leave them be.
        if (EmojiSuggestions.normalizeLaugh(finished.lowercase()) != finished.lowercase()) return
        // Known-fixes layer runs before the known-word gate: the word list knows "dont" and
        // "belive" as words, which is precisely why the normal path can never fix them.
        Contractions.fix(finished)?.let { fixed ->
            if (fixed != finished) applyCorrection(finished, separator, fixed)
            return
        }
        val dict = dictionary ?: return
        // Canonical casing: "september" -> "September" (en_caps.txt, corpus-mined).
        // Correctly spelled words get their capital here; misspelled ones are cased
        // right after their spelling fix below.
        dict.casing?.canonical(finished)?.let { fixed ->
            applyCorrection(finished, separator, fixed)
            return
        }
        dict.misspellings?.fix(finished)?.let { fixed ->
            if (!fixed.equals(finished, ignoreCase = true)) {
                applyCorrection(finished, separator, dict.casing?.canonical(fixed) ?: fixed)
            }
            return
        }
        if (finished.length < 2) return
        // Confusable words ("then"/"than", "there"/"their"/"they're", bare contractions
        // like "were"/"we're"): the prediction network decides from context; the table in
        // Confusables only says which words are worth scoring. The lookback pass runs
        // first because the word AFTER a confusable carries most of the signal ("your
        // welcome"); the forward pass covers the word just finished from its left context.
        if (maybeFixConfusablePrevious(dict, finished, separator)) return
        Confusables.alternativesOf(finished).takeIf { it.isNotEmpty() }?.let { alts ->
            maybeFixConfusable(dict, finished, separator, alts)
            return
        }
        if (dict.isKnown(finished)) {
            maybeCorrectRealWordSlip(dict, finished, separator)
            return
        }
        val before = textBeforeWord(finished, separator) ?: ""
        // A capitalized unknown word mid-sentence is almost always a name ("meet Alexei
        // at"): the model's guesses there mangle exactly the words it cannot know. At a
        // sentence start capitalization carries no signal and correction stays on.
        if (finished[0].isUpperCase() && finished.drop(1).any { it.isLowerCase() } &&
            Lexer.previousWord(before) != null
        ) return
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
     * Forward confusable pass, at the word's own boundary: score the typed word against
     * its alternatives given the words before it. Margins are calibrated on natural text
     * (ConfusableCalibrationHarness + tools/confusables_calibrate.py): above DIRECT the
     * net's pick is applied with undo; the gray zone asks the language model; anything
     * weaker is left alone.
     */
    private fun maybeFixConfusable(
        dict: Dictionary,
        finished: String,
        separator: String,
        alts: List<String>,
        endOfMessage: Boolean = false,
    ) {
        val n = neural ?: return
        val beforeText = textBeforeWord(finished, separator) ?: return
        val ctx = Lexer.previousWords(beforeText, 5).map { w -> dict.idOf(w).let { if (it >= 0) it else n.unk } }
        val typedId = dict.idOf(finished)
        val altIds = alts.map(dict::idOf)
        if (typedId < 0 || altIds.any { it < 0 }) return
        if (endOfMessage && n.eosTrained) {
            // At a send there is right context after all: the message ends. Score
            // P(word | ctx) + P(end | ctx, word); two log terms, lookback-scale margin.
            fun total(v: Int): Float =
                n.scoreCandidates(ctx, intArrayOf(v))[0] + n.scoreCandidates(ctx + v, intArrayOf(n.bos))[0]
            val typedTotal = total(typedId)
            var best = -1
            var bestTotal = Float.NEGATIVE_INFINITY
            for (j in altIds.indices) {
                val t = total(altIds[j])
                if (t > bestTotal) {
                    bestTotal = t
                    best = j
                }
            }
            if (bestTotal - typedTotal > Confusables.SEND_FORWARD_MARGIN) {
                applyCorrection(finished, separator, Contractions.applyCase(finished, alts[best]))
            }
            return
        }
        if (ctx.isEmpty()) return
        val scores = n.scoreCandidates(ctx, intArrayOf(typedId) + altIds.toIntArray())
        val best = (1 until scores.size).maxBy { scores[it] }
        val margin = scores[best] - scores[0]
        when {
            margin > Confusables.DIRECT_MARGIN ->
                applyCorrection(finished, separator, Contractions.applyCase(finished, alts[best - 1]))
            // A send cannot wait on the model; anything below DIRECT stays as typed.
            endOfMessage -> {}
            margin > Confusables.MODEL_MARGIN && llm.isReady -> mainScope.launch {
                val corrected = llm.correctWord(beforeText, finished)
                if (corrected != null && !corrected.equals(finished, ignoreCase = true)) {
                    applyCorrection(finished, separator, corrected)
                }
            }
        }
    }

    /**
     * Lookback confusable pass: once the word after a confusable is known, the network
     * scores P(variant | ctx) + P(current | ctx, variant) for each variant — "your
     * welcome" is undecidable at "your" but obvious at "welcome". This is where most of
     * the calibrated catch rate lives. Returns true when it rewrote the previous word.
     */
    private fun maybeFixConfusablePrevious(
        dict: Dictionary,
        finished: String,
        separator: String,
        endOfMessage: Boolean = false,
    ): Boolean {
        val n = neural ?: return false
        val beforeText = textBeforeWord(finished, separator) ?: return false
        // The previous word with its original casing, only across plain spaces: any
        // punctuation between the words means a sentence boundary the net must not cross.
        var i = beforeText.length
        while (i > 0 && beforeText[i - 1] == ' ') i--
        val end = i
        while (i > 0 && isWordChar(beforeText[i - 1])) i--
        if (end == beforeText.length || end == i) return false
        val prevCased = beforeText.substring(i, end)
        val prevLower = prevCased.lowercase()
        val alts = Confusables.alternativesOf(prevLower)
        if (alts.isEmpty()) return false
        if (prevLower in sessionAccepted) return false
        // Never flip a word this pass or another just wrote; the undo chip owns it now.
        if (pendingUndo?.replacement?.equals(prevCased, ignoreCase = true) == true) return false
        val prevId = dict.idOf(prevLower)
        val curId = dict.idOf(finished)
        val altIds = alts.map(dict::idOf)
        if (prevId < 0 || curId < 0 || altIds.any { it < 0 }) return false
        val ctx2 = Lexer.previousWords(beforeText.substring(0, i), 5)
            .map { w -> dict.idOf(w).let { id -> if (id >= 0) id else n.unk } }
        val eos = endOfMessage && n.eosTrained
        fun total(v: Int): Float {
            var t = n.scoreCandidates(ctx2, intArrayOf(v))[0] + n.scoreCandidates(ctx2 + v, intArrayOf(curId))[0]
            // At a send the sentence also ends right after the current word.
            if (eos) t += n.scoreCandidates(ctx2 + v + curId, intArrayOf(n.bos))[0]
            return t
        }
        val typedTotal = total(prevId)
        var best = -1
        var bestTotal = Float.NEGATIVE_INFINITY
        for (j in altIds.indices) {
            val t = total(altIds[j])
            if (t > bestTotal) {
                bestTotal = t
                best = j
            }
        }
        val bar = if (eos) {
            Confusables.sendLookbackMargin(prevLower, alts[best])
        } else {
            Confusables.lookbackMargin(prevLower, alts[best])
        }
        if (bestTotal - typedTotal <= bar) return false
        val sep = beforeText.substring(end)
        applyCorrection(prevCased, sep, Contractions.applyCase(prevCased, alts[best]))
        return true
    }

    /**
     * A finished word can pass the known-word gate and still be a fat-finger slip: "nake",
     * "mot" and "cam" are all in the 64k list. The dictionary flags rare words with a far
     * more common neighbour-substitution variant, the prediction net screens by context
     * (same shape as the ambiguous-contraction path), and only then does the language model
     * decide. Its output still goes through CorrectionFilter and arrives with undo, so the
     * weakened dictionary gate is backed by three stronger steps behind it.
     */
    private fun maybeCorrectRealWordSlip(dict: Dictionary, finished: String, separator: String) {
        val n = neural ?: return
        if (!llm.isReady) return
        val candidates = dict.slipCandidates(finished)
        if (candidates.isEmpty()) return
        val beforeText = textBeforeWord(finished, separator) ?: return
        val ctx = Lexer.previousWords(beforeText, 5).map { w -> dict.idOf(w).let { if (it >= 0) it else n.unk } }
        if (ctx.isEmpty()) return
        val typedId = dict.idOf(finished)
        val bestId = dict.idOf(candidates.first())
        if (typedId < 0 || bestId < 0) return
        val scores = n.scoreCandidates(ctx, intArrayOf(typedId, bestId))
        if (scores[1] - scores[0] <= SLIP_MARGIN) return
        mainScope.launch {
            val corrected = llm.correctWord(beforeText, finished)
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
        // Feedback into personalization: the boundary already recorded the mistyped word,
        // so the corrected one goes in at double weight, and the slip profile learns the
        // key-level substitution. An undo takes both back out (see revertCorrection).
        if (learningAllowed()) {
            slipProfile?.let {
                it.recordCorrection(original, corrected)
                slipDirty = true
            }
            val dict = dictionary
            val n = neural
            val p = personalizer
            if (dict != null && n != null && p != null) {
                val target = dict.idOf(corrected)
                if (target >= 0) {
                    val ctx = Lexer.previousWords(before.substring(0, idx), 5)
                        .map { w -> dict.idOf(w).let { id -> if (id >= 0) id else n.unk } }
                    p.record(ctx, target, copies = 2)
                }
            }
        }
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
        // The strongest signal a user ever sends: they saw the correction and rejected
        // it. The slip profile forgets the substitution at triple weight, and the net's
        // personal deltas learn that this word is what they mean in this context.
        if (learningAllowed()) {
            slipProfile?.let {
                it.recordRevert(undo.original, undo.replacement)
                slipDirty = true
            }
            val dict = dictionary
            val n = neural
            val p = personalizer
            if (dict != null && n != null && p != null) {
                val target = dict.idOf(undo.original)
                if (target >= 0) {
                    val ctx = Lexer.previousWords(before.substring(0, idx), 5)
                        .map { w -> dict.idOf(w).let { id -> if (id >= 0) id else n.unk } }
                    p.record(ctx, target, copies = 3)
                }
            }
        }
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
        val committed = Dictionary.matchCase(current.ifEmpty { text }, text)
        ic.commitText((dictionary?.casing?.canonical(committed) ?: committed) + " ", 1)
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
        var start = sentenceStart(before)
        // Corrections ignore emoji: the model never sees one (it would drop or mangle it),
        // so the segment begins after the last emoji if there is one.
        val lastEmoji = before.indexOfLast { Lexer.isEmojiChar(it) }
        if (lastEmoji >= start) start = lastEmoji + 1
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
        emojiPanel?.hapticsEnabled = prefs.haptics
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
        } else if (prefs.benchFingerprint != android.os.Build.FINGERPRINT) {
            // First run on this device or OS build: measure the one cost that scales with
            // model size, so the app can right-size models to the hardware.
            val ms = com.aosmith.type.llm.LlamaNative.nnBenchMs(16)
            if (ms > 0f) {
                prefs.deviceMatvecMs = ms
                prefs.benchFingerprint = android.os.Build.FINGERPRINT
                Log.i(TAG, "device benchmark: %.2f ms per prediction pass".format(ms))
            }
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
        private const val DOUBLE_SPACE_MS = 500L

        /**
         * Context margin before a real-word slip goes to the model. Lower than the
         * contraction thresholds: the dictionary has already demanded a 20x frequency
         * gap, so the net only has to confirm the common reading fits here too.
         */
        private const val SLIP_MARGIN = 1.0f

        // Automatic bounce shims for the stuck-surface state were tried across
        // 0.5.5-0.5.9 (rotation-armed, then cross-field-armed, then quiet-period
        // debounced) and are deliberately gone: the state lives in the OEM's
        // WindowManager (see onUnstick), can engage on a plain keyboard open, survives
        // framework hide/show, and an automatic bounce both re-latched apps and raced
        // the user's own recovery. Recovery is the explicit long-press on ✨.
    }
}
