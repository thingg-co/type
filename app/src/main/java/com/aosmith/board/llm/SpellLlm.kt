package com.aosmith.board.llm

import android.util.Log
import com.aosmith.board.dict.Dictionary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import kotlin.coroutines.coroutineContext

/**
 * High-level spell corrector on top of [LlamaNative].
 *
 * All native calls run on one dedicated thread. Two prompt "modes" exist (single word and
 * whole sentence); the fixed prefix for the active mode is kept in the KV cache so a request
 * only pays for its own few dozen tokens.
 */
class SpellLlm(private val dictionary: Dictionary) {

    sealed class State {
        object Idle : State()
        object Loading : State()
        data class Ready(val info: String) : State()
        data class Error(val message: String) : State()
    }

    private enum class Mode { WORD, SENTENCE }

    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "board-llm") }
    val dispatcher = executor.asCoroutineDispatcher()
    val scope = CoroutineScope(SupervisorJob() + dispatcher)

    @Volatile var state: State = State.Idle
        private set
    @Volatile var lastLatencyMs: Long = 0
        private set
    @Volatile private var loadedPath: String? = null

    private var mode: Mode? = null
    private var wordPrefix: String = ""
    private var wordTail: String = ""
    private var sentencePrefix: String = ""
    private var sentenceTail: String = ""

    private val cache = object : LinkedHashMap<String, String?>(64, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String?>?) = size > 200
    }

    val isReady: Boolean get() = state is State.Ready

    /** Loads [file] unless it is already loaded. Safe to call repeatedly. */
    suspend fun ensureLoaded(file: File, threads: Int): Boolean = withContext(dispatcher) {
        if (loadedPath == file.absolutePath && LlamaNative.isLoaded()) return@withContext true
        state = State.Loading
        val t0 = System.currentTimeMillis()
        val ok = try {
            LlamaNative.load(file.absolutePath, threads, N_CTX)
        } catch (t: Throwable) {
            Log.e(TAG, "native load threw", t)
            false
        }
        if (!ok) {
            state = State.Error("could not load ${file.name}")
            loadedPath = null
            return@withContext false
        }
        loadedPath = file.absolutePath
        mode = null
        synchronized(cache) { cache.clear() }
        buildTemplates()
        val info = LlamaNative.modelInfo()?.toString(Charsets.UTF_8) ?: file.name
        Log.i(TAG, "loaded in ${System.currentTimeMillis() - t0} ms: $info")
        state = State.Ready(info)
        // Warm the word prefix now so the first correction is not slow.
        switchMode(Mode.WORD)
        true
    }

    suspend fun unload() = withContext(dispatcher) {
        LlamaNative.free()
        loadedPath = null
        mode = null
        state = State.Idle
    }

    /** Cancels whatever the native side is doing right now. Callable from any thread. */
    fun cancelCurrent() = LlamaNative.cancel()

    /**
     * Returns the corrected form of [word] given the text [before] it, or null when the model
     * agrees with the user or produced something implausible.
     */
    suspend fun correctWord(before: String, word: String): String? = withContext(dispatcher) {
        if (!isReady) return@withContext null
        val request = Prompts.wordRequest(before, word)
        val key = "w|$request"
        synchronized(cache) { if (cache.containsKey(key)) return@withContext cache[key] }
        switchMode(Mode.WORD)
        coroutineContext.ensureActive()
        val t0 = System.currentTimeMillis()
        val raw = LlamaNative.complete(request + wordTail, 12, Prompts.WORD_GRAMMAR, true)
            ?.toString(Charsets.UTF_8)
        lastLatencyMs = System.currentTimeMillis() - t0
        val result = sanitizeWord(word, raw)
        Log.d(TAG, "word '$word' -> raw='$raw' accepted=$result (${lastLatencyMs} ms)")
        synchronized(cache) { cache[key] = result }
        result
    }

    /** Whole-sentence correction. Returns null when nothing changed or the output looks wrong. */
    suspend fun correctSentence(text: String): String? = withContext(dispatcher) {
        if (!isReady) return@withContext null
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return@withContext null
        switchMode(Mode.SENTENCE)
        coroutineContext.ensureActive()
        val t0 = System.currentTimeMillis()
        val maxTokens = (trimmed.length / 2 + 16).coerceIn(16, 160)
        val raw = LlamaNative.complete(trimmed + sentenceTail, maxTokens, null, true)?.toString(Charsets.UTF_8)
        lastLatencyMs = System.currentTimeMillis() - t0
        val result = sanitizeSentence(trimmed, raw)
        Log.d(TAG, "sentence -> raw='$raw' accepted=$result (${lastLatencyMs} ms)")
        result
    }

    // ---- prompt plumbing -------------------------------------------------------------

    private fun buildTemplates() {
        val (wp, wt) = splitTemplate(
            Prompts.WORD_EXAMPLES.map { (u, a) -> (Prompts.WORD_INSTRUCTION + u) to a },
            Prompts.WORD_INSTRUCTION,
        )
        wordPrefix = wp
        wordTail = wt
        val (sp, st) = splitTemplate(
            Prompts.SENTENCE_EXAMPLES.map { (u, a) -> (Prompts.SENTENCE_INSTRUCTION + u) to a },
            Prompts.SENTENCE_INSTRUCTION,
        )
        sentencePrefix = sp
        sentenceTail = st
    }

    /**
     * Renders system + few-shot turns + a final user turn through the model's chat template
     * and splits the result at the point where the per-request text goes.
     */
    private fun splitTemplate(examples: List<Pair<String, String>>, instruction: String): Pair<String, String> {
        val roles = ArrayList<String>()
        val contents = ArrayList<String>()
        roles += "system"; contents += Prompts.SYSTEM
        for ((u, a) in examples) {
            roles += "user"; contents += u
            roles += "assistant"; contents += a
        }
        roles += "user"; contents += instruction + MARKER
        var formatted = LlamaNative.formatChat(roles.toTypedArray(), contents.toTypedArray(), true)
            ?.toString(Charsets.UTF_8)
        if (formatted == null || !formatted.contains(MARKER)) {
            // Model without a usable template: fall back to ChatML, which most small models accept.
            val sb = StringBuilder()
            for (i in roles.indices) {
                sb.append("<|im_start|>").append(roles[i]).append('\n').append(contents[i]).append("<|im_end|>\n")
            }
            sb.append("<|im_start|>assistant\n")
            formatted = sb.toString()
        }
        val idx = formatted.indexOf(MARKER)
        var tail = formatted.substring(idx + MARKER.length)
        // Qwen3 templates open the assistant turn with an empty think block when thinking is off;
        // llama_chat_apply_template does not add it, so do it here.
        val info = LlamaNative.modelInfo()?.toString(Charsets.UTF_8) ?: ""
        if (info.startsWith("qwen3", ignoreCase = true) && !tail.contains("<think>")) {
            tail += "<think>\n\n</think>\n\n"
        }
        return formatted.substring(0, idx) to tail
    }

    private fun switchMode(m: Mode) {
        if (mode == m) return
        val prefix = if (m == Mode.WORD) wordPrefix else sentencePrefix
        val n = LlamaNative.setPrefix(prefix)
        if (n < 0) {
            Log.e(TAG, "setPrefix failed for $m")
            mode = null
        } else {
            mode = m
        }
    }

    // ---- output validation ----------------------------------------------------------

    private fun sanitizeWord(original: String, raw: String?): String? {
        if (raw == null) return null
        val candidate = raw.trim().trim('[', ']', '"', '.', ',', '!', '?', ':', ';').trim()
        if (candidate.isEmpty() || candidate.any { it.isWhitespace() }) return null
        if (candidate.equals(original, ignoreCase = true)) return null
        val distance = Dictionary.editDistance(candidate.lowercase(), original.lowercase())
        val limit = when {
            original.length <= 3 -> 1
            original.length <= 6 -> 2
            else -> 3
        }
        if (distance > limit) return null
        val known = dictionary.isKnown(candidate)
        // An unknown output is only trusted for a one-edit change; otherwise the model is guessing.
        if (!known && distance > 1) return null
        return Dictionary.matchCase(original, candidate)
    }

    private fun sanitizeSentence(original: String, raw: String?): String? {
        if (raw == null) return null
        val candidate = raw.trim().removeSurrounding("\"")
        if (candidate.isEmpty() || candidate == original) return null
        // Reject rewrites: length should stay close and most words should survive.
        val ratio = candidate.length.toDouble() / original.length
        if (ratio < 0.6 || ratio > 1.5) return null
        val origWords = original.lowercase().split(Regex("\\s+"))
        val candWords = candidate.lowercase().split(Regex("\\s+"))
        if (kotlin.math.abs(origWords.size - candWords.size) > 2) return null
        val overlap = origWords.count { it in candWords }
        if (overlap < origWords.size * 0.5) return null
        return candidate
    }

    companion object {
        private const val TAG = "SpellLlm"
        private const val MARKER = "REQ"
        const val N_CTX = 1024
    }
}
