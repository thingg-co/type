package com.aosmith.type.dict

/**
 * Words that are routinely swapped for each other ("then"/"than", "there"/"their"/
 * "they're", the bare contraction forms). The table only says which words are worth
 * scoring; every decision is made by the prediction network from context, with the
 * language model taking the gray zone. Nothing about usage is encoded here.
 *
 * Thresholds are calibrated, not chosen: tools/confusables_calibrate.py replays a
 * corpus of natural sentences (correct usages measure false positives, swapped ones
 * measure catch rate) through the real network via ConfusableCalibrationHarness.
 */
object Confusables {

    private val SETS = listOf(
        // contraction pairs (bare form is also a real word)
        listOf("were", "we're"), listOf("well", "we'll"), listOf("ill", "i'll"),
        listOf("id", "i'd"), listOf("its", "it's"), listOf("lets", "let's"),
        listOf("hell", "he'll"), listOf("shell", "she'll"), listOf("wed", "we'd"),
        listOf("whose", "who's"), listOf("your", "you're"), listOf("there", "their", "they're"),
        // homophone and near-homophone pairs
        listOf("then", "than"), listOf("lose", "loose"), listOf("quite", "quiet"),
        listOf("weather", "whether"), listOf("accept", "except"), listOf("affect", "effect"),
        listOf("advice", "advise"), listOf("passed", "past"), listOf("brake", "break"),
    )

    private val alternatives: Map<String, List<String>> = buildMap {
        for (set in SETS) for (w in set) put(w, set.filterNot { it == w })
    }

    /** The other members of [typed]'s confusable set, or empty when it is in none. */
    fun alternativesOf(typed: String): List<String> = alternatives[typed.lowercase()] ?: emptyList()

    /**
     * Margins for the net's typed-vs-alternative decision at the word's own boundary
     * (only the words before it are known). Above [DIRECT_MARGIN] the net's pick is
     * applied outright; between [MODEL_MARGIN] and [DIRECT_MARGIN] the language model
     * decides; below, the word is left alone.
     *
     * Calibrated for the 126k-vocabulary network (2026-09-01, Tatoeba held-out cases,
     * ~0.25% false-positive budget). DIRECT at 4.0: 0.18% false flips / 52% catch.
     */
    const val DIRECT_MARGIN = 4.0f
    const val MODEL_MARGIN = 2.0f

    /**
     * Margin for reconsidering the previous word once the word after it is known
     * ("your welcome": at "welcome", score P(variant|ctx) + P(welcome|ctx,variant)).
     * Two log-prob terms, and the pass where the real signal lives: at 5.0 the new
     * network measures 0.23% false flips / 75% catch. No model fallback here; the
     * word prompt cannot ask about an earlier word.
     */
    const val LOOKBACK_MARGIN = 5.0f

    /**
     * Send-time margins (enter that may dispatch the message): the totals gain the
     * trained end-of-message term, and there is no undo once sent, so the bars stay
     * strict. Forward+EOS at 6.0: 0.09% false flips / 83% catch; lookback+EOS at
     * 6.0: 0.09% / 80% (the new network's EOS term is sharp enough that 8.0 would
     * cost most of the catch for no measured safety).
     */
    const val SEND_FORWARD_MARGIN = 6.0f
    const val SEND_LOOKBACK_MARGIN = 6.0f

    /**
     * Directed per-pair lookback overrides, measured like the globals (Tatoeba
     * 2026-09-01 run, 500 cases per set per label):
     *
     * its -> it's at 3.0: FP 0/83 with 79% catch on the 126k network (the global 5.0
     * catches only 50%). The prior is real — typed "its" is usually a meant "it's" —
     * and the lookback pass is where the sentence decides it.
     *
     * id -> i'd at 6.0: bare "id" is legitimately ID in chat and the prose corpus
     * barely samples it, so the bar stays above the global; the new network cliffs
     * past 6.0 (94% catch there, 14% at 8.0, zero measured FP at either). The
     * acronym's casing itself waits for cased vocab ids.
     */
    private val LOOKBACK_OVERRIDES = mapOf(
        ("its" to "it's") to 3.0f,
        ("id" to "i'd") to 6.0f,
    )

    /** The lookback bar for flipping [typed] to [alt]; the global unless measured otherwise. */
    fun lookbackMargin(typed: String, alt: String): Float =
        LOOKBACK_OVERRIDES[typed.lowercase() to alt.lowercase()] ?: LOOKBACK_MARGIN

    /** Send-time lookback bar: never below the send global, whatever the pair override. */
    fun sendLookbackMargin(typed: String, alt: String): Float =
        maxOf(lookbackMargin(typed, alt), SEND_LOOKBACK_MARGIN)
}
