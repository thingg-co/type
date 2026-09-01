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
     * Calibrated on two corpora through the shipped network: held-out conversational
     * sentences (the keyboard's domain) and book prose (out of domain, pessimistic).
     * DIRECT at 6.0: 0.14% false flips / 58% catch conversational, 0.59% / 31% books.
     */
    const val DIRECT_MARGIN = 6.0f
    const val MODEL_MARGIN = 3.0f

    /**
     * Margin for reconsidering the previous word once the word after it is known
     * ("your welcome": at "welcome", score P(variant|ctx) + P(welcome|ctx,variant)).
     * Two log-prob terms, and the pass where the real signal lives: at 6.0 it is
     * 0.10% false flips / 85% catch conversational, 0.67% / 58% books. No model
     * fallback here; the word prompt cannot ask about an earlier word.
     */
    const val LOOKBACK_MARGIN = 6.0f

    /**
     * Send-time margins (enter that may dispatch the message): the totals gain the
     * trained end-of-message term, and there is no undo once sent, so the lookback
     * bar sits higher. Forward+EOS at 6.0 measured zero false flips on both corpora
     * (96% catch conversational); lookback+EOS at 8.0 is 0.17% / 82% conversational.
     */
    const val SEND_FORWARD_MARGIN = 6.0f
    const val SEND_LOOKBACK_MARGIN = 8.0f
}
