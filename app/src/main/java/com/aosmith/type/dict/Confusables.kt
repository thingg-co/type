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
     * ~0.25% false-positive budget): DIRECT at 4.0 measured 0.18% false flips / 52% catch.
     * Re-measured 2026-09-07 for the recurrent network on 21,000 dialogue cases, where every
     * net runs hotter: it flips 0.59% of correct usage at 4.0 (the dense net 0.40%) and 0.33%
     * at 5.0 with 58% catch, so the bar moved to 5.0; the lookback pass, which is where most
     * of the catch lives, is unchanged.
     */
    const val DIRECT_MARGIN = 5.0f
    const val MODEL_MARGIN = 2.0f

    /**
     * Margin for reconsidering the previous word once the word after it is known
     * ("your welcome": at "welcome", score P(variant|ctx) + P(welcome|ctx,variant)).
     * Two log-prob terms, and the pass where the real signal lives: at 5.0 the 126k
     * network measured 0.23% false flips / 75% catch on Tatoeba; on the dialogue cases the
     * recurrent network measures 0.35% / 86% at the same bar, against 0.43% / 79% for the
     * dense net it replaced. No model fallback here; the word prompt cannot ask about an
     * earlier word.
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
     * 2026-09-01 run, 500 cases per set per label; dialogue re-check 2026-09-06 on the
     * mixed-corpus network, 21,000 OpenSubtitles cases):
     *
     * its -> it's had its own bar at 3.0 for the 126k network (FP 0/83, 79% catch;
     * the global caught only 50%). The mixed-corpus network is sharper on this pair
     * and the low bar started flipping possessives ("the phone its battery" at 3.3),
     * so the pair is back on the global: at 5.0 it measures FP 0/24 and 86% catch
     * on dialogue, against 1/24 at 3.0 and at 4.0.
     *
     * id -> i'd at 6.0: bare "id" is legitimately ID in chat and the prose corpus
     * barely samples it, so the bar stays above the global; the new network cliffs
     * past 6.0 (94% catch there, 14% at 8.0, zero measured FP at either). The
     * acronym's casing itself waits for cased vocab ids.
     */
    private val LOOKBACK_OVERRIDES = mapOf(
        ("id" to "i'd") to 6.0f,
    )

    /** The lookback bar for flipping [typed] to [alt]; the global unless measured otherwise. */
    fun lookbackMargin(typed: String, alt: String): Float =
        LOOKBACK_OVERRIDES[typed.lowercase() to alt.lowercase()] ?: LOOKBACK_MARGIN

    /** Send-time lookback bar: never below the send global, whatever the pair override. */
    fun sendLookbackMargin(typed: String, alt: String): Float =
        maxOf(lookbackMargin(typed, alt), SEND_LOOKBACK_MARGIN)
}
