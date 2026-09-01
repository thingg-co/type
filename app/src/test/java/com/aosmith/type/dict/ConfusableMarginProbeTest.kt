package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test

/**
 * Pins the shipped network's behavior on canonical confusable sentences against the
 * calibrated thresholds in [Confusables]. The corpus-level rates come from
 * ConfusableCalibrationHarness; these are the named cases that must keep working
 * (and the correct usages that must never flip) across retrains.
 */
class ConfusableMarginProbeTest {

    companion object {
        private lateinit var dict: Dictionary
        private lateinit var lm: NeuralLm

        @JvmStatic
        @BeforeClass
        fun load() {
            dict = File("src/main/assets/en_words.txt").bufferedReader().useLines { Dictionary(it) }
            lm = NeuralLm.load(FileInputStream(File("src/main/assets/en_nextword.bin")))
        }
    }

    private fun ids(words: List<String>) = words.map { dict.idOf(it).let { id -> if (id >= 0) id else lm.unk } }

    private fun forwardMargin(ctx: List<String>, typed: String): Pair<Float, String> {
        val alts = Confusables.alternativesOf(typed)
        val scores = lm.scoreCandidates(ids(ctx), intArrayOf(dict.idOf(typed)) + alts.map(dict::idOf).toIntArray())
        val best = (1 until scores.size).maxBy { scores[it] }
        return (scores[best] - scores[0]) to alts[best - 1]
    }

    private fun lookbackMargin(ctx: List<String>, typed: String, next: String): Pair<Float, String> {
        val alts = Confusables.alternativesOf(typed)
        val ctxIds = ids(ctx)
        val nextId = dict.idOf(next)
        fun total(v: Int): Float =
            lm.scoreCandidates(ctxIds, intArrayOf(v))[0] + lm.scoreCandidates(ctxIds + v, intArrayOf(nextId))[0]
        val typedTotal = total(dict.idOf(typed))
        var bestAlt = Float.NEGATIVE_INFINITY
        var winner = ""
        for (a in alts) {
            val t = total(dict.idOf(a))
            if (t > bestAlt) {
                bestAlt = t
                winner = a
            }
        }
        return (bestAlt - typedTotal) to winner
    }

    @Test fun `forward pass fixes the clear cases and spares correct usage`() {
        // Must fire (margin above DIRECT_MARGIN, right winner).
        for ((ctx, typed, want) in listOf(
            Triple(listOf("i", "think"), "ill", "i'll"),
            Triple(listOf("maybe"), "lets", "let's"),
            Triple(listOf("if", "so"), "id", "i'd"),
            Triple(listOf("do", "you", "think"), "its", "it's"),
        )) {
            val (margin, winner) = forwardMargin(ctx, typed)
            assertTrue("[$ctx $typed] margin $margin must clear DIRECT", margin > Confusables.DIRECT_MARGIN)
            assertTrue("[$ctx $typed] winner $winner", winner == want)
        }
        // Correct usage must stay quiet (below MODEL_MARGIN: not even a model query).
        for ((ctx, typed) in listOf(
            listOf("they", "said", "we") to "were",
            listOf("that", "went") to "well",
            listOf("he", "was", "very") to "ill",
            listOf("the", "dog", "wagged") to "its",
            listOf("she") to "lets",
            listOf("put", "it") to "there",
            listOf("i", "like") to "their",
            listOf("see", "you") to "then",
        )) {
            val (margin, _) = forwardMargin(ctx, typed)
            assertTrue("[$ctx $typed] margin $margin must stay under MODEL", margin < Confusables.MODEL_MARGIN)
        }
    }

    @Test fun `lookback pass decides once the next word is known`() {
        for ((case, want) in listOf(
            Triple(listOf("i", "think"), "were", "going") to "we're",
            Triple(listOf("i", "hope"), "well", "see") to "we'll",
            Triple(listOf("is", "bigger"), "then", "me") to "than",
            Triple(listOf("rather"), "then", "that") to "than",
        )) {
            val (ctx, typed, next) = case
            val (margin, winner) = lookbackMargin(ctx, typed, next)
            println("lookback [$ctx $typed $next] margin=%.2f winner=$winner".format(margin))
            assertTrue("[$ctx $typed $next] margin $margin must clear LOOKBACK", margin > Confusables.LOOKBACK_MARGIN)
            assertTrue("[$ctx $typed $next] winner $winner", winner == want)
        }
        // Correct usage with the next word known must stay quiet.
        for (case in listOf(
            Triple(listOf("i", "like"), "their", "car"),
            Triple(listOf("put", "it"), "there", "now"),
            Triple(listOf("thanks", "for"), "your", "help"),
            Triple(listOf("see", "you"), "then", "tomorrow"),
            Triple(listOf("we"), "were", "there"),
        )) {
            val (ctx, typed, next) = case
            val (margin, winner) = lookbackMargin(ctx, typed, next)
            println("lookback-neg [$ctx $typed $next] margin=%.2f winner=$winner".format(margin))
            assertTrue("[$ctx $typed $next] margin $margin must stay under LOOKBACK", margin < Confusables.LOOKBACK_MARGIN)
        }
    }

    @Test fun `directed overrides hold their calibrated values`() {
        // its -> it's earns a lower bar (measured FP 0/83 at 3.0 on the 126k net);
        // id -> i'd a raised one (bare ID is real chat usage the corpus barely samples).
        assertEquals(3.0f, Confusables.lookbackMargin("its", "it's"))
        assertEquals(6.0f, Confusables.lookbackMargin("id", "i'd"))
        // Unlisted directions fall back to the global.
        assertEquals(Confusables.LOOKBACK_MARGIN, Confusables.lookbackMargin("your", "you're"))
        assertEquals(Confusables.LOOKBACK_MARGIN, Confusables.lookbackMargin("it's", "its"))
        // Send-time never dips below the send global.
        assertEquals(Confusables.SEND_LOOKBACK_MARGIN, Confusables.sendLookbackMargin("its", "it's"))
    }

    @Test fun `its flips to it's by lookback at the lowered bar`() {
        // The user prior: typed "its" is usually a meant "it's"; the sentence decides.
        val (margin, winner) = lookbackMargin(listOf("i", "think"), "its", "great")
        println("lookback [i think its great] margin=%.2f winner=$winner".format(margin))
        assertTrue("winner $winner", winner == "it's")
        assertTrue("margin $margin must clear the its->it's bar", margin > Confusables.lookbackMargin("its", "it's"))
        // Possessive usage must stay quiet even at the lowered bar.
        val (negMargin, negWinner) = lookbackMargin(listOf("the", "phone"), "its", "battery")
        println("lookback-neg [the phone its battery] margin=%.2f winner=$negWinner".format(negMargin))
        assertTrue(
            "possessive margin $negMargin must stay under the its->it's bar",
            negWinner != "it's" || negMargin < Confusables.lookbackMargin("its", "it's"),
        )
    }
}
