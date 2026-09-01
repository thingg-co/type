package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Calibration harness, not a regression test: replays confusable cases through the real
 * network and dumps margins for tools/confusables_calibrate.py to aggregate. Runs only
 * when CONFUSABLE_CASES points at a cases file (label \t ctx \t typed \t intended \t next
 * per line); writes margins.tsv next to it. See Confusables for the thresholds this
 * calibrates.
 */
class ConfusableCalibrationHarness {

    @Test fun `dump margins for the calibration corpus`() {
        val casesPath = System.getenv("CONFUSABLE_CASES")
        assumeTrue("CONFUSABLE_CASES not set; skipping calibration dump", casesPath != null)
        val cases = File(casesPath!!)
        assumeTrue("cases file missing", cases.exists())
        val dict = File("src/main/assets/en_words.txt").bufferedReader().useLines { Dictionary(it) }
        val lm = NeuralLm.load(FileInputStream(File("src/main/assets/en_nextword.bin")))

        fun id(w: String): Int = dict.idOf(w)
        fun idOrUnk(w: String): Int = id(w).let { if (it >= 0) it else lm.unk }

        val out = File(cases.parentFile, "confusable_margins.tsv").bufferedWriter()
        out.write("label\tset\tctxlen\ttyped\tintended\tnext\tfinal\tfwd_margin\tfwd_winner\tlook_margin\tlook_winner\tfwd_eos_margin\tfwd_eos_winner\tlook_eos_margin\tlook_eos_winner\tskip\n")
        cases.forEachLine { line ->
            val p = line.split('\t')
            if (p.size < 5) return@forEachLine
            val (label, ctxRaw, typed, intended, next) = p
            val final = p.getOrElse(5) { "0" }
            val alts = Confusables.alternativesOf(typed)
            if (alts.isEmpty()) return@forEachLine
            val ctxWords = ctxRaw.split(' ').filter { it.isNotEmpty() }
            val ctx = ctxWords.map(::idOrUnk)
            val typedId = id(typed)
            val altIds = alts.map(::id)
            val nextId = id(next)
            val skip = when {
                typedId < 0 || altIds.any { it < 0 } -> "vocab"
                else -> ""
            }
            var fwdMargin = Float.NaN
            var fwdWinner = ""
            var lookMargin = Float.NaN
            var lookWinner = ""
            var fwdEosMargin = Float.NaN
            var fwdEosWinner = ""
            var lookEosMargin = Float.NaN
            var lookEosWinner = ""

            fun best(totals: (Int) -> Float, assign: (Float, String) -> Unit) {
                val typedTotal = totals(typedId)
                var bestAlt = Float.NEGATIVE_INFINITY
                var winner = ""
                for ((i, a) in altIds.withIndex()) {
                    val t = totals(a)
                    if (t > bestAlt) {
                        bestAlt = t
                        winner = alts[i]
                    }
                }
                assign(bestAlt - typedTotal, winner)
            }

            if (skip.isEmpty()) {
                val scores = lm.scoreCandidates(ctx, intArrayOf(typedId) + altIds.toIntArray())
                val bestIdx = (1 until scores.size).maxBy { scores[it] }
                fwdMargin = scores[bestIdx] - scores[0]
                fwdWinner = alts[bestIdx - 1]
                if (nextId >= 0) {
                    best({ v ->
                        lm.scoreCandidates(ctx, intArrayOf(v))[0] +
                            lm.scoreCandidates(ctx + v, intArrayOf(nextId))[0]
                    }) { m, w -> lookMargin = m; lookWinner = w }
                }
                if (lm.eosTrained) {
                    if (next.isEmpty()) {
                        // sentence-final word: the send-time forward+EOS decision
                        best({ v ->
                            lm.scoreCandidates(ctx, intArrayOf(v))[0] +
                                lm.scoreCandidates(ctx + v, intArrayOf(lm.bos))[0]
                        }) { m, w -> fwdEosMargin = m; fwdEosWinner = w }
                    } else if (final == "1" && nextId >= 0) {
                        // next word ends the sentence: the send-time lookback+EOS decision
                        best({ v ->
                            lm.scoreCandidates(ctx, intArrayOf(v))[0] +
                                lm.scoreCandidates(ctx + v, intArrayOf(nextId))[0] +
                                lm.scoreCandidates(ctx + v + nextId, intArrayOf(lm.bos))[0]
                        }) { m, w -> lookEosMargin = m; lookEosWinner = w }
                    }
                }
            }
            val set = (listOf(typed) + alts).sorted().joinToString("/")
            out.write("$label\t$set\t${ctxWords.size}\t$typed\t$intended\t$next\t$final\t$fwdMargin\t$fwdWinner\t$lookMargin\t$lookWinner\t$fwdEosMargin\t$fwdEosWinner\t$lookEosMargin\t$lookEosWinner\t$skip\n")
        }
        out.close()
        println("margins written next to $casesPath (eosTrained=${lm.eosTrained})")
    }
}
