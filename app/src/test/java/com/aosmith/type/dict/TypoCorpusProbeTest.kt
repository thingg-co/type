package com.aosmith.type.dict

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Replays Wikipedia's common-misspellings corpus through the boundary pipeline exactly as
 * the service runs it (contraction map, then the known-word gate, then dictionary
 * suggestions standing in for the model path). Prints where typos die so the pipeline's
 * blind spots are measured, not guessed, and holds a floor so they cannot regress.
 */
class TypoCorpusProbeTest {

    @Test fun `corpus coverage has a floor and known blind spots are quantified`() {
        val corpus = File("../tools/data/typos.tsv")
        val words = File("src/main/assets/en_words.txt")
        assumeTrue(corpus.exists() && words.exists())
        val dict = words.bufferedReader().useLines { Dictionary(it) }
        val table = File("src/main/assets/en_typos.tsv")
        if (table.exists()) dict.misspellings = TypoTable.load(table.inputStream())

        var total = 0
        var mapFixed = 0
        var gateBlocked = 0        // typo is itself a "known word": nothing ever fires
        var top1 = 0
        var top3 = 0
        var unreachable = 0        // unknown but suggestions miss; the LLM's territory
        val blockedSamples = ArrayList<String>()
        corpus.readLines().forEach { line ->
            val parts = line.split('\t')
            if (parts.size < 2) return@forEach
            val typo = parts[0]
            val wanted = parts.drop(1).toSet()
            total++
            val m = Contractions.fix(typo) ?: dict.misspellings?.fix(typo)
            if (m != null) {
                if (m.lowercase() in wanted) mapFixed++
                return@forEach
            }
            if (dict.isKnown(typo)) {
                gateBlocked++
                if (blockedSamples.size < 12) blockedSamples += "$typo->${wanted.first()}"
                return@forEach
            }
            val s = dict.suggest(typo, 3).map { it.lowercase() }
            when {
                s.firstOrNull() in wanted -> { top1++; top3++ }
                s.any { it in wanted } -> top3++
                else -> unreachable++
            }
        }
        val reachable = total - gateBlocked
        println("corpus $total | map $mapFixed | GATE-BLOCKED $gateBlocked (${100 * gateBlocked / total}%)")
        println("of reachable $reachable: suggest top1 $top1 (${100 * top1 / reachable}%) top3 $top3 (${100 * top3 / reachable}%) unreachable-by-dict $unreachable")
        println("gate-blocked samples: $blockedSamples")
        assertTrue("instant-fix floor", mapFixed >= total * 80 / 100)
        assertTrue("gate-blocked must stay near zero", gateBlocked <= total / 100)
        assertTrue("total coverage floor", mapFixed + top1 >= total * 85 / 100)
    }
}
