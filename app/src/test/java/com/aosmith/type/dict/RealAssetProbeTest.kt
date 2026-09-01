package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import org.junit.Assert.assertTrue
import org.junit.Test

/** Probes the policy against the real shipped assets (working dir is the app module). */
class RealAssetProbeTest {
    @Test fun `should ha yields context predictions with the shipped assets`() {
        val words = File("src/main/assets/en_words.txt")
        val big = File("src/main/assets/en_bigrams.bin")
        assertTrue("asset paths wrong: ${File(".").absolutePath}", words.exists() && big.exists())
        val dict = words.bufferedReader().useLines { Dictionary(it) }
        val bigrams = Bigrams.load(FileInputStream(big))
        println("dict=${dict.size} bigrams=${bigrams.size}")
        println("idOf(should)=${dict.idOf("should")} idOf(have)=${dict.idOf("have")}")
        println("score(should,have)=${bigrams.score(dict.idOf("should"), dict.idOf("have"))}")
        println("predictions(ha)=${dict.predictions("ha", 8)}")
        println("completions(ha,3)=${dict.completions("ha", 3)}")
        println("hasPrefix(ha)=${dict.hasPrefix("ha")}")
        val action = TypingPolicy.midWord(dict, bigrams, null, listOf("should"), "ha")
        println("ACTION=$action")
        assertTrue("got $action", action is MidWordAction.Predictions)
    }

    @Test fun `real-word slips in the shipped list are flagged, everyday words are not`() {
        val words = File("src/main/assets/en_words.txt")
        assertTrue("asset paths wrong: ${File(".").absolutePath}", words.exists())
        val dict = words.bufferedReader().useLines { Dictionary(it) }
        // The i/k and m/n debris the known-word gate used to wave through.
        assertTrue("make" in dict.slipCandidates("nake"))
        assertTrue("not" in dict.slipCandidates("mot"))
        assertTrue("now" in dict.slipCandidates("mow"))
        assertTrue("can" in dict.slipCandidates("cam"))
        // Common words with a frequent neighbour variant must stay untouched.
        for (w in listOf("night", "mine", "info", "mode", "tight", "form")) {
            assertTrue("'$w' was flagged: ${dict.slipCandidates(w)}", dict.slipCandidates(w).isEmpty())
        }
    }
}
