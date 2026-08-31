package com.aosmith.type.dict

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Builds an in-memory bigram file the same way tools/build_bigrams.py does. */
object TestBigrams {
    fun build(dict: Dictionary, vararg pairs: Triple<String, String, Int>): Bigrams {
        val entries = pairs.map { (a, b, count) ->
            val ia = dict.idOf(a)
            val ib = dict.idOf(b)
            require(ia >= 0 && ib >= 0) { "unknown word in $a/$b" }
            val score = (10 * ln(count.toDouble())).toInt().coerceIn(1, 255)
            (ia.toLong() shl 16 or ib.toLong()) to score
        }.sortedBy { it.first }
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { d ->
            d.write("TBG1".toByteArray())
            d.writeInt(entries.size)
            for ((key, _) in entries) d.writeLong(key)
            for ((_, score) in entries) d.writeByte(score)
        }
        return Bigrams.load(ByteArrayInputStream(bytes.toByteArray()))
    }
}

class BigramsTest {

    private val dict = Dictionary(TestWords.LIST.asSequence())
    private val bigrams = TestBigrams.build(
        dict,
        Triple("the", "cat", 500),
        Triple("the", "restaurant", 90),
        Triple("the", "experiment", 40),
        Triple("we", "can", 300),
    )

    @Test fun `scores round-trip through the binary format`() {
        assertTrue(bigrams.score(dict.idOf("the"), dict.idOf("cat")) > bigrams.score(dict.idOf("the"), dict.idOf("restaurant")))
        assertEquals(0, bigrams.score(dict.idOf("the"), dict.idOf("typing")))
        assertEquals(0, bigrams.score(-1, dict.idOf("cat")))
    }

    @Test fun `next words come back best first`() {
        val next = bigrams.nextWords(dict.idOf("the"), 10).map { dict.wordOf(it.first) }
        assertEquals(listOf("cat", "restaurant", "experiment"), next)
        assertEquals(listOf("cat"), bigrams.nextWords(dict.idOf("the"), 1).map { dict.wordOf(it.first) })
        assertTrue(bigrams.nextWords(dict.idOf("typing"), 5).isEmpty())
    }

    @Test fun `size reports the pair count`() {
        assertEquals(4, bigrams.size)
    }
}
