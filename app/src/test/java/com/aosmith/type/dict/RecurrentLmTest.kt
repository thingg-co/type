package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin GRU must reproduce tools/nn/tnw.py's recurrence (itself checked against torch) on a
 * tiny TNW5 asset generated with it: same int8 arithmetic, same BOS handling, and the prefix cache
 * must never change a result.
 */
class RecurrentLmTest {

    private fun load() = NeuralLm.load(FileInputStream(File("src/test/resources/tiny_tnw5.bin")))

    private fun golden() = JSONObject(File("src/test/resources/tiny_tnw5_golden.json").readText())

    private fun logitsOf(lm: NeuralLm, ctx: List<Int>): FloatArray {
        val h = lm.hidden(ctx)
        return FloatArray(lm.vocab) { lm.logit(it, h) }
    }

    @Test fun `loads as a recurrent network`() {
        val lm = load()
        assertTrue(lm.isRecurrent)
        assertEquals(16, lm.vocab)
        assertEquals(8, lm.dim)
        assertTrue(lm.eosTrained)
    }

    @Test fun `logits match the python reference on every golden case`() {
        val lm = load()
        val cases = golden().getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val ctx = c.getJSONArray("context").let { a -> List(a.length()) { a.getInt(it) } }
            val expected = c.getJSONArray("logits").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val got = logitsOf(lm, ctx)
            for (v in expected.indices) {
                assertEquals("case $ctx id $v", expected[v], got[v], 0.02f)
            }
        }
    }

    @Test fun `the prefix cache never changes a result`() {
        val warm = load()
        warm.hidden(listOf(3))
        warm.hidden(listOf(3, 7))
        val cached = logitsOf(warm, listOf(3, 7, 1))           // one step on top of the cached prefix
        val cold = logitsOf(load(), listOf(3, 7, 1))
        assertArrayEquals(cold, cached, 1e-5f)
        // a different sentence after the cache is warm
        val other = logitsOf(warm, listOf(12, 0, 5, 9, 2))
        assertArrayEquals(logitsOf(load(), listOf(12, 0, 5, 9, 2)), other, 1e-5f)
        // shorter than the cached prefix falls back to a fresh pass
        assertArrayEquals(logitsOf(load(), listOf(3)), logitsOf(warm, listOf(3)), 1e-5f)
    }

    @Test fun `leading BOS padding is dropped and unknown ids map to UNK`() {
        val lm = load()
        assertArrayEquals(logitsOf(lm, listOf(3, 7, 1)), logitsOf(lm, listOf(lm.bos, lm.bos, 3, 7, 1)), 1e-5f)
        assertArrayEquals(logitsOf(lm, listOf(lm.unk, 7)), logitsOf(lm, listOf(999, 7)), 1e-5f)
    }
}
