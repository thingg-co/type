package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Golden-vector test: the Kotlin quantized inference must reproduce the logits computed by
 * tools/nn/train.py at export time (same int8 arithmetic). Skips until the trained asset
 * and tools/nn/out/golden.json exist.
 */
class NeuralLmTest {

    @Test fun `kotlin inference matches the python export`() {
        val asset = File("src/main/assets/en_nextword.bin")
        val golden = File("../tools/nn/out/golden.json")
        assumeTrue("model not trained yet", asset.exists() && golden.exists())

        val lm = NeuralLm.load(FileInputStream(asset))
        val g = JSONObject(golden.readText())
        val ctx = g.getJSONArray("context").let { a -> List(a.length()) { a.getInt(it) } }
        val topIds = g.getJSONArray("top_ids").let { a -> List(a.length()) { a.getInt(it) } }
        val topLogits = g.getJSONArray("top_logits").let { a -> List(a.length()) { a.getDouble(it) } }

        val h = lm.hidden(ctx)
        for (i in topIds.indices) {
            assertEquals("logit for id ${topIds[i]}", topLogits[i], lm.logit(topIds[i], h).toDouble(), 0.05)
        }
        val kotlinTop = lm.topNext(ctx, 5)
        assertEquals(topIds.filter { it < lm.bos }.take(3), kotlinTop.take(3))
    }
}
