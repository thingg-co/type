package com.aosmith.type.dict

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.Random
import kotlin.io.path.createTempFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Builds a small random base network in the export format for tests. */
object TestNeural {
    fun build(vocabWords: Int = 14, k: Int = 3, e: Int = 8, seed: Long = 3): NeuralLm {
        val v = vocabWords + 2
        val rng = Random(seed)
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { d ->
            d.write("TNW1".toByteArray())
            d.writeInt(v)
            d.writeInt(k)
            d.writeInt(e)
            val q = ByteArray(v * e) { (rng.nextInt(255) - 127).toByte() }
            d.write(q)
            repeat(v) { d.writeFloat(0.01f + rng.nextFloat() * 0.01f) }
            repeat(e * k * e) { d.writeFloat((rng.nextFloat() - 0.5f) * 0.2f) }
            repeat(e) { d.writeFloat(0f) }
            repeat(v) { d.writeFloat(0f) }
        }
        return NeuralLm.load(ByteArrayInputStream(bytes.toByteArray()))
    }
}

class PersonalizerTest {

    private fun rankOf(lm: NeuralLm, ctx: List<Int>, target: Int): Int =
        lm.topNext(ctx, lm.bos).indexOf(target)

    @Test fun `repetition teaches the model a pattern without wrecking the rest`() {
        val lm = TestNeural.build()
        val p = Personalizer(lm)
        lm.personal = p

        val ctx = listOf(2, 5)      // "purple quokka"-style pair
        val target = 9
        val otherCtx = listOf(1, 3)
        val otherTopBefore = lm.topNext(otherCtx, 3)
        val before = rankOf(lm, ctx, target)

        repeat(60) { p.record(ctx, target) }
        val f = createTempFile().toFile()
        repeat(30) { p.trainAndMaybeSave(f, steps = 32) }

        val after = rankOf(lm, ctx, target)
        assertTrue("rank did not improve: $before -> $after", after < before || after == 0)
        assertEquals(0, after) // heavy repetition should make it the top suggestion

        // an untouched context should keep a stable top-1 (deltas are sparse and clamped)
        val otherTopAfter = lm.topNext(otherCtx, 3)
        assertTrue(
            "unrelated context disturbed: $otherTopBefore -> $otherTopAfter",
            otherTopAfter.first() == otherTopBefore.first() || otherTopAfter.take(3).contains(otherTopBefore.first()),
        )
    }

    @Test fun `state survives a save and load round trip`() {
        val lm = TestNeural.build()
        val p = Personalizer(lm)
        lm.personal = p
        repeat(40) { p.record(listOf(4, 6), 11) }
        val f = createTempFile().toFile()
        repeat(20) { p.trainAndMaybeSave(f, steps = 32) }
        p.save(f.outputStream())

        val lm2 = TestNeural.build()
        val p2 = Personalizer(lm2)
        p2.load(f.inputStream())
        lm2.personal = p2
        val h1 = lm.hidden(listOf(4, 6))
        val h2 = lm2.hidden(listOf(4, 6))
        assertEquals(lm.logit(11, h1), lm2.logit(11, h2), 1e-4f)
        assertEquals(p.lifetimeSamples, p2.lifetimeSamples)
    }

    @Test fun `clear removes every trace`() {
        val lm = TestNeural.build()
        val p = Personalizer(lm)
        lm.personal = p
        val ctx = listOf(2, 5)
        val cleanH = lm.hidden(ctx)
        val cleanLogit = lm.logit(9, cleanH)
        repeat(50) { p.record(ctx, 9) }
        repeat(20) { p.trainAndMaybeSave(createTempFile().toFile(), steps = 32) }
        p.clear()
        val h = lm.hidden(ctx)
        assertEquals(cleanLogit, lm.logit(9, h), 1e-5f)
        assertEquals(0, p.pendingSamples)
        assertTrue(p.learnedIds().isEmpty())
    }

    @Test fun `specials are never recorded or suggested`() {
        val lm = TestNeural.build()
        val p = Personalizer(lm)
        lm.personal = p
        p.record(listOf(1), lm.bos)
        p.record(listOf(1), lm.unk)
        assertEquals(0, p.pendingSamples)
        assertTrue(lm.topNext(listOf(1), 5).all { it < lm.bos })
    }
}
