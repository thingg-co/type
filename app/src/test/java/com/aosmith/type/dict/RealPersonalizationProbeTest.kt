package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import kotlin.io.path.createTempFile
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Personalization against the real shipped network: teaching it "purple walrus" a handful
 * of times must lift "walrus" into the top next-word suggestions after "purple". Prints
 * the trajectory so tuning changes are measurable, not guessed.
 */
class RealPersonalizationProbeTest {

    @Test fun `a few repetitions surface the learned word`() {
        val asset = File("src/main/assets/en_nextword.bin")
        val words = File("src/main/assets/en_words.txt")
        assumeTrue(asset.exists() && words.exists())
        val dict = words.bufferedReader().useLines { Dictionary(it) }
        val lm = NeuralLm.load(FileInputStream(asset))
        val p = Personalizer(lm)
        lm.personal = p

        val purple = dict.idOf("purple")
        val walrus = dict.idOf("walrus")
        assertTrue(purple >= 0 && walrus >= 0)
        val ctx = listOf(purple)

        fun rank(): Int = lm.topNext(ctx, 200).indexOf(walrus)
        fun logitOf(id: Int) = lm.logit(id, lm.hidden(ctx))

        println("before: rank=${rank()} walrusLogit=${logitOf(walrus)}")
        val f = createTempFile().toFile()
        repeat(4) { burst ->
            repeat(8) {
                p.record(listOf(walrus), purple)
                p.record(ctx, walrus)
            }
            p.trainAndMaybeSave(f, steps = 192)
            println("after burst ${burst + 1}: rank=${rank()} walrusLogit=${logitOf(walrus)}")
        }
        val finalRank = rank()
        assertTrue("walrus should reach the top-3 after four bursts, rank=$finalRank", finalRank in 0..2)
    }
}
