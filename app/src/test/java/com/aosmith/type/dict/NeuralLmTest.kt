package com.aosmith.type.dict

import java.io.File
import java.io.FileInputStream
import java.util.Random
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

    @Test fun `two-layer trunk backprop matches finite differences`() {
        // Build a two-layer net (vocabWords 14, k 3, e 8, hidden 16)
        val lm = TestNeural.build(vocabWords = 14, k = 3, e = 8, layers = 2, hidden = 16, seed = 7)
        val rng = Random(42)

        // Pick a context of three ids and an output id t
        val ctx = listOf(2, 5, 9)
        val t = 11

        // Compute hidden state
        val hidden = lm.hidden(ctx)

        // Get a random dOut of size e (same size as trunk output)
        val dOut = FloatArray(lm.dim) { (rng.nextFloat() - 0.5f) * 0.1f }

        // Compute backprop gradient
        val g = lm.backpropToInput(hidden, dOut)

        // Get the layers through trunkLayers() accessor
        val layers = lm.trunkLayers()
        assertEquals(2, layers.size)

        // Implement a tiny float forward in the test that reads layers
        // The input x is the concatenated context embedding rows (hidden.acts[0])
        // which is exactly k * dim = 3 * 8 = 24 floats
        val x = FloatArray(lm.k * lm.dim) { idx ->
            val slot = idx / lm.dim
            val within = idx % lm.dim
            hidden.acts[0][idx]
        }

        // Forward function for the trunk
        fun forward(input: FloatArray): FloatArray {
            var current = input
            for (layer in layers) {
                val h = FloatArray(layer.outDim)
                for (o in 0 until layer.outDim) {
                    var acc = layer.b[o]
                    val wRow = o * layer.inDim
                    for (j in 0 until layer.inDim) acc += layer.w[wRow + j] * current[j]
                    h[o] = if (acc > 0f) acc else 0f
                }
                current = h
            }
            return current
        }

        // For 5 random input coordinates j, check finite difference
        val eps = 1e-3f
        val e = lm.dim
        val k = lm.k
        val inputSize = k * e

        repeat(5) {
            val j = rng.nextInt(inputSize)
            val xPlus = FloatArray(x.size) { x[it] }
            val xMinus = FloatArray(x.size) { x[it] }
            xPlus[j] += eps
            xMinus[j] -= eps

            val yPlus = forward(xPlus)
            val yMinus = forward(xMinus)

            // Check if ReLU activation pattern changed (skip if so)
            var patternChanged = false
            // Reconstruct intermediate activations to check pattern
            var currentPlus = xPlus
            var currentMinus = xMinus
            for (layer in layers) {
                val hPlus = FloatArray(layer.outDim)
                val hMinus = FloatArray(layer.outDim)
                for (o in 0 until layer.outDim) {
                    var accPlus = layer.b[o]
                    var accMinus = layer.b[o]
                    val wRow = o * layer.inDim
                    for (jj in 0 until layer.inDim) {
                        accPlus += layer.w[wRow + jj] * currentPlus[jj]
                        accMinus += layer.w[wRow + jj] * currentMinus[jj]
                    }
                    hPlus[o] = if (accPlus > 0f) accPlus else 0f
                    hMinus[o] = if (accMinus > 0f) accMinus else 0f
                }
                // Check if any unit crossed zero (changed activation)
                for (o in 0 until layer.outDim) {
                    if ((hPlus[o] > 0f) != (hMinus[o] > 0f)) {
                        patternChanged = true
                        break
                    }
                }
                currentPlus = hPlus
                currentMinus = hMinus
                if (patternChanged) break
            }

            if (patternChanged) {
                // Skip this coordinate; pick another
                repeat(10) {
                    val j2 = rng.nextInt(inputSize)
                    if (j2 != j) {
                        val xPlus2 = FloatArray(x.size) { x[it] }
                        val xMinus2 = FloatArray(x.size) { x[it] }
                        xPlus2[j2] += eps
                        xMinus2[j2] -= eps

                        val yPlus2 = forward(xPlus2)
                        val yMinus2 = forward(xMinus2)

                        // Check pattern change again
                        var patternChanged2 = false
                        var currentPlus2 = xPlus2
                        var currentMinus2 = xMinus2
                        for (layer in layers) {
                            val hPlus = FloatArray(layer.outDim)
                            val hMinus = FloatArray(layer.outDim)
                            for (o in 0 until layer.outDim) {
                                var accPlus = layer.b[o]
                                var accMinus = layer.b[o]
                                val wRow = o * layer.inDim
                                for (jj in 0 until layer.inDim) {
                                    accPlus += layer.w[wRow + jj] * currentPlus2[jj]
                                    accMinus += layer.w[wRow + jj] * currentMinus2[jj]
                                }
                                hPlus[o] = if (accPlus > 0f) accPlus else 0f
                                hMinus[o] = if (accMinus > 0f) accMinus else 0f
                            }
                            for (o in 0 until layer.outDim) {
                                if ((hPlus[o] > 0f) != (hMinus[o] > 0f)) {
                                    patternChanged2 = true
                                    break
                                }
                            }
                            currentPlus2 = hPlus
                            currentMinus2 = hMinus
                            if (patternChanged2) break
                        }

                        if (!patternChanged2) {
                            // Finite difference of dot(forward(x), dOut)
                            val fdDot = (yPlus2 dot dOut - yMinus2 dot dOut) / (2 * eps)
                            assertEquals("finite diff at $j2", g[j2], fdDot, 1e-2f)
                            return@repeat
                        }
                    }
                }
                // If we still can't find a good coordinate, skip
                return@repeat
            }

            // Finite difference: dot((f(x+eps) - f(x-eps)), dOut) / (2*eps)
            val fdDot = (yPlus dot dOut - yMinus dot dOut) / (2 * eps)
            assertEquals("finite diff at $j", g[j], fdDot, 1e-2f)
        }
    }

    private infix fun FloatArray.dot(other: FloatArray): Float {
        require(this.size == other.size) { "Arrays must have same size" }
        var sum = 0f
        for (i in this.indices) sum += this[i] * other[i]
        return sum
    }
}
