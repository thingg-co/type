package com.aosmith.type.dict

import java.io.DataInputStream
import java.io.IOException
import java.io.InputStream
import kotlin.math.abs
import kotlin.math.max

/**
 * The next-word network: last K word ids -> tied int8 embeddings -> ReLU hidden -> logits
 * over the vocabulary (see tools/nn/train.py for the format and training).
 *
 * The split of labor: this class does the small dense math (context embedding, hidden
 * layer, per-candidate logits) in Kotlin, exactly mirroring the exported quantized
 * arithmetic so it is unit-testable against golden vectors. The one heavy step, scoring
 * the whole vocabulary for top-k next words, goes through the native matvec in the llama
 * JNI lib when available and falls back to the same loop in Kotlin.
 *
 * Ids are en_words.txt line numbers; BOS = V-2 pads short contexts, UNK = V-1 stands in
 * for out-of-vocabulary context words.
 */
class NeuralLm private constructor(
    val vocab: Int,
    val k: Int,
    val dim: Int,
    private val emb: ByteArray,      // vocab x dim, int8
    private val scale: FloatArray,   // vocab
    private val w1: FloatArray,      // dim x (k*dim), row-major
    private val b1: FloatArray,      // dim
    private val bout: FloatArray,    // vocab
    private val nativeTopK: ((ByteArray, Float, Int) -> IntArray?)? = null,
) {
    val bos: Int get() = vocab - 2
    val unk: Int get() = vocab - 1

    /** Sparse user-taught delta applied on top of the frozen base; see [Personalizer]. */
    @Volatile
    var personal: Personalizer? = null

    class Hidden(val q: ByteArray, val scale: Float, val f: FloatArray)

    internal fun embRowInto(id: Int, out: FloatArray) {
        val row = id * dim
        val s = scale[id]
        for (j in 0 until dim) out[j] = emb[row + j] * s
    }

    internal fun trunkW(): FloatArray = w1

    internal fun trunkB(): FloatArray = b1

    /** Left-pads with BOS, maps unknown ids to UNK, and runs the dense trunk. */
    fun hidden(contextIds: List<Int>): Hidden {
        val ctx = IntArray(k)
        for (i in 0 until k) {
            val idx = contextIds.size - k + i
            val id = if (idx < 0) bos else contextIds[idx]
            ctx[i] = if (id in 0 until vocab) id else unk
        }
        val x = FloatArray(k * dim)
        for (i in 0 until k) {
            val row = ctx[i] * dim
            val s = scale[ctx[i]]
            for (j in 0 until dim) x[i * dim + j] = emb[row + j] * s
            personal?.inputDelta(ctx[i])?.let { d ->
                for (j in 0 until dim) x[i * dim + j] += d[j]
            }
        }
        val h = FloatArray(dim)
        var absMax = 1e-8f
        for (o in 0 until dim) {
            var acc = b1[o]
            val wRow = o * k * dim
            for (j in 0 until k * dim) acc += w1[wRow + j] * x[j]
            val r = if (acc > 0f) acc else 0f
            h[o] = r
            if (r > absMax) absMax = r
        }
        val hs = absMax / 127f
        val q = ByteArray(dim)
        for (o in 0 until dim) q[o] = (h[o] / hs + 0.5f).toInt().coerceIn(-127, 127).toByte()
        return Hidden(q, hs, h)
    }

    /** Logit of word [id] given a computed [hidden] state, personal delta included. */
    fun logit(id: Int, hidden: Hidden): Float {
        val row = id * dim
        var acc = 0
        for (j in 0 until dim) acc += emb[row + j] * hidden.q[j]
        var v = acc * scale[id] * hidden.scale + bout[id]
        personal?.let { p ->
            p.outputDelta(id)?.let { d ->
                var s = 0f
                for (j in 0 until dim) s += d[j] * hidden.f[j]
                v += s
            }
            v += p.bias(id)
        }
        return v
    }

    /** Logits for a small candidate set (mid-word reranking). Ids beyond this network's
     * vocabulary (words appended to the list after training) score as UNK. */
    fun scoreCandidates(contextIds: List<Int>, candidateIds: IntArray): FloatArray {
        val h = hidden(contextIds)
        return FloatArray(candidateIds.size) {
            val id = candidateIds[it]
            logit(if (id in 0 until vocab) id else unk, h)
        }
    }

    /** Best next-word ids over the whole vocabulary, specials excluded. */
    fun topNext(contextIds: List<Int>, k: Int): List<Int> {
        val h = hidden(contextIds)
        val p = personal
        nativeTopK?.invoke(h.q, h.scale, if (p == null) k else maxOf(k * 4, 12))?.let { ids ->
            val basePool = ids.filter { it < bos }
            if (p == null) return basePool.take(k)
            // Learned words can outrank the base list even when the frozen model ignores
            // them; the native pass cannot see the deltas, so union and rescore here.
            val pool = (basePool + p.learnedIds().filter { it < bos }).distinct()
            return pool.sortedByDescending { logit(it, h) }.take(k)
        }
        // Kotlin fallback: one pass, keep the k best.
        val bestIds = IntArray(k) { -1 }
        val bestVals = FloatArray(k) { Float.NEGATIVE_INFINITY }
        for (v in 0 until bos) {
            val s = logit(v, h)
            if (s > bestVals[k - 1]) {
                var i = k - 1
                while (i > 0 && s > bestVals[i - 1]) {
                    bestVals[i] = bestVals[i - 1]
                    bestIds[i] = bestIds[i - 1]
                    i--
                }
                bestVals[i] = s
                bestIds[i] = v
            }
        }
        return bestIds.filter { it >= 0 }
    }

    companion object {
        fun load(input: InputStream, nativeTopK: ((ByteArray, Float, Int) -> IntArray?)? = null): NeuralLm =
            DataInputStream(input.buffered(1 shl 16)).use { d ->
                val magic = ByteArray(4)
                d.readFully(magic)
                if (!magic.contentEquals("TNW1".toByteArray())) throw IOException("bad nextword file")
                val v = d.readInt()
                val k = d.readInt()
                val e = d.readInt()
                val emb = ByteArray(v * e).also { d.readFully(it) }
                val scale = FloatArray(v) { d.readFloat() }
                val w1 = FloatArray(e * k * e) { d.readFloat() }
                val b1 = FloatArray(e) { d.readFloat() }
                val bout = FloatArray(v) { d.readFloat() }
                NeuralLm(v, k, e, emb, scale, w1, b1, bout, nativeTopK)
            }

        fun load(context: android.content.Context, asset: String = "en_nextword.bin"): NeuralLm {
            val t0 = System.currentTimeMillis()
            val m = load(
                context.assets.open(asset),
                nativeTopK = { q, s, k -> com.aosmith.type.llm.LlamaNative.nnTopK(q, s, k) },
            )
            android.util.Log.i("NeuralLm", "loaded V=${m.vocab} E=${m.dim} in ${System.currentTimeMillis() - t0} ms")
            return m
        }
    }
}
