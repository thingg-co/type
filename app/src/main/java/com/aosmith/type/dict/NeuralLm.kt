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
    private val layers: List<Layer>, // dense trunk: relu(W x + b) per layer, last one produces `dim` values
    private val bout: FloatArray,    // vocab
    private val nativeTopK: ((ByteArray, Float, Int) -> IntArray?)? = null,
) {
    /** One dense trunk layer, row-major (out x in). */
    class Layer(val outDim: Int, val inDim: Int, val w: FloatArray, val b: FloatArray)

    val bos: Int get() = vocab - 2
    val unk: Int get() = vocab - 1

    /** True when the asset trained BOS as an end-of-sentence target (TNW2+). */
    var eosTrained: Boolean = false
        internal set

    /** Sparse user-taught delta applied on top of the frozen base; see [Personalizer]. */
    @Volatile
    var personal: Personalizer? = null

    /** The trunk output quantized for the int8 dot products, its scale, the float output, and the
     * activations entering each layer (the context rows first), kept so the personaliser can send a
     * gradient back through the trunk. */
    class Hidden(val q: ByteArray, val scale: Float, val f: FloatArray, internal val acts: List<FloatArray>)

    internal fun embRowInto(id: Int, out: FloatArray) {
        val row = id * dim
        val s = scale[id]
        for (j in 0 until dim) out[j] = emb[row + j] * s
    }

    internal fun trunkLayers(): List<Layer> = layers

    /** Left-pads with BOS, maps unknown ids to UNK, and runs the dense trunk. */
    fun hidden(contextIds: List<Int>): Hidden {
        val ctx = IntArray(k)
        for (i in 0 until k) {
            val idx = contextIds.size - k + i
            val id = if (idx < 0) bos else contextIds[idx]
            ctx[i] = if (id in 0 until vocab) id else unk
        }
        var x = FloatArray(k * dim)
        for (i in 0 until k) {
            val row = ctx[i] * dim
            val s = scale[ctx[i]]
            for (j in 0 until dim) x[i * dim + j] = emb[row + j] * s
            personal?.inputDelta(ctx[i])?.let { d ->
                for (j in 0 until dim) x[i * dim + j] += d[j]
            }
        }
        val acts = ArrayList<FloatArray>(layers.size)
        for (L in layers) {
            acts.add(x)
            val h = FloatArray(L.outDim)
            for (o in 0 until L.outDim) {
                var acc = L.b[o]
                val wRow = o * L.inDim
                val w = L.w
                for (j in 0 until L.inDim) acc += w[wRow + j] * x[j]
                h[o] = if (acc > 0f) acc else 0f
            }
            x = h
        }
        val h = x
        var absMax = 1e-8f
        for (o in 0 until dim) if (h[o] > absMax) absMax = h[o]
        val hs = absMax / 127f
        val q = ByteArray(dim)
        for (o in 0 until dim) q[o] = (h[o] / hs + 0.5f).toInt().coerceIn(-127, 127).toByte()
        return Hidden(q, hs, h, acts)
    }

    /** Sends a gradient on the trunk output back to the context rows (k*dim values), through every
     * layer's ReLU and frozen weights. Used by the personaliser's input-side learning. */
    internal fun backpropToInput(hidden: Hidden, dOut: FloatArray): FloatArray {
        var g = dOut
        for (li in layers.indices.reversed()) {
            val L = layers[li]
            val input = hidden.acts[li]
            // the layer's output activation is the next layer's input, or the trunk output for the last
            val out = if (li + 1 < layers.size) hidden.acts[li + 1] else hidden.f
            val gIn = FloatArray(L.inDim)
            for (o in 0 until L.outDim) {
                if (out[o] <= 0f) continue                  // ReLU: dead units pass nothing back
                val go = g[o]
                if (go == 0f) continue
                val wRow = o * L.inDim
                for (j in 0 until L.inDim) gIn[j] += L.w[wRow + j] * go
            }
            g = gIn
            if (input.size != L.inDim) break
        }
        return g
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
                val tag = String(magic)
                // TNW1: one dense layer, BOS never a target. TNW2: same layout, BOS trained as the
                // end-of-sentence target. TNW3: any number of dense layers, BOS trained.
                if (tag != "TNW1" && tag != "TNW2" && tag != "TNW3") throw IOException("bad nextword file")
                val v = d.readInt()
                val k = d.readInt()
                val e = d.readInt()
                val nLayers = if (tag == "TNW3") d.readInt() else 1
                val emb = ByteArray(v * e).also { d.readFully(it) }
                val scale = FloatArray(v) { d.readFloat() }
                val layers = ArrayList<Layer>(nLayers)
                for (i in 0 until nLayers) {
                    val outDim = if (tag == "TNW3") d.readInt() else e
                    val inDim = if (tag == "TNW3") d.readInt() else k * e
                    val w = FloatArray(outDim * inDim) { d.readFloat() }
                    val b = FloatArray(outDim) { d.readFloat() }
                    layers.add(Layer(outDim, inDim, w, b))
                }
                if (layers.last().outDim != e) throw IOException("nextword trunk must end in the embedding width")
                val bout = FloatArray(v) { d.readFloat() }
                NeuralLm(v, k, e, emb, scale, layers, bout, nativeTopK).also { it.eosTrained = tag != "TNW1" }
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
