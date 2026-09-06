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
    private val recurrent: Recurrent? = null, // TNW5: a GRU trunk instead of the dense layers
) {
    /** One dense trunk layer, row-major (out x in). */
    class Layer(val outDim: Int, val inDim: Int, val w: FloatArray, val b: FloatArray)

    /** One GRU layer, PyTorch layout: gate rows stacked (r, z, n), each [hidden] rows, row-major. */
    class GruLayer(val inDim: Int, val hidden: Int, val wIh: FloatArray, val wHh: FloatArray, val bIh: FloatArray, val bHh: FloatArray)

    /**
     * A recurrent trunk (TNW5): the sentence so far, one word at a time, through [layers], then a
     * linear map back to the embedding width so the vocabulary product is the same int8 dot
     * product as for the dense trunk.
     */
    class Recurrent(val layers: List<GruLayer>, val projW: FloatArray, val projB: FloatArray)

    /** True for a recurrent (TNW5) network; its context is the whole sentence, not a window. */
    val isRecurrent: Boolean get() = recurrent != null

    /**
     * How many previous words the keyboard should hand this network: the whole sentence for a
     * recurrent trunk (its cache only pays off on a growing prefix), five for the dense one, which
     * was calibrated with five even though its window is wider.
     */
    val contextWords: Int get() = if (recurrent != null) k else DENSE_CONTEXT_WORDS

    /** Called after each recurrent pass with (steps computed, prefix length, microseconds); for a debug log. */
    var onRecurrentPass: ((Int, Int, Long) -> Unit)? = null

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

    /** Left-pads with BOS, maps unknown ids to UNK, and runs the trunk. */
    fun hidden(contextIds: List<Int>): Hidden {
        recurrent?.let { return recurrentHidden(it, contextIds) }
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

    // ---- recurrent trunk -----------------------------------------------------------------

    private val cacheLock = Any()
    private var cachedIds = IntArray(0)                          // [BOS, w1, ...] the trajectory below has consumed
    private var cachedTrajectory: Array<Array<FloatArray>> = emptyArray()   // [t] = per-layer states after ids[0..t]

    /**
     * Runs the GRU over BOS plus the context (its last [k] words, leading BOS padding dropped).
     * The keyboard scores several variants of one prefix in a row, from more than one thread: the
     * base context, one-word extensions for the confusable checks, then the next word. So the states
     * after every position of the last prefix are kept, and any request shares the longest common
     * prefix with them: a shorter context is free, an extension costs one step per new word, and
     * only a new sentence pays a full pass.
     */
    private fun recurrentHidden(r: Recurrent, contextIds: List<Int>): Hidden {
        val words = contextIds.dropWhile { it == bos }.let { if (it.size > k) it.subList(it.size - k, it.size) else it }
        val ids = IntArray(words.size + 1)
        ids[0] = bos
        for (i in words.indices) ids[i + 1] = words[i].let { if (it in 0 until vocab) it else unk }
        val nLayers = r.layers.size
        val trajectory = arrayOfNulls<Array<FloatArray>>(ids.size)
        var from = 0
        synchronized(cacheLock) {
            var m = 0
            while (m < cachedIds.size && m < ids.size && cachedIds[m] == ids[m]) m++
            for (t in 0 until m) trajectory[t] = cachedTrajectory[t]      // states are never mutated once stored
            from = m
        }
        var states: Array<FloatArray> = if (from > 0) trajectory[from - 1]!! else Array(nLayers) { FloatArray(r.layers[it].hidden) }
        val tStart = System.nanoTime()
        val x0 = FloatArray(dim)
        for (t in from until ids.size) {
            embRowInto(ids[t], x0)
            personal?.inputDelta(ids[t])?.let { d -> for (j in 0 until dim) x0[j] += d[j] }
            var x = x0
            val next = Array(nLayers) { FloatArray(0) }
            for (l in 0 until nLayers) {
                next[l] = gruStep(r.layers[l], x, states[l])
                x = next[l]
            }
            states = next
            trajectory[t] = next
        }
        synchronized(cacheLock) {
            // keep the longer of the two when one is a prefix of the other; otherwise the newest wins
            val extendsCache = cachedIds.size <= ids.size && (0 until cachedIds.size).all { cachedIds[it] == ids[it] }
            val cacheExtends = !extendsCache && ids.size < cachedIds.size && (0 until ids.size).all { cachedIds[it] == ids[it] }
            if (!cacheExtends) {
                cachedIds = ids
                @Suppress("UNCHECKED_CAST")
                cachedTrajectory = trajectory as Array<Array<FloatArray>>
            }
        }
        val top = states[nLayers - 1]
        val hTop = r.layers[nLayers - 1].hidden
        val v = FloatArray(dim)
        for (o in 0 until dim) {
            var acc = r.projB[o]
            val row = o * hTop
            for (j in 0 until hTop) acc += r.projW[row + j] * top[j]
            v[o] = acc
        }
        // The projection is signed, unlike a ReLU output: scale by the absolute maximum, round to nearest.
        var absMax = 1e-8f
        for (o in 0 until dim) { val a = kotlin.math.abs(v[o]); if (a > absMax) absMax = a }
        val vs = absMax / 127f
        val q = ByteArray(dim)
        for (o in 0 until dim) q[o] = Math.round(v[o] / vs).coerceIn(-127, 127).toByte()
        onRecurrentPass?.invoke(ids.size - from, ids.size, (System.nanoTime() - tStart) / 1000)
        return Hidden(q, vs, v, emptyList())
    }

    private fun gruStep(L: GruLayer, x: FloatArray, h: FloatArray): FloatArray {
        val H = L.hidden
        val gi = FloatArray(3 * H)
        val gh = FloatArray(3 * H)
        for (o in 0 until 3 * H) {
            var a = L.bIh[o]
            val rowI = o * L.inDim
            for (j in 0 until L.inDim) a += L.wIh[rowI + j] * x[j]
            gi[o] = a
            var b = L.bHh[o]
            val rowH = o * H
            for (j in 0 until H) b += L.wHh[rowH + j] * h[j]
            gh[o] = b
        }
        val out = FloatArray(H)
        for (j in 0 until H) {
            val rGate = sigmoid(gi[j] + gh[j])
            val zGate = sigmoid(gi[H + j] + gh[H + j])
            val n = kotlin.math.tanh(gi[2 * H + j] + rGate * gh[2 * H + j])
            out[j] = (1f - zGate) * n + zGate * h[j]
        }
        return out
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + kotlin.math.exp(-x))

    /** Sends a gradient on the trunk output back to the context rows (k*dim values), through every
     * layer's ReLU and frozen weights. Used by the personaliser's input-side learning. Empty for a
     * recurrent trunk, which learns on the output side only. */
    internal fun backpropToInput(hidden: Hidden, dOut: FloatArray): FloatArray {
        if (recurrent != null) return FloatArray(0)
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
                // end-of-sentence target. TNW3: any number of dense layers, BOS trained. TNW5: a GRU
                // trunk over the whole sentence, then a linear map back to the embedding width.
                if (tag == "TNW5") return loadRecurrent(d, nativeTopK)
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

        /** Sentences in training were cut at this many words; the context passed in is trimmed to it. */
        private const val RECURRENT_WINDOW = 40
        private const val DENSE_CONTEXT_WORDS = 5

        private fun loadRecurrent(d: DataInputStream, nativeTopK: ((ByteArray, Float, Int) -> IntArray?)?): NeuralLm {
            val v = d.readInt()
            val e = d.readInt()
            val h = d.readInt()
            val nLayers = d.readInt()
            val emb = ByteArray(v * e).also { d.readFully(it) }
            val scale = FloatArray(v) { d.readFloat() }
            val layers = ArrayList<GruLayer>(nLayers)
            for (i in 0 until nLayers) {
                val inDim = if (i == 0) e else h
                val wIh = FloatArray(3 * h * inDim) { d.readFloat() }
                val wHh = FloatArray(3 * h * h) { d.readFloat() }
                val bIh = FloatArray(3 * h) { d.readFloat() }
                val bHh = FloatArray(3 * h) { d.readFloat() }
                layers.add(GruLayer(inDim, h, wIh, wHh, bIh, bHh))
            }
            val projW = FloatArray(e * h) { d.readFloat() }
            val projB = FloatArray(e) { d.readFloat() }
            val bout = FloatArray(v) { d.readFloat() }
            return NeuralLm(v, RECURRENT_WINDOW, e, emb, scale, emptyList(), bout, nativeTopK, Recurrent(layers, projW, projB))
                .also { it.eosTrained = true }
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
