package com.aosmith.type.dict

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.Random

/**
 * On-device personalization of the next-word network.
 *
 * The base model stays frozen. What the user types trains a sparse delta on top of it:
 * per-word adjustments to the embedding rows in their context and output roles, plus a
 * per-word output bias, learned by SGD with negative sampling. Applied at inference by
 * [NeuralLm] when installed as its `personal` field.
 *
 * Privacy: only in-vocabulary word ids are recorded (never raw text), the data lives in
 * app-private storage, and [clear] wipes everything. The service additionally skips
 * capture in sensitive fields and when an editor sets the no-personalized-learning flag.
 */
class Personalizer(private val base: NeuralLm) {

    private val lock = Any()
    private val inputDelta = java.util.concurrent.ConcurrentHashMap<Int, FloatArray>()
    private val outputDelta = java.util.concurrent.ConcurrentHashMap<Int, FloatArray>()
    private val bias = java.util.concurrent.ConcurrentHashMap<Int, Float>()
    private val samples = ArrayDeque<IntArray>() // [ctx.. , target]
    private val rng = Random(42)
    private var lastSaveAt = 0L

    var lifetimeSamples = 0L
        private set

    // ---- inference hooks (called by NeuralLm on hot paths; must stay allocation-free) --

    fun inputDelta(id: Int): FloatArray? = inputDelta[id]

    fun outputDelta(id: Int): FloatArray? = outputDelta[id]

    fun bias(id: Int): Float = bias[id] ?: 0f

    /** Words the user has taught the model; candidates for next-word suggestions. */
    fun learnedIds(): List<Int> = synchronized(lock) { (outputDelta.keys + bias.keys).toList() }

    // ---- capture ---------------------------------------------------------------------

    fun record(contextIds: List<Int>, target: Int) {
        if (target < 0 || target >= base.bos) return
        synchronized(lock) {
            val s = IntArray(base.k + 1)
            for (i in 0 until base.k) {
                val idx = contextIds.size - base.k + i
                s[i] = if (idx < 0) base.bos else contextIds[idx].coerceIn(0, base.unk)
            }
            s[base.k] = target
            samples.addLast(s)
            while (samples.size > MAX_SAMPLES) samples.removeFirst()
            lifetimeSamples++
        }
    }

    val pendingSamples: Int get() = synchronized(lock) { samples.size }

    // ---- learning --------------------------------------------------------------------

    /** A budgeted burst of SGD; called when the keyboard hides. Saves at most every 30 s. */
    fun trainAndMaybeSave(file: File, steps: Int = 192) {
        synchronized(lock) {
            if (samples.isEmpty()) return
            repeat(minOf(steps, samples.size * 8)) { step(samples[rng.nextInt(samples.size)]) }
            val now = System.currentTimeMillis()
            if (now - lastSaveAt > 30_000) {
                lastSaveAt = now
                runCatching { file.outputStream().use { save(it) } }
            }
        }
    }

    private fun step(sample: IntArray) {
        val e = base.dim
        val k = base.k
        val ctx = sample.copyOfRange(0, k).toList()
        val target = sample[k]

        val hidden = base.hidden(ctx) // includes current deltas via base.personal == this
        val h = hidden.f

        // candidate set: target + negatives sampled from the frequent half of the vocab
        val cands = IntArray(NEGATIVES + 1)
        cands[0] = target
        for (i in 1..NEGATIVES) cands[i] = rng.nextInt(minOf(base.bos, 20_000))

        val logits = FloatArray(cands.size) { base.logit(cands[it], hidden) }
        var maxL = Float.NEGATIVE_INFINITY
        for (l in logits) if (l > maxL) maxL = l
        var sum = 0f
        val p = FloatArray(cands.size)
        for (i in cands.indices) {
            p[i] = kotlin.math.exp((logits[i] - maxL).toDouble()).toFloat()
            sum += p[i]
        }
        val dh = FloatArray(e)
        val row = FloatArray(e)
        for (i in cands.indices) {
            val g = p[i] / sum - (if (i == 0) 1f else 0f) // dL/dlogit
            val id = cands[i]
            base.embRowInto(id, row)
            outputDelta(id)?.let { d -> for (j in 0 until e) row[j] += d[j] }
            // accumulate hidden grad before updating the output side
            for (j in 0 until e) dh[j] += g * row[j]
            val od = outputDelta.getOrPut(id) { FloatArray(e) }
            for (j in 0 until e) od[j] = clamp(od[j] - LR * (g * h[j] + DECAY * od[j]))
            bias[id] = ((bias[id] ?: 0f) - LR * (g + DECAY * (bias[id] ?: 0f))).coerceIn(-BIAS_CLAMP, BIAS_CLAMP)
        }
        // through ReLU (h==0 kills the grad) and the frozen trunk back to the context rows
        val w1 = base.trunkW()
        for (j in 0 until e) if (h[j] <= 0f) dh[j] = 0f
        for (slot in 0 until k) {
            val id = ctx[slot]
            if (id >= base.bos) continue
            val d = inputDelta.getOrPut(id) { FloatArray(e) }
            for (x in 0 until e) {
                var g = 0f
                for (j in 0 until e) g += w1[j * k * e + slot * e + x] * dh[j]
                d[x] = clamp(d[x] - LR * (g + DECAY * d[x]))
            }
        }
    }

    private fun clamp(v: Float) = v.coerceIn(-CLAMP, CLAMP)

    // ---- persistence -----------------------------------------------------------------

    fun save(out: java.io.OutputStream) = DataOutputStream(out.buffered()).use { d ->
        synchronized(lock) {
            d.write(MAGIC)
            d.writeLong(lifetimeSamples)
            d.writeInt(base.dim)
            d.writeInt(base.k)
            fun writeRows(m: java.util.concurrent.ConcurrentHashMap<Int, FloatArray>) {
                d.writeInt(m.size)
                for ((id, v) in m) {
                    d.writeInt(id)
                    for (x in v) d.writeFloat(x)
                }
            }
            writeRows(inputDelta)
            writeRows(outputDelta)
            d.writeInt(bias.size)
            for ((id, v) in bias) {
                d.writeInt(id)
                d.writeFloat(v)
            }
            d.writeInt(samples.size)
            for (s in samples) for (x in s) d.writeShort(x)
        }
    }

    fun load(input: InputStream) = DataInputStream(input.buffered()).use { d ->
        synchronized(lock) {
            val magic = ByteArray(4)
            d.readFully(magic)
            if (!magic.contentEquals(MAGIC)) throw IOException("bad personalization file")
            lifetimeSamples = d.readLong()
            if (d.readInt() != base.dim || d.readInt() != base.k) {
                throw IOException("saved for a different model shape; starting fresh")
            }
            fun readRows(m: java.util.concurrent.ConcurrentHashMap<Int, FloatArray>) {
                m.clear()
                repeat(d.readInt()) {
                    val id = d.readInt()
                    m[id] = FloatArray(base.dim) { d.readFloat() }
                }
            }
            readRows(inputDelta)
            readRows(outputDelta)
            bias.clear()
            repeat(d.readInt()) { bias[d.readInt()] = d.readFloat() }
            samples.clear()
            repeat(d.readInt()) {
                samples.addLast(IntArray(base.k + 1) { d.readUnsignedShort() })
            }
        }
    }

    fun clear() = synchronized(lock) {
        inputDelta.clear()
        outputDelta.clear()
        bias.clear()
        samples.clear()
        lifetimeSamples = 0
    }

    companion object {
        private val MAGIC = "TPL1".toByteArray()
        private const val MAX_SAMPLES = 20_000
        private const val NEGATIVES = 16
        private const val LR = 0.02f
        private const val DECAY = 1e-3f
        private const val CLAMP = 0.6f

        /** The context-free bias gets more room so often-typed words surface in any context. */
        private const val BIAS_CLAMP = 1.5f
    }
}
