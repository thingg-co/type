package com.aosmith.type.dict

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

/**
 * Per-user fat-finger profile: which letter-for-letter slips this user's accepted
 * corrections keep revealing. Counts are directional (typed 'n' when 'm' was meant) and
 * feed [KeyNeighbors.substitutionCost] as a multiplier, so the whole correction stack —
 * dictionary suggestions, slip candidates, model-output validation — gradually prefers
 * the explanations that match these hands. An undone correction takes its evidence back
 * out at triple weight.
 *
 * Bounded on purpose: counts saturate, learned discounts have floors, and a pair outside
 * the physical adjacency can never become cheaper than one inside it, so the geometry
 * stays the prior and the profile only bends it.
 */
class SlipProfile {

    private val counts = java.util.concurrent.ConcurrentHashMap<Int, Int>()

    private fun key(typed: Char, meant: Char): Int = typed.code * 1000 + meant.code

    /** Learns from an accepted correction: for each aligned substitution, typed -> meant. */
    fun recordCorrection(typed: String, corrected: String) {
        forAlignedSubstitution(typed, corrected) { t, m ->
            counts.merge(key(t, m), 1) { old, _ -> minOf(old + 1, MAX_COUNT) }
        }
    }

    /** An undone correction was wrong: remove its evidence at triple weight. */
    fun recordRevert(typed: String, corrected: String) {
        forAlignedSubstitution(typed, corrected) { t, m ->
            counts.compute(key(t, m)) { _, old ->
                val v = (old ?: 0) - 3
                if (v <= 0) null else v
            }
        }
    }

    /**
     * Only clean single-substitution pairs of equal length train the profile: they are
     * the unambiguous fat-finger events. Everything else (insertions, transpositions,
     * multi-edit rewrites) carries no per-key signal worth the noise.
     */
    private inline fun forAlignedSubstitution(typed: String, corrected: String, learn: (Char, Char) -> Unit) {
        val a = typed.lowercase()
        val b = corrected.lowercase()
        if (a.length != b.length) return
        var at = -1
        for (i in a.indices) {
            if (a[i] != b[i]) {
                if (at >= 0) return // second difference: not a single slip
                at = i
            }
        }
        if (at < 0) return
        val t = a[at]
        val m = b[at]
        if (t !in 'a'..'z' || m !in 'a'..'z') return
        learn(t, m)
    }

    /**
     * Personal substitution cost for typing [typed] when [meant] was intended, given the
     * geometric [base] cost (ADJACENT_COST or 1). Evidence discounts down to a floor:
     * adjacent pairs to [MIN_ADJACENT], distant pairs to [MIN_DISTANT] — still above the
     * adjacent base, so personal history never outranks physics entirely.
     */
    fun cost(typed: Char, meant: Char, base: Float): Float {
        val n = counts[key(typed, meant)] ?: return base
        val shrink = 1f - DISCOUNT * (n.toFloat() / MAX_COUNT)
        val floor = if (base <= KeyNeighbors.ADJACENT_COST) MIN_ADJACENT else MIN_DISTANT
        return (base * shrink).coerceAtLeast(floor)
    }

    fun size(): Int = counts.size

    // ---- persistence -----------------------------------------------------------------

    fun save(out: OutputStream) = DataOutputStream(out.buffered()).use { d ->
        d.write(MAGIC)
        d.writeInt(counts.size)
        for ((k, v) in counts) {
            d.writeInt(k)
            d.writeInt(v)
        }
    }

    fun load(input: InputStream) = DataInputStream(input.buffered()).use { d ->
        val magic = ByteArray(4)
        d.readFully(magic)
        if (!magic.contentEquals(MAGIC)) throw IOException("bad slip profile file")
        counts.clear()
        repeat(d.readInt()) {
            val k = d.readInt()
            counts[k] = d.readInt().coerceIn(1, MAX_COUNT)
        }
    }

    fun clear() = counts.clear()

    companion object {
        private val MAGIC = "TSP1".toByteArray()
        private const val MAX_COUNT = 20
        private const val DISCOUNT = 0.5f
        private const val MIN_ADJACENT = 0.25f
        private const val MIN_DISTANT = 0.6f
    }
}
