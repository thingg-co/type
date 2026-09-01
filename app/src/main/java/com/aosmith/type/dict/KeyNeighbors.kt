package com.aosmith.type.dict

import kotlin.math.abs

/**
 * Physical adjacency on the QWERTY layout, for fat-finger-aware costs: substituting a
 * letter for one of its neighbours is a far likelier slip than a distant one.
 *
 * Derived from the same row geometry the keyboard draws (see ime/KeyboardLayouts.kt:
 * rows of 10/9/7 keys with half-key stagger), not hardcoded pairs, so a layout change
 * only needs the row strings updated here.
 */
object KeyNeighbors {

    private val ROWS = listOf(
        "qwertyuiop" to 0.0f,
        "asdfghjkl" to 0.5f,
        "zxcvbnm" to 1.5f,
    )

    private val pos = HashMap<Char, Pair<Float, Float>>().apply {
        ROWS.forEachIndexed { row, (letters, offset) ->
            letters.forEachIndexed { col, c -> put(c, (col + offset) to row.toFloat()) }
        }
    }

    /** True when [a] and [b] are touching keys (including diagonals across the stagger). */
    fun adjacent(a: Char, b: Char): Boolean {
        if (a == b) return false
        val pa = pos[a.lowercaseChar()] ?: return false
        val pb = pos[b.lowercaseChar()] ?: return false
        val dx = abs(pa.first - pb.first)
        val dy = abs(pa.second - pb.second)
        return dy <= 1f && dx <= 1.01f && (dx * dx + dy * dy) <= 2.02f
    }

    private val neighborMap: Map<Char, List<Char>> =
        pos.keys.associateWith { a -> pos.keys.filter { b -> adjacent(a, b) }.sorted() }

    /** Every key touching [c], from the same geometry as [adjacent]. */
    fun neighbors(c: Char): List<Char> = neighborMap[c.lowercaseChar()] ?: emptyList()

    /** Per-user slip profile; when set, learned pairs discount the geometric costs. */
    @Volatile
    var personal: SlipProfile? = null

    /** Substitution cost for typing [a] when [b] was meant: neighbours are cheap slips. */
    fun substitutionCost(a: Char, b: Char): Float {
        if (a == b) return 0f
        val base = if (adjacent(a, b)) ADJACENT_COST else 1f
        return personal?.cost(a.lowercaseChar(), b.lowercaseChar(), base) ?: base
    }

    const val ADJACENT_COST = 0.45f
    const val TRANSPOSE_COST = 0.8f
}
