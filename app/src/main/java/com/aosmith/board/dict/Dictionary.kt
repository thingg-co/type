package com.aosmith.board.dict

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Frequency-ordered English word list with a trie for prefix queries.
 *
 * Used for three things: deciding whether a finished word needs the model at all, cheap
 * edit-distance suggestions while the model thinks, and the adaptive-key weights.
 */
class Dictionary private constructor(
    private val rank: HashMap<String, Int>,
    private val root: Node,
) {
    class Node {
        val children = HashMap<Char, Node>(4)
        var terminal = false

        /** Sum of frequency weights of every word under this node; drives adaptive keys. */
        var weight = 0f
    }

    val size: Int get() = rank.size

    fun isKnown(word: String): Boolean {
        if (word.isEmpty()) return false
        val w = word.lowercase()
        if (rank.containsKey(w)) return true
        if (w.endsWith("'s") && rank.containsKey(w.dropLast(2))) return true
        if (w.endsWith("s") && w.length > 3 && rank.containsKey(w.dropLast(1))) return true
        return false
    }

    fun rankOf(word: String): Int = rank[word.lowercase()] ?: Int.MAX_VALUE

    /** True when some dictionary word starts with [prefix]. */
    fun hasPrefix(prefix: String): Boolean {
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return false
        return true
    }

    /** Letters that can follow [prefix] in some dictionary word, weighted by frequency (sums to 1). */
    fun nextLetters(prefix: String): Map<Char, Float> {
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return emptyMap()
        if (node.children.isEmpty()) return emptyMap()
        val total = node.children.values.sumOf { it.weight.toDouble() }.toFloat().coerceAtLeast(1e-6f)
        return node.children.mapValues { it.value.weight / total }
    }

    /** Up to [max] known words within a small edit distance of [word], best first. */
    fun suggest(word: String, max: Int = 3): List<String> {
        val w = word.lowercase()
        if (w.isEmpty()) return emptyList()
        val scored = HashMap<String, Int>()
        for (c in edits1(w)) rank[c]?.let { scored[c] = it }
        if (scored.size < max && w.length in 4..10) {
            for (e1 in edits1(w)) for (c in edits1(e1)) rank[c]?.let { r -> scored.merge(c, r + 20_000, ::minOf) }
        }
        return scored.entries.sortedBy { it.value }.take(max).map { matchCase(word, it.key) }
    }

    private fun edits1(w: String): Sequence<String> = sequence {
        val n = w.length
        for (i in 0..n) {
            val left = w.substring(0, i)
            val right = w.substring(i)
            if (right.isNotEmpty()) yield(left + right.substring(1))
            if (right.length > 1) yield(left + right[1] + right[0] + right.substring(2))
            for (c in ALPHABET) {
                if (right.isNotEmpty()) yield(left + c + right.substring(1))
                yield(left + c + right)
            }
        }
    }

    companion object {
        private const val TAG = "Dictionary"
        private const val ALPHABET = "abcdefghijklmnopqrstuvwxyz'"

        fun load(context: Context, asset: String = "en_words.txt"): Dictionary {
            val t0 = System.currentTimeMillis()
            val rank = HashMap<String, Int>(80_000)
            val root = Node()
            BufferedReader(InputStreamReader(context.assets.open(asset), Charsets.UTF_8)).useLines { lines ->
                var i = 0
                for (line in lines) {
                    val w = line.trim()
                    if (w.isEmpty()) continue
                    rank[w] = i
                    // Zipf-like weight: rank 0 counts ~1.0, rank 50k counts ~0.02.
                    val weight = 1f / (1f + i / 1000f)
                    var node = root
                    node.weight += weight
                    for (c in w) {
                        node = node.children.getOrPut(c) { Node() }
                        node.weight += weight
                    }
                    node.terminal = true
                    i++
                }
            }
            Log.i(TAG, "loaded ${rank.size} words in ${System.currentTimeMillis() - t0} ms")
            return Dictionary(rank, root)
        }

        /** Damerau-Levenshtein distance (optimal string alignment). */
        fun editDistance(a: String, b: String): Int {
            if (a == b) return 0
            val m = a.length
            val n = b.length
            if (m == 0) return n
            if (n == 0) return m
            val d = Array(m + 1) { IntArray(n + 1) }
            for (i in 0..m) d[i][0] = i
            for (j in 0..n) d[0][j] = j
            for (i in 1..m) for (j in 1..n) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                d[i][j] = minOf(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    d[i][j] = minOf(d[i][j], d[i - 2][j - 2] + 1)
                }
            }
            return d[m][n]
        }

        /** Applies the capitalisation pattern of [original] to [replacement]. */
        fun matchCase(original: String, replacement: String): String = when {
            original.length > 1 && original.all { !it.isLetter() || it.isUpperCase() } -> replacement.uppercase()
            original.first().isUpperCase() -> replacement.replaceFirstChar { it.uppercaseChar() }
            else -> replacement.lowercase()
        }
    }
}
