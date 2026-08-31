package com.aosmith.type.dict

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Frequency-ordered word list with a trie for prefix queries.
 *
 * Pure JVM on purpose: everything here is exercised by plain unit tests. Android only
 * appears in the [load] factory.
 *
 * Used for: deciding whether a finished word needs the model at all, instant suggestions
 * and completions while the model thinks, and the adaptive-key weights.
 */
class Dictionary(words: Sequence<String>) {

    class Node {
        val children = HashMap<Char, Node>(4)
        var terminal = false

        /** Sum of frequency weights of every word under this node; drives adaptive keys. */
        var weight = 0f
    }

    private val rank = HashMap<String, Int>(80_000)
    private val root = Node()
    private val byLength = HashMap<Int, ArrayList<String>>()
    private val byId = ArrayList<String>(60_000)

    init {
        var i = 0
        for (line in words) {
            val w = line.trim()
            if (w.isEmpty() || rank.containsKey(w)) continue
            rank[w] = i
            byId += w
            byLength.getOrPut(w.length) { ArrayList() } += w
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

    /** Stable id of [word] (its line number in the word list), or -1. Pairs with [Bigrams]. */
    fun idOf(word: String): Int = rank[word.lowercase()] ?: -1

    fun wordOf(id: Int): String? = byId.getOrNull(id)

    /** True when some dictionary word starts with [prefix]. */
    fun hasPrefix(prefix: String): Boolean {
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return false
        return true
    }

    /**
     * All dictionary words starting with [prefix], best first, but only when they collapse to
     * at most [limit] word families; otherwise empty. A family is a stem plus its plural and
     * possessive variants, represented by its most frequent member. Powers word-key morphing.
     */
    fun completions(prefix: String, limit: Int): List<String> {
        if (prefix.isEmpty()) return emptyList()
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return emptyList()
        val rawCap = limit * 3 + 3
        val found = ArrayList<String>(rawCap + 1)
        val sb = StringBuilder(prefix.lowercase())
        fun walk(n: Node): Boolean { // returns false when over the raw cap
            if (n.terminal) {
                found += sb.toString()
                if (found.size > rawCap) return false
            }
            for ((c, child) in n.children) {
                sb.append(c)
                val ok = walk(child)
                sb.setLength(sb.length - 1)
                if (!ok) return false
            }
            return true
        }
        if (!walk(node)) return emptyList()
        val families = found.groupBy(::stem)
        if (families.size > limit) return emptyList()
        return families.values
            .map { members -> members.minBy { rank[it] ?: Int.MAX_VALUE } }
            .sortedBy { rank[it] ?: Int.MAX_VALUE }
            .map { matchCase(prefix, it) }
    }

    /**
     * The most frequent words starting with [prefix], regardless of how many there are.
     * Mid-word prediction: "typi" gives "typing", "typical", ...
     */
    fun predictions(prefix: String, max: Int): List<String> {
        if (prefix.isEmpty()) return emptyList()
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return emptyList()
        val found = ArrayList<String>(32)
        val sb = StringBuilder(prefix.lowercase())
        // Heaviest branches first, stop early: frequent words surface without a full walk.
        fun walk(n: Node) {
            if (found.size >= 25) return
            if (n.terminal) found += sb.toString()
            for ((c, child) in n.children.entries.sortedByDescending { it.value.weight }) {
                if (found.size >= 25) return
                sb.append(c)
                walk(child)
                sb.setLength(sb.length - 1)
            }
        }
        walk(node)
        return found.sortedBy { rank[it] ?: Int.MAX_VALUE }.take(max).map { matchCase(prefix, it) }
    }

    /** Letters that can follow [prefix] in some dictionary word, weighted by frequency (sums to 1). */
    fun nextLetters(prefix: String): Map<Char, Float> {
        var node = root
        for (c in prefix.lowercase()) node = node.children[c] ?: return emptyMap()
        if (node.children.isEmpty()) return emptyMap()
        val total = node.children.values.sumOf { it.weight.toDouble() }.toFloat().coerceAtLeast(1e-6f)
        return node.children.mapValues { it.value.weight / total }
    }

    /**
     * Up to [max] known words within a small edit distance of [word], best first.
     *
     * Scans the words whose length is within two of the input with a bounded
     * Damerau-Levenshtein distance: a few thousand cheap comparisons, no per-candidate
     * allocation.
     */
    fun suggest(word: String, max: Int = 3): List<String> {
        val w = word.lowercase()
        if (w.isEmpty()) return emptyList()
        val maxDistance = if (w.length <= 4) 1 else 2
        val scored = ArrayList<Pair<String, Int>>()
        val buf = DistanceBuffers(w.length + 1 + maxDistance)
        for (len in (w.length - maxDistance)..(w.length + maxDistance)) {
            val bucket = byLength[len] ?: continue
            for (candidate in bucket) {
                val d = boundedDistance(w, candidate, maxDistance, buf)
                if (d > maxDistance) continue
                val r = rank[candidate] ?: continue
                // Distance dominates; frequency breaks ties. A first-letter match is a mild bonus.
                val firstBonus = if (candidate[0] == w[0]) 0 else 3000
                scored += candidate to (d * 100_000 + r + firstBonus)
            }
        }
        scored.sortBy { it.second }
        return scored.take(max).map { matchCase(word, it.first) }
    }

    private class DistanceBuffers(n: Int) {
        var prev2 = IntArray(n)
        var prev = IntArray(n)
        var cur = IntArray(n)
    }

    /** Optimal-string-alignment distance with an early exit once every cell exceeds [limit]. */
    private fun boundedDistance(a: String, b: String, limit: Int, buf: DistanceBuffers): Int {
        val m = a.length
        val n = b.length
        if (kotlin.math.abs(m - n) > limit) return limit + 1
        var prev2 = buf.prev2
        var prev = buf.prev
        var cur = buf.cur
        for (i in 0..m) prev[i] = i
        for (j in 1..n) {
            cur[0] = j
            var rowMin = cur[0]
            for (i in 1..m) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                var v = minOf(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    v = minOf(v, prev2[i - 2] + 1)
                }
                cur[i] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > limit) return limit + 1
            val t = prev2
            prev2 = prev
            prev = cur
            cur = t
        }
        return prev[m]
    }

    companion object {

        /** "restaurant", "restaurants" and "restaurant's" share one stem. */
        fun stem(w: String): String =
            w.removeSuffix("'s").removeSuffix("s'").let { if (it.length > 3) it.removeSuffix("s") else it }

        fun load(context: android.content.Context, asset: String = "en_words.txt"): Dictionary {
            val t0 = System.currentTimeMillis()
            val dict = fromStream(context.assets.open(asset))
            android.util.Log.i("Dictionary", "loaded ${dict.size} words in ${System.currentTimeMillis() - t0} ms")
            return dict
        }

        fun fromStream(input: InputStream): Dictionary =
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines { Dictionary(it) }

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
            original.firstOrNull()?.isUpperCase() == true -> replacement.replaceFirstChar { it.uppercaseChar() }
            else -> replacement.lowercase()
        }
    }
}
