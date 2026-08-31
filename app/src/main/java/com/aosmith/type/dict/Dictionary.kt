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

    /** Common misspellings present in the word list; see [TypoTable]. */
    @Volatile
    var misspellings: TypoTable? = null

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
        val bad = misspellings
        if (bad?.isMisspelling(w) == true) return false
        if (rank.containsKey(w)) return true
        // The suffix heuristics must not resurrect denylisted stems ("belives" -> "belive").
        if (w.endsWith("'s") && rank.containsKey(w.dropLast(2)) && bad?.isMisspelling(w.dropLast(2)) != true) return true
        if (w.endsWith("s") && w.length > 3 && rank.containsKey(w.dropLast(1)) && bad?.isMisspelling(w.dropLast(1)) != true) return true
        return false
    }

    fun rankOf(word: String): Int = rank[word.lowercase()] ?: Int.MAX_VALUE

    /** Stable id of [word] (its line number in the word list), or -1. Pairs with [Bigrams]. */
    fun idOf(word: String): Int = rank[word.lowercase()] ?: -1

    fun wordOf(id: Int): String? = byId.getOrNull(id)

    fun isMisspelledWord(word: String): Boolean = misspellings?.isMisspelling(word) == true

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
        // Best-first over the whole frontier, not depth-first: a plain DFS with a cap can
        // exhaust its budget inside the first heavy branch ("har...") and never reach the
        // sibling that holds the frequent words ("have"). Found the hard way.
        class Entry(val node: Node, val text: String)

        val queue = java.util.PriorityQueue<Entry>(16, compareByDescending { it.node.weight })
        queue.add(Entry(node, prefix.lowercase()))
        val found = ArrayList<String>(max * 3)
        var visited = 0
        while (queue.isNotEmpty() && found.size < max * 3 && visited < 400) {
            val e = queue.poll()
            visited++
            if (e.node.terminal) found += e.text
            for ((c, child) in e.node.children) queue.add(Entry(child, e.text + c))
        }
        val bad = misspellings
        return found.asSequence()
            .filter { bad?.isMisspelling(it) != true }
            .sortedBy { rank[it] ?: Int.MAX_VALUE }
            .take(max)
            .map { matchCase(prefix, it) }
            .toList()
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
        val maxDistance = if (w.length <= 4) 1f else 2f
        val span = 2
        val scored = ArrayList<Pair<String, Int>>()
        val buf = DistanceBuffers(w.length + 1 + span)
        for (len in (w.length - span)..(w.length + span)) {
            val bucket = byLength[len] ?: continue
            for (candidate in bucket) {
                val d = boundedDistance(w, candidate, maxDistance, buf)
                if (d > maxDistance) continue
                val r = rank[candidate] ?: continue
                // Weighted distance dominates (adjacent-key slips are cheap); frequency
                // breaks ties, and a matching or neighbouring first letter helps a little.
                val firstBonus = when {
                    candidate[0] == w[0] -> 0
                    KeyNeighbors.adjacent(candidate[0], w[0]) -> 1000
                    else -> 3000
                }
                scored += candidate to ((d * 100_000).toInt() + r + firstBonus)
            }
        }
        scored.sortBy { it.second }
        val bad = misspellings
        return scored.asSequence()
            .filter { bad?.isMisspelling(it.first) != true }
            .take(max)
            .map { matchCase(word, it.first) }
            .toList()
    }

    private class DistanceBuffers(n: Int) {
        var prev2 = FloatArray(n)
        var prev = FloatArray(n)
        var cur = FloatArray(n)
    }

    /** Keyboard-weighted OSA distance with an early exit once every cell exceeds [limit]. */
    private fun boundedDistance(a: String, b: String, limit: Float, buf: DistanceBuffers): Float {
        val m = a.length
        val n = b.length
        if (kotlin.math.abs(m - n) > limit + 0.01f) return limit + 1f
        var prev2 = buf.prev2
        var prev = buf.prev
        var cur = buf.cur
        for (i in 0..m) prev[i] = i.toFloat()
        for (j in 1..n) {
            cur[0] = j.toFloat()
            var rowMin = cur[0]
            for (i in 1..m) {
                val cost = KeyNeighbors.substitutionCost(a[i - 1], b[j - 1])
                var v = minOf(prev[i] + 1f, cur[i - 1] + 1f, prev[i - 1] + cost)
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                    v = minOf(v, prev2[i - 2] + KeyNeighbors.TRANSPOSE_COST)
                }
                cur[i] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > limit) return limit + 1f
            val t = prev2
            prev2 = prev
            prev = cur
            cur = t
        }
        return prev[m]
    }

    /** Keyboard-weighted edit distance between two words (adjacent-key slips cost less). */
    fun weightedDistance(a: String, b: String): Float =
        boundedDistance(a.lowercase(), b.lowercase(), 1e9f, DistanceBuffers(maxOf(a.length, b.length) + 1))

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
