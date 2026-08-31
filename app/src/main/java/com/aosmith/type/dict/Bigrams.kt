package com.aosmith.type.dict

import java.io.DataInputStream
import java.io.IOException
import java.io.InputStream

/**
 * Word-pair frequencies packed for phones: sorted int64 keys (prevId shl 16 or nextId)
 * with a byte score each (10 * ln(count)). ~300k pairs cost a few MB and answer both
 * "how likely is B after A" and "what follows A" with binary search plus a range scan.
 *
 * Word ids are line numbers in en_words.txt, i.e. [Dictionary.idOf]. Built by
 * tools/build_bigrams.py from Norvig's web-corpus bigram counts.
 */
class Bigrams private constructor(
    private val keys: LongArray,
    private val scores: ByteArray,
) {
    val size: Int get() = keys.size

    /** Score of the pair (prev -> next), 0 when unseen. Higher is more likely. */
    fun score(prevId: Int, nextId: Int): Int {
        if (prevId < 0 || nextId < 0) return 0
        val i = keys.binarySearch(key(prevId, nextId))
        return if (i >= 0) scores[i].toInt() and 0xFF else 0
    }

    /** All (nextId, score) pairs seen after [prevId], best first, at most [max]. */
    fun nextWords(prevId: Int, max: Int): List<Pair<Int, Int>> {
        if (prevId < 0) return emptyList()
        var i = lowerBound(key(prevId, 0))
        val end = key(prevId + 1, 0)
        val found = ArrayList<Pair<Int, Int>>(32)
        while (i < keys.size && keys[i] < end) {
            found += (keys[i] and 0xFFFF).toInt() to (scores[i].toInt() and 0xFF)
            i++
        }
        found.sortByDescending { it.second }
        return if (found.size > max) found.subList(0, max) else found
    }

    private fun key(prev: Int, next: Int): Long = (prev.toLong() shl 16) or next.toLong()

    private fun LongArray.binarySearch(k: Long): Int {
        var lo = 0
        var hi = size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val v = this[mid]
            when {
                v < k -> lo = mid + 1
                v > k -> hi = mid - 1
                else -> return mid
            }
        }
        return -(lo + 1)
    }

    private fun lowerBound(k: Long): Int {
        val i = keys.binarySearch(k)
        return if (i >= 0) i else -(i + 1)
    }

    companion object {
        fun load(input: InputStream): Bigrams = DataInputStream(input.buffered(1 shl 16)).use { d ->
            val magic = ByteArray(4)
            d.readFully(magic)
            if (!magic.contentEquals(byteArrayOf('T'.code.toByte(), 'B'.code.toByte(), 'G'.code.toByte(), '1'.code.toByte()))) {
                throw IOException("bad bigram file")
            }
            val n = d.readInt()
            val keys = LongArray(n)
            for (i in 0 until n) keys[i] = d.readLong()
            val scores = ByteArray(n)
            d.readFully(scores)
            Bigrams(keys, scores)
        }

        fun load(context: android.content.Context, asset: String = "en_bigrams.bin"): Bigrams {
            val t0 = System.currentTimeMillis()
            val b = load(context.assets.open(asset))
            android.util.Log.i("Bigrams", "loaded ${b.size} pairs in ${System.currentTimeMillis() - t0} ms")
            return b
        }
    }
}
