package com.aosmith.type.dict

/**
 * Minimal lexer over the text left of the cursor. Its one job today: find the previous
 * completed token so suggestions can be conditioned on it. Sentence punctuation, digits
 * and blank starts all mean "no context", which callers treat as frequency-only ranking.
 */
object Lexer {

    /**
     * The word immediately before the cursor position, lowercased, or null when the cursor
     * follows a sentence boundary, punctuation, a number, or nothing at all.
     *
     * The caller passes text that ends where the current (possibly partial) word begins.
     */
    fun previousWord(before: CharSequence): String? = previousWords(before, 1).lastOrNull()

    /**
     * Up to [k] completed words before the cursor, oldest first, all lowercased. The walk
     * stops at the first non-word character going backwards, so a sentence boundary yields
     * an empty list and a mid-sentence cursor yields only the words of this sentence run.
     */
    fun previousWords(before: CharSequence, k: Int): List<String> {
        val words = ArrayList<String>(k)
        var i = before.length
        var first = true
        while (words.size < k) {
            while (i > 0 && before[i - 1].isWhitespace()) {
                if (before[i - 1] == '\n') return words.reversed()
                i--
            }
            if (i == 0) break
            val last = before[i - 1]
            if (!last.isLetter() && last != '\'') {
                // punctuation or a digit: for the nearest token this kills the context
                // entirely; further back it just ends the walk.
                if (first) return emptyList()
                break
            }
            val end = i
            while (i > 0 && (before[i - 1].isLetter() || before[i - 1] == '\'')) i--
            val word = before.subSequence(i, end).toString().lowercase()
            if (word.isNotEmpty()) words += word
            first = false
        }
        return words.reversed()
    }
}
