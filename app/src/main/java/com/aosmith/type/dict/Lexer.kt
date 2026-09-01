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
            // Emoji are invisible to context, like whitespace: "sounds good 😀" must still
            // condition on "good", and an emoji next to the cursor must not kill the context
            // the way punctuation does.
            while (i > 0 && (before[i - 1].isWhitespace() || isEmojiChar(before[i - 1]))) {
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

    /**
     * A UTF-16 unit that belongs to an emoji (or another astral-plane glyph). Surrogates
     * cover everything above the BMP, where almost all emoji live; the listed ranges are
     * the BMP emoji blocks (misc symbols, dingbats, arrows, geometric shapes, technical)
     * plus the variation selector, ZWJ and keycap combiner. Char-level on purpose: callers
     * walk backwards one UTF-16 unit at a time. Not included: '‼' and '⁉', which end a
     * sentence like the plain punctuation they contain.
     */
    fun isEmojiChar(c: Char): Boolean =
        c.isSurrogate() ||
            c.code == 0xFE0F || c.code == 0x200D || c.code == 0x20E3 ||
            c.code in 0x2600..0x27BF || c.code in 0x2B00..0x2BFF ||
            c.code in 0x2300..0x23FF || c.code in 0x25A0..0x25FF ||
            c.code in 0x2190..0x21FF ||
            c.code == 0x2139 || c.code == 0x24C2 ||
            c.code == 0x3030 || c.code == 0x303D || c.code == 0x3297 || c.code == 0x3299
}
