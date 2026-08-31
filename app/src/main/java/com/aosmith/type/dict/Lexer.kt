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
    fun previousWord(before: CharSequence): String? {
        var i = before.length
        while (i > 0 && before[i - 1].isWhitespace()) {
            if (before[i - 1] == '\n') return null
            i--
        }
        if (i == 0) return null
        val last = before[i - 1]
        if (!last.isLetter() && last != '\'') return null // punctuation or digit ends the context
        val end = i
        while (i > 0 && (before[i - 1].isLetter() || before[i - 1] == '\'')) i--
        val word = before.subSequence(i, end).toString().lowercase()
        return word.ifEmpty { null }
    }
}
