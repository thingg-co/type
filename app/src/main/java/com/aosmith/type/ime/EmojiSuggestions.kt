package com.aosmith.type.ime

/**
 * Word-to-emoji suggestions for the strip: "lol" offers 😂, "555" (the Thai laugh) offers
 * the same. Triggers live in assets/emoji_words.txt (same line format as emoji.txt), so
 * another language adds triggers without touching this logic; only the laugh-variant
 * normalization is code, because "hahahaha" and "5555555" are patterns, not list entries.
 */
class EmojiSuggestions(categories: List<EmojiData.Category>) {

    private val byWord: Map<String, List<String>> =
        categories.associate { it.icon to it.emoji }

    /** Emoji for a completed-looking [word], or empty. Case-insensitive, variant-tolerant. */
    fun forWord(word: String): List<String> {
        if (word.isEmpty()) return emptyList()
        val w = word.lowercase()
        byWord[w]?.let { return it }
        return byWord[normalizeLaugh(w)] ?: emptyList()
    }

    /**
     * Emoji when the text just typed ends in a digit-run laugh ("555", "55555"): digits
     * never enter the word tracker, so the caller hands the raw editor tail instead.
     */
    fun forTextTail(tail: CharSequence): List<String> {
        var i = tail.length
        while (i > 0 && tail[i - 1] == '5') i--
        val run = tail.length - i
        if (run < 3) return emptyList()
        if (i > 0 && (tail[i - 1].isLetterOrDigit() || tail[i - 1] == '.')) return emptyList() // part of a number
        return byWord["555"] ?: emptyList()
    }

    companion object {
        /**
         * Folds laugh variants onto their canonical trigger: "hahahah" -> "haha",
         * "loool" -> "lol", "hehehe" -> "hehe", "55555" -> "555". Anything else is
         * returned unchanged.
         */
        fun normalizeLaugh(w: String): String = when {
            w.length >= 4 && Regex("(ha)+h?").matches(w) -> "haha"
            w.length >= 4 && Regex("(he)+h?").matches(w) -> "hehe"
            w.length >= 4 && Regex("lo+l").matches(w) -> "lol"
            w.length >= 3 && w.all { it == '5' } -> "555"
            else -> w
        }

        fun load(context: android.content.Context): EmojiSuggestions =
            EmojiSuggestions(context.assets.open("emoji_words.txt").use(EmojiData::fromStream))
    }
}
