package com.aosmith.type.dict

/** What the keyboard should offer for the word currently being typed. */
sealed class MidWordAction {
    object None : MidWordAction()

    /** Few enough ways to finish the word: show them as whole-word keys. */
    data class WordKeys(val words: List<String>) : MidWordAction()

    /** The prefix is on track for many words: show the most likely completions. */
    data class Predictions(val words: List<String>) : MidWordAction()

    /** Nothing typed yet: likely next words after the previous tokens. */
    data class NextWords(val words: List<String>) : MidWordAction()

    /** The prefix cannot start any dictionary word: it is already a typo. */
    data class Typo(val typed: String, val dictSuggestions: List<String>, val askModel: Boolean) : MidWordAction()
}

/**
 * The mid-word decision tree, kept free of Android and model dependencies so it can be
 * unit-tested directly.
 *
 * The principle (learned the hard way): an unfinished word is prediction territory, not
 * correction territory. "typi" must never be "corrected" to "type" while the user is on
 * their way to "typing". The language model is only consulted mid-word once the prefix
 * cannot start any dictionary word, because at that point a typo has already happened.
 * Finished words (at the word boundary) are handled elsewhere.
 *
 * Context: [previousWords] comes from [Lexer.previousWords]. When the next-word network
 * ([NeuralLm]) is loaded it conditions on up to K previous tokens (including sentence
 * starts); otherwise the [Bigrams] table conditions on the last one.
 */
object TypingPolicy {
    const val WORD_KEY_LIMIT = 3
    const val PREDICTION_COUNT = 3
    private const val CANDIDATES = 8

    fun midWord(
        dict: Dictionary,
        bigrams: Bigrams?,
        neural: NeuralLm?,
        previousWords: List<String>,
        word: String,
    ): MidWordAction {
        if (word.isEmpty()) {
            val next = nextWords(dict, bigrams, neural, previousWords, PREDICTION_COUNT)
            return if (next.isEmpty()) MidWordAction.None else MidWordAction.NextWords(next)
        }
        if (word.length < 2) return MidWordAction.None

        val completions = rerank(dict, bigrams, neural, previousWords, dict.completions(word, WORD_KEY_LIMIT))
            .filterNot { it.equals(word, ignoreCase = true) }
        if (completions.isNotEmpty()) return MidWordAction.WordKeys(completions)

        if (dict.hasPrefix(word)) {
            val ranked = rerank(dict, bigrams, neural, previousWords, dict.predictions(word, CANDIDATES))
                .filterNot { it.equals(word, ignoreCase = true) }
            // Two-letter prefixes only speak up when the context genuinely knows something:
            // "ha" after "should" is worth offering "have"; bare "ha" is noise.
            val contextBacked = ranked.isNotEmpty() &&
                contextScore(dict, bigrams, neural, previousWords, ranked.first()) > 0f
            if (word.length < 3 && !contextBacked) return MidWordAction.None
            val predictions = ranked.take(PREDICTION_COUNT)
            return if (predictions.isEmpty()) MidWordAction.None else MidWordAction.Predictions(predictions)
        }

        if (dict.isKnown(word)) return MidWordAction.None

        return MidWordAction.Typo(word, dict.suggest(word, 2), askModel = word.length >= 3)
    }

    /** Likely words for an empty prefix. The network handles sentence starts; bigrams cannot. */
    fun nextWords(dict: Dictionary, bigrams: Bigrams?, neural: NeuralLm?, previousWords: List<String>, max: Int): List<String> {
        if (neural != null) {
            val ctx = contextIds(dict, neural, previousWords) ?: return emptyList()
            return neural.topNext(ctx, max).mapNotNull { dict.wordOf(it) }
        }
        val prev = previousWords.lastOrNull() ?: return emptyList()
        if (bigrams == null) return emptyList()
        val prevId = dict.idOf(prev)
        if (prevId < 0) return emptyList()
        return bigrams.nextWords(prevId, max).mapNotNull { dict.wordOf(it.first) }
    }

    /**
     * Context-aware ordering: candidates the context expects rise to the front. The sort is
     * stable, so the incoming frequency order breaks ties.
     */
    private fun rerank(
        dict: Dictionary,
        bigrams: Bigrams?,
        neural: NeuralLm?,
        previousWords: List<String>,
        words: List<String>,
    ): List<String> {
        if (words.size < 2) return words
        if (neural != null) {
            val ctx = contextIds(dict, neural, previousWords) ?: return words
            val ids = IntArray(words.size) { dict.idOf(words[it]).let { id -> if (id >= 0) id else neural.unk } }
            val scores = neural.scoreCandidates(ctx, ids)
            val order = words.indices.sortedByDescending { scores[it] }
            return order.map { words[it] }
        }
        val prev = previousWords.lastOrNull() ?: return words
        if (bigrams == null) return words
        val prevId = dict.idOf(prev)
        if (prevId < 0) return words
        return words.sortedByDescending { bigrams.score(prevId, dict.idOf(it)) }
    }

    /** Positive when the context model actually knows the pair, 0 when it is guessing. */
    private fun contextScore(dict: Dictionary, bigrams: Bigrams?, neural: NeuralLm?, previousWords: List<String>, word: String): Float {
        if (neural != null) {
            // The network always has an opinion; require real context so bare two-letter
            // prefixes stay quiet at sentence starts too.
            return if (previousWords.isNotEmpty()) 1f else 0f
        }
        val prev = previousWords.lastOrNull() ?: return 0f
        if (bigrams == null) return 0f
        val prevId = dict.idOf(prev)
        if (prevId < 0) return 0f
        return bigrams.score(prevId, dict.idOf(word)).toFloat()
    }

    /** Word ids for the network; null when there is no usable context at all. */
    private fun contextIds(dict: Dictionary, neural: NeuralLm, previousWords: List<String>): List<Int>? {
        if (previousWords.isEmpty()) return null
        return previousWords.map { dict.idOf(it).let { id -> if (id >= 0) id else neural.unk } }
    }
}
