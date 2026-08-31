package com.aosmith.type.dict

/** What the keyboard should offer for the word currently being typed. */
sealed class MidWordAction {
    object None : MidWordAction()

    /** Few enough ways to finish the word: show them as whole-word keys. */
    data class WordKeys(val words: List<String>) : MidWordAction()

    /** The prefix is on track for many words: show the most likely completions. */
    data class Predictions(val words: List<String>) : MidWordAction()

    /** Nothing typed yet: likely next words after the previous token. */
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
 * Context: [previousWord] comes from [Lexer.previousWord]; with a [Bigrams] table it
 * ranks completions and predictions, and fills the bar with next-word suggestions when
 * nothing has been typed yet.
 */
object TypingPolicy {
    const val WORD_KEY_LIMIT = 3
    const val PREDICTION_COUNT = 3
    private const val CANDIDATES = 8

    fun midWord(dict: Dictionary, bigrams: Bigrams?, previousWord: String?, word: String): MidWordAction {
        if (word.isEmpty()) {
            val next = nextWords(dict, bigrams, previousWord, PREDICTION_COUNT)
            return if (next.isEmpty()) MidWordAction.None else MidWordAction.NextWords(next)
        }
        if (word.length < 2) return MidWordAction.None

        val completions = rerank(dict, bigrams, previousWord, dict.completions(word, WORD_KEY_LIMIT))
            .filterNot { it.equals(word, ignoreCase = true) }
        if (completions.isNotEmpty()) return MidWordAction.WordKeys(completions)

        if (dict.hasPrefix(word)) {
            if (word.length < 3) return MidWordAction.None
            val predictions = rerank(dict, bigrams, previousWord, dict.predictions(word, CANDIDATES))
                .filterNot { it.equals(word, ignoreCase = true) }
                .take(PREDICTION_COUNT)
            return if (predictions.isEmpty()) MidWordAction.None else MidWordAction.Predictions(predictions)
        }

        if (dict.isKnown(word)) return MidWordAction.None

        return MidWordAction.Typo(word, dict.suggest(word, 2), askModel = word.length >= 3)
    }

    /** Likely words after [previousWord], for the empty-prefix bar. */
    fun nextWords(dict: Dictionary, bigrams: Bigrams?, previousWord: String?, max: Int): List<String> {
        if (bigrams == null || previousWord == null) return emptyList()
        val prevId = dict.idOf(previousWord)
        if (prevId < 0) return emptyList()
        return bigrams.nextWords(prevId, max).mapNotNull { dict.wordOf(it.first) }
    }

    /**
     * Context-aware ordering: pairs seen after the previous word rise to the front.
     * The sort is stable, so the incoming frequency order breaks ties.
     */
    private fun rerank(dict: Dictionary, bigrams: Bigrams?, previousWord: String?, words: List<String>): List<String> {
        if (bigrams == null || previousWord == null || words.size < 2) return words
        val prevId = dict.idOf(previousWord)
        if (prevId < 0) return words
        return words.sortedByDescending { bigrams.score(prevId, dict.idOf(it)) }
    }
}
