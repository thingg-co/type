package com.aosmith.type.dict

/** What the keyboard should offer for the word currently being typed. */
sealed class MidWordAction {
    object None : MidWordAction()

    /** Few enough ways to finish the word: show them as whole-word keys. */
    data class WordKeys(val words: List<String>) : MidWordAction()

    /** The prefix is on track for many words: show the most likely completions. */
    data class Predictions(val words: List<String>) : MidWordAction()

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
 */
object TypingPolicy {
    const val WORD_KEY_LIMIT = 3
    const val PREDICTION_COUNT = 3

    fun midWord(dict: Dictionary, word: String): MidWordAction {
        if (word.length < 2) return MidWordAction.None

        val completions = dict.completions(word, WORD_KEY_LIMIT).filterNot { it.equals(word, ignoreCase = true) }
        if (completions.isNotEmpty()) return MidWordAction.WordKeys(completions)

        if (dict.hasPrefix(word)) {
            if (word.length < 3) return MidWordAction.None
            val predictions = dict.predictions(word, PREDICTION_COUNT).filterNot { it.equals(word, ignoreCase = true) }
            return if (predictions.isEmpty()) MidWordAction.None else MidWordAction.Predictions(predictions)
        }

        if (dict.isKnown(word)) return MidWordAction.None

        return MidWordAction.Typo(word, dict.suggest(word, 2), askModel = word.length >= 3)
    }
}
