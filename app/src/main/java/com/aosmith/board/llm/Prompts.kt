package com.aosmith.board.llm

/**
 * Prompt material shared by the app and by tools/eval on the desktop. Keep the two in sync:
 * the desktop eval is how the few-shot examples were chosen.
 */
object Prompts {
    const val SYSTEM = "You are a spell checker for text typed on a phone keyboard. " +
        "Fix spelling mistakes and nothing else. Do not change wording, grammar, punctuation, or capitalization. " +
        "Reply with only the corrected text."

    /** Few-shot turns for single-word mode: the word in brackets is the one to fix. */
    val WORD_EXAMPLES: List<Pair<String, String>> = listOf(
        "I will meet you at the [resturant]" to "restaurant",
        "she said it was [thier] turn" to "their",
        "[definately]" to "definitely",
        "the [wether] is nice today" to "weather",
        "can you send me the [adress]" to "address",
        "we went [tommorow]" to "tomorrow",
        "he is a good [freind]" to "friend",
        "I [recieved] your message" to "received",
    )

    /** Few-shot turns for sentence mode. */
    val SENTENCE_EXAMPLES: List<Pair<String, String>> = listOf(
        "I recieved you're mesage yesterday and will reply tommorow" to "I received your message yesterday and will reply tomorrow",
        "are you comming to the resturant tonight" to "are you coming to the restaurant tonight",
        "Thanks, that works for me." to "Thanks, that works for me.",
    )

    const val WORD_INSTRUCTION = "Fix the word in brackets: "
    const val SENTENCE_INSTRUCTION = "Fix the spelling: "

    /** GBNF grammar that limits single-word output to letters, apostrophes and hyphens. */
    const val WORD_GRAMMAR = "root ::= [A-Za-z] [A-Za-z'-]*"

    fun wordRequest(before: String, word: String): String {
        val ctx = before.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.takeLast(6).joinToString(" ")
        return if (ctx.isEmpty()) "[$word]" else "$ctx [$word]"
    }
}
