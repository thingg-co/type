package com.aosmith.board.llm

/**
 * Prompt material shared by the app and by tools/eval.py on the desktop. Keep the two in
 * sync: the desktop eval is how these were chosen. Small models follow alternating
 * user/assistant example turns far better than examples listed inside one message, and the
 * "previous words / current word" layout beat bracket-style prompts for every model tried.
 */
object Prompts {
    const val WORD_SYSTEM = "You are the spell checker of a phone keyboard. " +
        "The user sends the previous words and the current word. " +
        "The current word is usually misspelled. Reply with only the intended word, nothing else. " +
        "If the current word is already a correctly spelled English word or a name, reply with it unchanged."

    const val SENTENCE_SYSTEM = "You fix spelling mistakes in short messages typed on a phone. " +
        "Reply with only the corrected message, nothing else. " +
        "Keep the wording, casing and punctuation as they are, only fix misspelled words. " +
        "If there are no mistakes, repeat the message unchanged."

    /** Few-shot turns for single-word mode, as (user, assistant) pairs. */
    val WORD_EXAMPLES: List<Pair<String, String>> = listOf(
        wordRequest("we walked to", "shcool") to "school",
        wordRequest("see you", "latre") to "later",
        wordRequest("thank you for", "everyhting") to "everything",
        wordRequest("the weather is", "sunny") to "sunny",
        wordRequest("I want to", "recieve") to "receive",
        wordRequest("it was", "realy") to "really",
        wordRequest("my friend", "Priya") to "Priya",
        wordRequest("do you", "rememebr") to "remember",
    )

    /** Few-shot turns for sentence mode. */
    val SENTENCE_EXAMPLES: List<Pair<String, String>> = listOf(
        "Th meeting is at nooon" to "The meeting is at noon",
        "can you send me the recipt" to "can you send me the receipt",
        "Good morning!" to "Good morning!",
    )

    /** GBNF grammar that limits single-word output to letters, apostrophes and hyphens. */
    const val WORD_GRAMMAR = "root ::= [A-Za-z] [A-Za-z'-]*"

    /** The user turn for one word-correction request. */
    fun wordRequest(before: String, word: String): String {
        val ctx = before.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }.takeLast(5).joinToString(" ")
        return "previous words: ${ctx.ifEmpty { "(start of message)" }}\ncurrent word: $word"
    }
}
