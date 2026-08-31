package com.aosmith.type.dict

/**
 * Auto-apostrophe at the word boundary. The letters layer has no apostrophe key, so bare
 * contraction forms must be fixed automatically; the frequency word list knows "dont" and
 * "youre" as words, which is exactly why the normal unknown-word path never catches them.
 *
 * Only forms whose bare spelling is not a word someone plausibly meant are listed: "were",
 * "ill", "id", "hell", "wed", "shed", "lets" and friends stay untouched (the sentence pass
 * can fix those from context). The pronoun "i" rides along because it is the one word
 * English always capitalizes.
 */
object Contractions {

    private val MAP = mapOf(
        "i" to "I",
        "im" to "I'm", "ive" to "I've",
        "dont" to "don't", "cant" to "can't", "wont" to "won't",
        "isnt" to "isn't", "arent" to "aren't",
        "wasnt" to "wasn't", "werent" to "weren't",
        "hasnt" to "hasn't", "havent" to "haven't", "hadnt" to "hadn't",
        "doesnt" to "doesn't", "didnt" to "didn't",
        "couldnt" to "couldn't", "wouldnt" to "wouldn't", "shouldnt" to "shouldn't",
        "mustnt" to "mustn't", "neednt" to "needn't", "aint" to "ain't",
        "youre" to "you're", "youve" to "you've", "youll" to "you'll", "youd" to "you'd",
        "theyre" to "they're", "theyve" to "they've", "theyll" to "they'll", "theyd" to "they'd",
        "weve" to "we've", "whos" to "who's", "whats" to "what's", "thats" to "that's",
        "theres" to "there's", "heres" to "here's", "wheres" to "where's", "hows" to "how's",
        "hes" to "he's", "shes" to "she's", "itll" to "it'll",
        "wouldve" to "would've", "couldve" to "could've", "shouldve" to "should've",
        "mightve" to "might've", "mustve" to "must've",
        "oclock" to "o'clock", "yall" to "y'all", "maam" to "ma'am", "cmon" to "c'mon",
        "thatll" to "that'll", "itd" to "it'd", "whyd" to "why'd", "whatd" to "what'd",
        "whered" to "where'd", "howd" to "how'd", "whatll" to "what'll", "wholl" to "who'll",
        "whens" to "when's", "shant" to "shan't", "neednt" to "needn't",
    )

    /**
     * Bare forms that are real words AND common contractions. These are never map-fixed;
     * the caller screens them with the prediction network and lets the language model
     * decide from context.
     */
    private val AMBIGUOUS = mapOf(
        "were" to "we're", "well" to "we'll", "ill" to "I'll", "id" to "I'd",
        "its" to "it's", "lets" to "let's", "hell" to "he'll", "shell" to "she'll",
        "wed" to "we'd",
    )

    /** The contraction reading of an ambiguous bare form, or null. */
    fun ambiguousReading(typed: String): String? = AMBIGUOUS[typed.lowercase()]

    /** The apostrophized form of [typed] with its capitalization kept, or null. */
    fun fix(typed: String): String? {
        val mapped = MAP[typed.lowercase()] ?: return null
        val cased = Dictionary.matchCase(typed, mapped)
        return if (cased == "i" || cased.startsWith("i'")) {
            cased.replaceFirstChar { it.uppercaseChar() }
        } else {
            cased
        }
    }
}
