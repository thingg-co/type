package com.aosmith.type.llm

import com.aosmith.type.dict.Dictionary

/**
 * Decides whether raw model output is allowed to touch the user's text. Pure functions,
 * unit-tested directly: every rule here is the difference between a helpful correction and
 * a keyboard that mangles what you typed.
 */
object CorrectionFilter {

    /**
     * Validates a single-word correction of [original]. Null means "make no change": the
     * model agreed with the user, wandered too far from what was typed, or invented a word.
     */
    fun word(dictionary: Dictionary?, original: String, raw: String?): String? {
        if (raw == null) return null
        val candidate = raw.trim().trim('[', ']', '"', '.', ',', '!', '?', ':', ';').trim()
        if (candidate.isEmpty() || candidate.any { it.isWhitespace() }) return null
        if (candidate.equals(original, ignoreCase = true)) return null
        // Keyboard-weighted distance: adjacent-key slips barely count, so honest fat-finger
        // fixes pass while same-length rewrites with distant letters stay blocked.
        val distance = dictionary?.weightedDistance(candidate, original)
            ?: Dictionary.editDistance(candidate.lowercase(), original.lowercase()).toFloat()
        val limit = when {
            original.length <= 3 -> 1f
            original.length <= 6 -> 2f
            else -> 3f
        }
        if (distance > limit) return null
        val known = dictionary?.isKnown(candidate) ?: true
        // An unknown output is only trusted for a one-edit change; otherwise the model is guessing.
        if (!known && distance > 1f) return null
        // applyCase = matchCase plus the English "I" rule: "ill" -> "I'll", never "i'll";
        // the casing table then upgrades proper nouns ("septmber" -> "September").
        val cased = com.aosmith.type.dict.Contractions.applyCase(original, candidate)
        return dictionary?.casing?.canonical(cased) ?: cased
    }

    /** Validates a whole-sentence correction. Null means "make no change". */
    fun sentence(original: String, raw: String?): String? {
        if (raw == null) return null
        val candidate = raw.trim().removeSurrounding("\"")
        if (candidate.isEmpty() || candidate == original) return null
        // Reject rewrites: length should stay close and most words should survive.
        val ratio = candidate.length.toDouble() / original.length
        if (ratio < 0.6 || ratio > 1.5) return null
        val origWords = original.lowercase().split(Regex("\\s+"))
        val candWords = candidate.lowercase().split(Regex("\\s+"))
        if (kotlin.math.abs(origWords.size - candWords.size) > 2) return null
        val overlap = origWords.count { it in candWords }
        if (overlap < origWords.size * 0.5) return null
        return candidate
    }
}
