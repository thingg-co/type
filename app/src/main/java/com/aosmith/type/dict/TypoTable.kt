package com.aosmith.type.dict

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Curated common-misspelling fixes (Wikipedia's list), applied instantly at the word
 * boundary alongside [Contractions]. Doubles as a denylist: a typo that happens to be in
 * the frequency word list ("belive", "begining") must not count as a known word or ever be
 * suggested, or the correction machinery politely ignores the commonest typos in English.
 */
class TypoTable private constructor(private val fixes: HashMap<String, String>) {

    val size: Int get() = fixes.size

    /** The correction for [typed], capitalization preserved, or null. */
    fun fix(typed: String): String? {
        val f = fixes[typed.lowercase()] ?: return null
        return Dictionary.matchCase(typed, f)
    }

    /** True when [word] is a known common misspelling (and so never a valid suggestion). */
    fun isMisspelling(word: String): Boolean = fixes.containsKey(word.lowercase())

    companion object {
        fun load(input: InputStream): TypoTable {
            val map = HashMap<String, String>(6000)
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines { lines ->
                for (line in lines) {
                    val tab = line.indexOf('\t')
                    if (tab > 0) map[line.substring(0, tab)] = line.substring(tab + 1).trim()
                }
            }
            return TypoTable(map)
        }

        fun load(context: android.content.Context, asset: String = "en_typos.tsv"): TypoTable =
            load(context.assets.open(asset))
    }
}
