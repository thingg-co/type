package com.aosmith.type.dict

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Canonical capitalization for words English always writes cased: months, weekdays,
 * languages, places, names (assets/en_caps.txt, mined from corpus casing statistics by
 * the asset's generator — see tools; "may" and "march" are correctly absent because
 * their lowercase readings dominate real text). Data, not rules: a new language ships
 * its own file or none.
 */
class Casing(pairs: Sequence<Pair<String, String>>) {

    private val map = HashMap<String, String>(1024)

    init {
        for ((lower, cased) in pairs) map[lower] = cased
    }

    val size: Int get() = map.size

    /**
     * The canonical form of [typed], or null when nothing should change. Only fully
     * lowercase input is upgraded: any capital the user typed is a choice, and ALL-CAPS
     * stays ALL-CAPS.
     */
    fun canonical(typed: String): String? {
        if (typed.isEmpty() || !typed.all { !it.isLetter() || it.isLowerCase() }) return null
        val cased = map[typed] ?: return null
        return if (cased == typed) null else cased
    }

    companion object {
        fun fromStream(input: InputStream): Casing =
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines { lines ->
                Casing(
                    lines.mapNotNull { line ->
                        if (line.isEmpty() || line.startsWith("#")) return@mapNotNull null
                        val tab = line.indexOf('\t')
                        if (tab <= 0) return@mapNotNull null
                        line.substring(0, tab) to line.substring(tab + 1)
                    }.asSequence().toList().asSequence(),
                )
            }

        fun load(context: android.content.Context, asset: String = "en_caps.txt"): Casing =
            fromStream(context.assets.open(asset))
    }
}
