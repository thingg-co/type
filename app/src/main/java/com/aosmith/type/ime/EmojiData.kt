package com.aosmith.type.ime

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Emoji categories for the panel, loaded from assets/emoji.txt. Data, not code, per the
 * layout convention: adding or reordering emoji is an asset edit. One category per line,
 * "tab-icon|space-separated emoji"; '#' lines are comments.
 */
object EmojiData {

    class Category(val icon: String, val emoji: List<String>)

    fun parse(lines: Sequence<String>): List<Category> =
        lines.mapNotNull { line ->
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) return@mapNotNull null
            val bar = trimmed.indexOf('|')
            if (bar <= 0) return@mapNotNull null
            val emoji = trimmed.substring(bar + 1).split(' ').filter { it.isNotEmpty() }
            if (emoji.isEmpty()) return@mapNotNull null
            Category(trimmed.substring(0, bar), emoji)
        }.toList()

    fun fromStream(input: InputStream): List<Category> =
        BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines(::parse)

    fun load(context: android.content.Context): List<Category> =
        fromStream(context.assets.open("emoji.txt"))
}
