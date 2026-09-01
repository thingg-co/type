package com.aosmith.type.ime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EmojiDataTest {

    @Test fun `parses categories and skips comments`() {
        val cats = EmojiData.parse(
            sequenceOf(
                "# comment",
                "",
                "😀|😀 😃 😄",
                "❤️|❤️ 💔",
                "broken line without bar",
            ),
        )
        assertEquals(2, cats.size)
        assertEquals("😀", cats[0].icon)
        assertEquals(listOf("😀", "😃", "😄"), cats[0].emoji)
        assertEquals(2, cats[1].emoji.size)
    }

    @Test fun `shipped asset parses and looks sane`() {
        val f = File("src/main/assets/emoji.txt")
        assertTrue("asset missing: ${File(".").absolutePath}", f.exists())
        val cats = f.inputStream().use(EmojiData::fromStream)
        assertTrue("too few categories: ${cats.size}", cats.size >= 6)
        for (c in cats) {
            assertTrue("empty category ${c.icon}", c.emoji.isNotEmpty())
            for (e in c.emoji) {
                assertTrue("blank entry in ${c.icon}", e.isNotBlank())
                // Every entry must be a single grapheme-ish unit: no stray letters or spaces.
                assertTrue("suspicious entry '$e'", e.none { it.isLetterOrDigit() && it.code < 0x2100 })
            }
        }
        val all = cats.flatMap { it.emoji }
        assertEquals("duplicate emoji across categories", all.size, all.toSet().size)
    }
}
