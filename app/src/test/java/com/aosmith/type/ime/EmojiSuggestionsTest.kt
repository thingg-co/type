package com.aosmith.type.ime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EmojiSuggestionsTest {

    private val sugs = EmojiSuggestions(
        EmojiData.parse(
            sequenceOf(
                "lol|😂 🤣",
                "haha|😂 😄",
                "555|😂 🤣",
                "love|❤️ 😍",
            ),
        ),
    )

    @Test fun `matches triggers case-insensitively`() {
        assertEquals(listOf("😂", "🤣"), sugs.forWord("lol"))
        assertEquals(listOf("😂", "🤣"), sugs.forWord("LOL"))
        assertEquals(listOf("❤️", "😍"), sugs.forWord("Love"))
        assertTrue(sugs.forWord("lot").isEmpty())
        assertTrue(sugs.forWord("").isEmpty())
    }

    @Test fun `laugh variants fold onto their trigger`() {
        assertEquals(listOf("😂", "😄"), sugs.forWord("hahahah"))
        assertEquals(listOf("😂", "😄"), sugs.forWord("HAHAHA"))
        assertEquals(listOf("😂", "🤣"), sugs.forWord("loool"))
        assertEquals(listOf("😂", "🤣"), sugs.forWord("55555"))
        assertTrue(sugs.forWord("hah").isEmpty())      // too short to be sure
        assertTrue(sugs.forWord("hahnot").isEmpty())
    }

    @Test fun `digit runs come from the editor tail`() {
        assertEquals(listOf("😂", "🤣"), sugs.forTextTail("so funny 555"))
        assertEquals(listOf("😂", "🤣"), sugs.forTextTail("555"))
        assertTrue(sugs.forTextTail("call 555-0123").isEmpty()) // no trailing 5-run
        assertTrue(sugs.forTextTail("55").isEmpty())            // too short
        assertTrue(sugs.forTextTail("1555").isEmpty())          // part of a number
        assertTrue(sugs.forTextTail("3.555").isEmpty())         // decimal
        assertTrue(sugs.forTextTail("").isEmpty())
    }

    @Test fun `normalization folds variants and leaves ordinary words alone`() {
        assertEquals("hello", EmojiSuggestions.normalizeLaugh("hello"))
        assertEquals("lolly", EmojiSuggestions.normalizeLaugh("lolly"))
        assertEquals("haha", EmojiSuggestions.normalizeLaugh("hahah"))
        assertEquals("555", EmojiSuggestions.normalizeLaugh("55555"))
    }

    @Test fun `shipped trigger asset parses`() {
        val f = File("src/main/assets/emoji_words.txt")
        assertTrue(f.exists())
        val cats = f.inputStream().use(EmojiData::fromStream)
        assertTrue("too few triggers: ${cats.size}", cats.size >= 50)
        val s = EmojiSuggestions(cats)
        assertTrue(s.forWord("lol").isNotEmpty())
        assertTrue(s.forWord("hahahaha").isNotEmpty())
        assertTrue(s.forTextTail("555").isNotEmpty())
    }
}
