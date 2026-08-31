package com.aosmith.board.llm

import com.aosmith.board.dict.Dictionary
import com.aosmith.board.dict.TestWords
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CorrectionFilterTest {

    private val dict = Dictionary(TestWords.LIST.asSequence())

    // ---- single word -----------------------------------------------------------------

    @Test fun `accepts a close known correction and mirrors case`() {
        assertEquals("restaurant", CorrectionFilter.word(dict, "resturant", "restaurant"))
        assertEquals("Friend", CorrectionFilter.word(dict, "Freind", "friend"))
        assertEquals("EXPERT", CorrectionFilter.word(dict, "EXPRT", "expert"))
    }

    @Test fun `rejects echoes`() {
        assertNull(CorrectionFilter.word(dict, "typing", "typing"))
        assertNull(CorrectionFilter.word(dict, "Typing", "typing"))
    }

    @Test fun `rejects output that wandered from the input`() {
        // A real failure: the model returned the whole sentence squeezed into one "word".
        assertNull(CorrectionFilter.word(dict, "resturant", "I'llmeetyouattherestaurant"))
        assertNull(CorrectionFilter.word(dict, "cat", "experience"))
    }

    @Test fun `rejects invented words beyond one edit`() {
        assertNull(CorrectionFilter.word(dict, "blorbit", "florbix"))
    }

    @Test fun `rejects empty and multi-word output`() {
        assertNull(CorrectionFilter.word(dict, "resturant", null))
        assertNull(CorrectionFilter.word(dict, "resturant", "  "))
        assertNull(CorrectionFilter.word(dict, "resturant", "the restaurant"))
    }

    @Test fun `strips wrapping punctuation before judging`() {
        assertEquals("restaurant", CorrectionFilter.word(dict, "resturant", "\"restaurant.\""))
    }

    // ---- sentence --------------------------------------------------------------------

    @Test fun `accepts a faithful sentence fix`() {
        assertEquals("i have a cat", CorrectionFilter.sentence("i hav a cat", "i have a cat"))
    }

    @Test fun `rejects unchanged sentences`() {
        assertNull(CorrectionFilter.sentence("all good here", "all good here"))
    }

    @Test fun `rejects rewrites`() {
        assertNull(CorrectionFilter.sentence("hello there", "a completely different sentence rewrite"))
        assertNull(CorrectionFilter.sentence("short", "short plus a very long unjustified continuation of text"))
    }
}
