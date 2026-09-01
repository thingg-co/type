package com.aosmith.type.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ContractionsTest {

    @Test fun `bare forms gain their apostrophes`() {
        assertEquals("don't", Contractions.fix("dont"))
        assertEquals("they're", Contractions.fix("theyre"))
        assertEquals("we've", Contractions.fix("weve"))
        assertEquals("won't", Contractions.fix("wont"))
        assertEquals("o'clock", Contractions.fix("oclock"))
    }

    @Test fun `capitalization is preserved`() {
        assertEquals("Don't", Contractions.fix("Dont"))
        assertEquals("DON'T", Contractions.fix("DONT"))
        assertEquals("You're", Contractions.fix("Youre"))
    }

    @Test fun `the pronoun is always capital`() {
        assertEquals("I", Contractions.fix("i"))
        assertEquals("I'm", Contractions.fix("im"))
        assertEquals("I'm", Contractions.fix("Im"))
        assertEquals("I've", Contractions.fix("ive"))
    }

    @Test fun `extended forms fix too`() {
        assertEquals("that'll", Contractions.fix("thatll"))
        assertEquals("why'd", Contractions.fix("whyd"))
        assertEquals("who'll", Contractions.fix("wholl"))
        assertEquals("shan't", Contractions.fix("shant"))
    }

    @Test fun `ambiguous forms live in Confusables, never in the auto-fix map`() {
        for ((bare, reading) in listOf(
            "were" to "we're", "well" to "we'll", "ill" to "i'll", "id" to "i'd",
            "its" to "it's", "lets" to "let's", "hell" to "he'll", "wed" to "we'd",
        )) {
            assertNull("must not auto-fix $bare", Contractions.fix(bare))
            assertEquals(listOf(reading), Confusables.alternativesOf(bare))
        }
        assertEquals(emptyList<String>(), Confusables.alternativesOf("hello"))
    }

    @Test fun `applyCase mirrors the typed case and always capitalizes I`() {
        assertEquals("Their", Contractions.applyCase("There", "their"))
        assertEquals("THAN", Contractions.applyCase("THEN", "than"))
        assertEquals("I'll", Contractions.applyCase("ill", "i'll"))
        assertEquals("I'll", Contractions.applyCase("Ill", "i'll"))
        assertEquals("we're", Contractions.applyCase("were", "we're"))
    }

    @Test fun `words someone may have meant stay untouched`() {
        for (w in listOf("were", "ill", "id", "hell", "wed", "shed", "lets", "well", "its", "hello", "cannot")) {
            assertNull("should not touch $w", Contractions.fix(w))
        }
    }
}
