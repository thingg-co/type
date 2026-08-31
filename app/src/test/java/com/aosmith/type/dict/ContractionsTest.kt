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

    @Test fun `words someone may have meant stay untouched`() {
        for (w in listOf("were", "ill", "id", "hell", "wed", "shed", "lets", "well", "its", "hello", "cannot")) {
            assertNull("should not touch $w", Contractions.fix(w))
        }
    }
}
