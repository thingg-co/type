package com.aosmith.type.dict

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CasingTest {

    private val casing = Casing(
        sequenceOf("september" to "September", "london" to "London", "i" to "I", "usa" to "USA"),
    )

    @Test fun `upgrades only fully lowercase input`() {
        assertEquals("September", casing.canonical("september"))
        assertEquals("London", casing.canonical("london"))
        assertEquals("I", casing.canonical("i"))
        assertNull(casing.canonical("September"))   // already cased: user's choice
        assertNull(casing.canonical("SEPTEMBER"))   // ALL-CAPS stays
        assertNull(casing.canonical("hello"))
        assertNull(casing.canonical(""))
    }

    @Test fun `shipped asset loads with the expected entries and exclusions`() {
        val f = File("src/main/assets/en_caps.txt")
        assertTrue(f.exists())
        val c = f.inputStream().use(Casing::fromStream)
        assertTrue("suspiciously few entries: ${c.size}", c.size >= 400)
        for (w in listOf("september", "monday", "january", "english", "london", "christmas")) {
            assertEquals(w.replaceFirstChar { it.uppercaseChar() }, c.canonical(w))
        }
        // Ambiguous words must stay out: their lowercase readings dominate real text.
        for (w in listOf("may", "march", "the", "will", "mark")) {
            assertNull("'$w' must not be auto-capitalized", c.canonical(w))
        }
    }
}
