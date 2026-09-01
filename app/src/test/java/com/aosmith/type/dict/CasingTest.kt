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

    @Test fun `curated given names are cased, ambiguous ones held out`() {
        val c = File("src/main/assets/en_caps.txt").inputStream().use(Casing::fromStream)
        for (w in listOf("jason", "jessica", "michelle", "brandon", "steven", "timothy")) {
            assertEquals(w.replaceFirstChar { it.uppercaseChar() }, c.canonical(w))
        }
        // Names with a live lowercase reading wait for the cased-vocab retrain
        // (tools/curated_words.py prints the full holdout list). Corpus-mined entries
        // (joe, jean) are not held out: measured casing beats judgment.
        for (w in listOf("mark", "jack", "holly", "hunter", "amber", "robin", "jake")) {
            assertNull("'$w' must not be auto-capitalized", c.canonical(w))
        }
    }

    @Test fun `curated slang is in the vocabulary`() {
        val words = File("src/main/assets/en_words.txt").readLines().toHashSet()
        for (w in listOf("ngl", "bruh", "smh", "gimme", "lemme", "y'all", "welp", "tmrw")) {
            assertTrue("'$w' missing from en_words.txt", w in words)
        }
    }
}
