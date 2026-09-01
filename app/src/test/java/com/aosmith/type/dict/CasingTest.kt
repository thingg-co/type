package com.aosmith.type.dict

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
        // Ambiguous non-name words must stay out: their lowercase readings dominate.
        for (w in listOf("may", "march", "the", "will")) {
            assertNull("'$w' must not be auto-capitalized", c.canonical(w))
        }
    }

    @Test fun `curated given names are cased, ambiguity notwithstanding`() {
        val c = File("src/main/assets/en_caps.txt").inputStream().use(Casing::fromStream)
        // Every curated name capitalizes, including those with a live lowercase
        // reading (mark, jack, holly): a stray capital is a smaller wrong than
        // leaving someone's name lowercase. See tools/curated_words.py.
        for (w in listOf(
                "jason", "jessica", "michelle", "brandon", "steven", "timothy",
                "mark", "jack", "holly", "hunter", "amber", "robin", "jake",
            )) {
            assertEquals(w.replaceFirstChar { it.uppercaseChar() }, c.canonical(w))
        }
    }

    @Test fun `states and cities are cased, primary-use collisions held out`() {
        val c = File("src/main/assets/en_caps.txt").inputStream().use(Casing::fromStream)
        for (w in listOf("texas", "dallas", "phoenix", "hampshire", "york", "bangkok", "tokyo")) {
            assertEquals(w.replaceFirstChar { it.uppercaseChar() }, c.canonical(w))
        }
        // Held out: the lowercase reading is the primary chat use, not an occasional one.
        for (w in listOf("turkey", "nice", "mobile", "orange", "reading", "buffalo", "jersey", "new")) {
            assertNull("'$w' must not be auto-capitalized", c.canonical(w))
        }
    }

    @Test fun `curated slang is in the vocabulary`() {
        val words = File("src/main/assets/en_words.txt").readLines().toHashSet()
        for (w in listOf("ngl", "bruh", "smh", "gimme", "lemme", "y'all", "welp", "tmrw")) {
            assertTrue("'$w' missing from en_words.txt", w in words)
        }
    }

    @Test fun `closed compounds are known, run-together typos are not`() {
        val words = File("src/main/assets/en_words.txt").readLines().toHashSet()
        // Compound admission (tools/expand_vocab.py): both parts trusted and 4+ letters.
        for (w in listOf("minefield", "weeknight", "campsite", "screenshot")) {
            assertTrue("'$w' missing from en_words.txt", w in words)
        }
        // Short function-word concatenations must stay out or autocorrect stops
        // fixing the missing space.
        for (w in listOf("andthe", "ofthe", "inthe")) {
            assertFalse("'$w' must not be in en_words.txt", w in words)
        }
    }
}
