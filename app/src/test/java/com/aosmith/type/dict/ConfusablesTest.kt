package com.aosmith.type.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConfusablesTest {

    @Test fun `alternatives exclude the word itself and work in any case`() {
        assertEquals(listOf("than"), Confusables.alternativesOf("then"))
        assertEquals(listOf("then"), Confusables.alternativesOf("Than"))
        assertEquals(listOf("their", "they're"), Confusables.alternativesOf("there"))
        assertEquals(listOf("there", "they're"), Confusables.alternativesOf("THEIR"))
        assertTrue(Confusables.alternativesOf("banana").isEmpty())
        assertTrue(Confusables.alternativesOf("").isEmpty())
    }

    @Test fun `every member of a set resolves to the others`() {
        for (w in listOf("your", "you're", "its", "it's", "passed", "past")) {
            assertTrue("$w should be confusable", Confusables.alternativesOf(w).isNotEmpty())
        }
    }
}
