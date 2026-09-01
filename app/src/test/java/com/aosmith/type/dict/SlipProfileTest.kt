package com.aosmith.type.dict

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SlipProfileTest {

    @After fun detach() {
        KeyNeighbors.personal = null
    }

    @Test fun `accepted corrections cheapen the learned pair, with floors`() {
        val p = SlipProfile()
        val base = KeyNeighbors.ADJACENT_COST
        assertEquals(base, p.cost('n', 'm', base), 1e-4f)   // unseen: unchanged
        repeat(30) { p.recordCorrection("nake", "make") }    // saturates at the cap
        val learned = p.cost('n', 'm', base)
        assertTrue("learned $learned should be under $base", learned < base)
        assertTrue("floor holds", learned >= 0.25f)
        // direction matters: m->n never trained
        assertEquals(base, p.cost('m', 'n', base), 1e-4f)
        // distant pairs discount but never under the adjacent base
        repeat(30) { p.recordCorrection("cat", "cut") }      // a->u is not adjacent
        assertTrue(p.cost('a', 'u', 1f) >= 0.6f)
    }

    @Test fun `only clean single substitutions train`() {
        val p = SlipProfile()
        p.recordCorrection("teh", "the")        // transposition: two diffs
        p.recordCorrection("helo", "hello")     // insertion: length differs
        p.recordCorrection("same", "same")      // no diff
        assertEquals(0, p.size())
    }

    @Test fun `undo removes the evidence at triple weight`() {
        val p = SlipProfile()
        repeat(3) { p.recordCorrection("cam", "can") }
        p.recordRevert("cam", "can")
        assertEquals(0, p.size())
    }

    @Test fun `profile survives a save and load`() {
        val p = SlipProfile()
        repeat(10) { p.recordCorrection("nake", "make") }
        val out = ByteArrayOutputStream()
        p.save(out)
        val q = SlipProfile()
        q.load(ByteArrayInputStream(out.toByteArray()))
        assertEquals(p.cost('n', 'm', 0.45f), q.cost('n', 'm', 0.45f), 1e-4f)
    }

    @Test fun `installed profile steers dictionary suggestions`() {
        // "cot" is one distant substitution from both "cat" (o->a) and "cut" (o->u);
        // frequency puts "cat" first until the profile learns this user's o->u slip.
        val dict = Dictionary((TestWords.LIST + listOf("cot", "cut")).asSequence())
        val before = dict.suggest("cot", 3).filterNot { it == "cot" }
        assertEquals("cat", before.first())
        val p = SlipProfile()
        repeat(20) { p.recordCorrection("cot", "cut") }
        KeyNeighbors.personal = p
        val after = dict.suggest("cot", 3).filterNot { it == "cot" }
        KeyNeighbors.personal = null
        assertEquals("learned o->u slip should promote cut: $before -> $after", "cut", after.first())
    }
}
