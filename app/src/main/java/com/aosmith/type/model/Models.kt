package com.aosmith.type.model

import android.content.Context
import android.util.Log
import com.aosmith.type.Prefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

data class ModelSpec(
    val id: String,
    val name: String,
    val fileName: String,
    val url: String,
    val sizeBytes: Long,
    val description: String,
)

/**
 * Downloadable models. Sizes are approximate and only used for progress. Order matters: the
 * first entry is the default offered in settings.
 */
object ModelCatalog {
    val models: List<ModelSpec> = listOf(
        ModelSpec(
            id = "smollm2-360m-q8",
            name = "SmolLM2 360M (fast)",
            fileName = "smollm2-360m-instruct-q8_0.gguf",
            url = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
            sizeBytes = 386_404_992L,
            description = "Quick on any recent phone and careful: it almost never changes a correct word.",
        ),
        ModelSpec(
            id = "llama3.2-1b-q4",
            name = "Llama 3.2 1B (balanced)",
            fileName = "llama-3.2-1b-instruct-q4_k_m.gguf",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            sizeBytes = 807_694_464L,
            description = "A step up in accuracy over SmolLM2, still reasonably quick.",
        ),
        ModelSpec(
            id = "qwen2.5-1.5b-q8",
            name = "Qwen2.5 1.5B (recommended)",
            fileName = "qwen2.5-1.5b-instruct-q8_0.gguf",
            url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf",
            sizeBytes = 1_894_532_128L,
            description = "Fixed every typo and sentence in our tests. Big download, worth it on a recent phone.",
        ),
    )

    fun byId(id: String?): ModelSpec? = models.firstOrNull { it.id == id }
}

/** Where model files live and which one is active. */
class ModelStore(context: Context) {
    val dir: File = File(context.filesDir, "models").apply { mkdirs() }

    fun fileFor(spec: ModelSpec): File = File(dir, spec.fileName)

    fun installed(): List<File> = dir.listFiles { f -> f.isFile && f.name.endsWith(".gguf") }?.sortedBy { it.name } ?: emptyList()

    fun isInstalled(spec: ModelSpec): Boolean = fileFor(spec).let { it.exists() && it.length() > 1_000_000 }

    /** The model the keyboard should load: the preferred one if present, else any installed file. */
    fun activeFile(prefs: Prefs): File? {
        val preferred = prefs.modelId
        if (preferred != null) {
            val spec = ModelCatalog.byId(preferred)
            val f = if (spec != null) fileFor(spec) else File(dir, preferred)
            if (f.exists() && f.length() > 1_000_000) return f
        }
        return installed().firstOrNull { it.length() > 1_000_000 }
    }

    fun delete(spec: ModelSpec) {
        fileFor(spec).delete()
        File(dir, spec.fileName + ".part").delete()
    }

    /** Copies a user-picked .gguf into the store. */
    fun importFrom(input: InputStream, name: String): File {
        val safe = name.substringAfterLast('/').replace(Regex("[^A-Za-z0-9._-]"), "_")
        val target = File(dir, if (safe.endsWith(".gguf")) safe else "$safe.gguf")
        val tmp = File(dir, target.name + ".part")
        input.use { src -> FileOutputStream(tmp).use { dst -> src.copyTo(dst, 1 shl 20) } }
        if (!tmp.renameTo(target)) throw IOException("rename failed")
        return target
    }
}

/** Resumable HTTP download into the store. */
object ModelDownloader {
    private const val TAG = "ModelDownloader"

    suspend fun download(spec: ModelSpec, store: ModelStore, onProgress: (downloaded: Long, total: Long) -> Unit) =
        withContext(Dispatchers.IO) {
            val target = store.fileFor(spec)
            val part = File(store.dir, spec.fileName + ".part")
            var offset = if (part.exists()) part.length() else 0L
            var url = URL(spec.url)
            var conn: HttpURLConnection? = null
            try {
                // Follow redirects across hosts by hand (HttpURLConnection will not hop http->https etc).
                var redirects = 0
                while (true) {
                    conn = (url.openConnection() as HttpURLConnection).apply {
                        instanceFollowRedirects = false
                        connectTimeout = 20_000
                        readTimeout = 60_000
                        if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
                        setRequestProperty("User-Agent", "Type-Keyboard/0.1")
                    }
                    val code = conn.responseCode
                    if (code in 300..399 && redirects < 8) {
                        val loc = conn.getHeaderField("Location") ?: throw IOException("redirect without location")
                        url = URL(url, loc)
                        conn.disconnect()
                        redirects++
                        continue
                    }
                    if (code == 200 && offset > 0) {
                        // Server ignored the range; start over.
                        offset = 0
                        part.delete()
                    } else if (code != 200 && code != 206) {
                        throw IOException("HTTP $code")
                    }
                    break
                }
                val c = conn!!
                val remaining = c.contentLengthLong
                val total = if (remaining > 0) remaining + offset else spec.sizeBytes
                c.inputStream.use { input ->
                    FileOutputStream(part, offset > 0).use { out ->
                        val buf = ByteArray(256 * 1024)
                        var done = offset
                        var lastReport = 0L
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            if (done - lastReport > 2_000_000) {
                                lastReport = done
                                onProgress(done, total)
                            }
                        }
                        onProgress(done, total)
                    }
                }
                if (!part.renameTo(target)) throw IOException("could not move file into place")
                Log.i(TAG, "downloaded ${spec.id} (${target.length()} bytes)")
            } finally {
                conn?.disconnect()
            }
        }
}
