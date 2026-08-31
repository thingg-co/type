package com.aosmith.type

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.aosmith.type.model.ModelCatalog
import com.aosmith.type.model.ModelDownloader
import com.aosmith.type.model.ModelSpec
import com.aosmith.type.model.ModelStore
import com.google.android.material.materialswitch.MaterialSwitch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SettingsActivity : AppCompatActivity() {

    private lateinit var prefs: Prefs
    private lateinit var store: ModelStore
    private val downloads = HashMap<String, Job>()
    private val rows = HashMap<String, ModelRow>()

    private class ModelRow(val root: View, val title: TextView, val status: TextView, val progress: ProgressBar, val primary: Button, val secondary: Button)

    private val importLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        lifecycleScope.launch {
            val name = queryName(uri) ?: "imported.gguf"
            findViewById<TextView>(R.id.modelInfo).text = "Importing $name…"
            val result = withContext(Dispatchers.IO) {
                runCatching { contentResolver.openInputStream(uri)!!.let { store.importFrom(it, name) } }
            }
            result.onSuccess { file ->
                prefs.modelId = file.name
                Toast.makeText(this@SettingsActivity, "Imported ${file.name}", Toast.LENGTH_SHORT).show()
            }.onFailure { e ->
                Toast.makeText(this@SettingsActivity, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
            refreshModels()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        // Edge-to-edge is enforced on API 35+: keep the content clear of bars and keyboard.
        val root = findViewById<View>(android.R.id.content)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(
                androidx.core.view.WindowInsetsCompat.Type.systemBars() or
                    androidx.core.view.WindowInsetsCompat.Type.ime(),
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        prefs = Prefs(this)
        store = ModelStore(this)

        findViewById<Button>(R.id.enableButton).setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        findViewById<Button>(R.id.selectButton).setOnClickListener {
            (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
        }
        findViewById<Button>(R.id.importButton).setOnClickListener {
            importLauncher.launch(arrayOf("*/*"))
        }

        bindSwitch(R.id.optAutocorrect, prefs.autocorrect) { prefs.autocorrect = it }
        bindSwitch(R.id.optLive, prefs.liveSuggestions) { prefs.liveSuggestions = it }
        bindSwitch(R.id.optAdaptive, prefs.adaptiveKeys) { prefs.adaptiveKeys = it }
        bindSwitch(R.id.optHaptics, prefs.haptics) { prefs.haptics = it }

        findViewById<TextView>(R.id.version).text = "Type ${BuildConfig.VERSION_NAME} · llama.cpp · everything runs on this device"

        buildModelRows()
    }

    override fun onResume() {
        super.onResume()
        refreshImeStatus()
        refreshModels()
    }

    private fun bindSwitch(id: Int, initial: Boolean, onChange: (Boolean) -> Unit) {
        val sw = findViewById<MaterialSwitch>(id)
        sw.isChecked = initial
        sw.setOnCheckedChangeListener { _, checked -> onChange(checked) }
    }

    private fun refreshImeStatus() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        val enabled = imm.enabledInputMethodList.any { it.packageName == packageName }
        val current = Settings.Secure.getString(contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD) ?: ""
        val selected = current.startsWith("$packageName/")
        findViewById<TextView>(R.id.enableStatus).text = if (enabled) "✓ " + getString(R.string.step_enable_done) else getString(R.string.step_enable_body)
        findViewById<TextView>(R.id.selectStatus).text = if (selected) "✓ " + getString(R.string.step_select_done) else getString(R.string.step_select_body)
        findViewById<Button>(R.id.selectButton).isEnabled = enabled
    }

    private fun buildModelRows() {
        val list = findViewById<LinearLayout>(R.id.modelList)
        list.removeAllViews()
        rows.clear()
        val dp = resources.displayMetrics.density
        for (spec in ModelCatalog.models) {
            val root = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(0, (8 * dp).toInt(), 0, (8 * dp).toInt())
            }
            val title = TextView(this).apply {
                text = "${spec.name} · ${spec.sizeBytes / 1_000_000} MB"
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleSmall)
            }
            val status = TextView(this).apply {
                text = spec.description
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall)
            }
            val progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 1000
                visibility = View.GONE
            }
            val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            val primary = com.google.android.material.button.MaterialButton(this)
            val secondary = com.google.android.material.button.MaterialButton(
                this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle,
            ).apply { (layoutParams as? LinearLayout.LayoutParams)?.marginStart = (8 * dp).toInt() }
            buttons.addView(primary)
            buttons.addView(secondary)
            root.addView(title)
            root.addView(status)
            root.addView(progress)
            root.addView(buttons)
            list.addView(root)
            rows[spec.id] = ModelRow(root, title, status, progress, primary, secondary)
        }
    }

    private fun refreshModels() {
        val active = store.activeFile(prefs)
        for (spec in ModelCatalog.models) {
            val row = rows[spec.id] ?: continue
            val installed = store.isInstalled(spec)
            val downloading = downloads[spec.id]?.isActive == true
            val isActive = installed && active?.name == spec.fileName
            when {
                downloading -> {
                    row.progress.visibility = View.VISIBLE
                    row.primary.text = getString(R.string.model_cancel)
                    row.primary.setOnClickListener { downloads[spec.id]?.cancel(); refreshModels() }
                    row.secondary.visibility = View.GONE
                }
                installed -> {
                    row.progress.visibility = View.GONE
                    row.status.text = if (isActive) "✓ Installed and active" else "Installed"
                    row.primary.text = if (isActive) "Active" else "Use this model"
                    row.primary.isEnabled = !isActive
                    row.primary.setOnClickListener { prefs.modelId = spec.id; refreshModels() }
                    row.secondary.visibility = View.VISIBLE
                    row.secondary.text = getString(R.string.model_delete)
                    row.secondary.setOnClickListener {
                        store.delete(spec)
                        if (prefs.modelId == spec.id) prefs.modelId = null
                        refreshModels()
                    }
                }
                else -> {
                    row.progress.visibility = View.GONE
                    row.status.text = spec.description
                    row.primary.text = getString(R.string.model_download)
                    row.primary.isEnabled = true
                    row.primary.setOnClickListener { startDownload(spec) }
                    row.secondary.visibility = View.GONE
                }
            }
        }
        val info = findViewById<TextView>(R.id.modelInfo)
        val imported = store.installed().filter { f -> ModelCatalog.models.none { it.fileName == f.name } }
        info.text = buildString {
            append(if (active != null) "Keyboard will use: ${active.name}" else getString(R.string.model_status_none))
            if (imported.isNotEmpty()) {
                append("\nImported: ")
                append(imported.joinToString { it.name })
            }
        }
    }

    private fun startDownload(spec: ModelSpec) {
        val row = rows[spec.id] ?: return
        val job = lifecycleScope.launch {
            try {
                ModelDownloader.download(spec, store) { done, total ->
                    val pct = if (total > 0) (done * 1000 / total).toInt() else 0
                    runOnUiThread {
                        row.progress.progress = pct
                        row.status.text = getString(R.string.model_status_downloading, pct / 10)
                    }
                }
                if (prefs.modelId == null) prefs.modelId = spec.id
                Toast.makeText(this@SettingsActivity, "${spec.name} ready", Toast.LENGTH_SHORT).show()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                row.status.text = getString(R.string.model_status_error, e.message ?: e.javaClass.simpleName)
            } finally {
                downloads.remove(spec.id)
                refreshModels()
            }
        }
        downloads[spec.id] = job
        refreshModels()
    }

    private fun queryName(uri: Uri): String? {
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) return c.getString(idx)
        }
        return uri.lastPathSegment
    }
}
