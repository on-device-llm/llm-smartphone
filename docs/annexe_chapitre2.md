# Annexes — Chapitre 2 : Mise en Œuvre : Déploiement et Analyse des Performances

**Annexe A : Configuration Gradle du projet Android (**build.gradle.kts**)**

```
// build.gradle.kts (module app)
android {
compileSdk = 35
defaultConfig {
minSdk = 29
targetSdk = 35
}
buildFeatures {
viewBinding = true
}
compileOptions {
sourceCompatibility = JavaVersion.VERSION_17
targetCompatibility = JavaVersion.VERSION_17
}
kotlinOptions {
jvmTarget = "17"
}
}
dependencies {
// ML Kit GenAI — Summarization, Proofreading, Free-form inference
implementation("com.google.mlkit:genai-common:1.0.0-beta1")
implementation("com.google.mlkit:genai-inference:1.0.0-beta1")
// Coroutines pour l'inférence asynchrone
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
// UI
implementation("androidx.recyclerview:recyclerview:1.3.2")
implementation("androidx.constraintlayout:constraintlayout:2.1.4")
implementation("com.google.android.material:material:1.11.0")
}
```

**Annexe B : Manifeste Android (**AndroidManifest.xml**)**

```
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
<\!-- Requis pour télécharger le modèle Gemini Nano -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-library
android:name="android.ext.adservices"
android:required="false" />
<application
android:name=".LlmChatApplication"
android:allowBackup="true"
android:label="@string/app_name"
android:theme="@style/Theme.LlmChat">
<activity
android:name=".MainActivity"
android:exported="true">
<intent-filter>
<action android:name="android.intent.action.MAIN" />
<category android:name="android.intent.category.LAUNCHER" />
</intent-filter>
</activity>
<meta-data
android:name="com.google.mlkit.genai.ENABLED"
android:value="true" />
</application>
</manifest>
```

**Annexe C : Code source complet —** LlmViewModel.kt

```
// LlmViewModel.kt
package com.PFE.llmchat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.mlkit.genai.inference.LanguageModelInference
import com.google.mlkit.genai.inference.InferenceOptions
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
data class ChatMessage(
val content: String,
val isUser: Boolean,
val timestamp: Long = System.currentTimeMillis(),
val latencyMs: Long = 0
)
sealed class InferenceState {
object Idle : InferenceState()
object ModelLoading : InferenceState()
object ModelReady : InferenceState()
data class Generating(val partialText: String) : InferenceState()
data class Error(val message: String) : InferenceState()
}
class LlmViewModel : ViewModel() {
private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
val messages: StateFlow<List<ChatMessage>> = _messages
private val _state = MutableStateFlow<InferenceState>(InferenceState.Idle)
val state: StateFlow<InferenceState> = _state
private var modelClient: LanguageModelInference? = null
fun initializeModel() {
viewModelScope.launch {
_state.value = InferenceState.ModelLoading
try {
val availability = LanguageModelInference.checkAvailability()
if (\!availability.isAvailable) {
_state.value = InferenceState.Error(
"Gemini Nano non disponible sur cet appareil. " +
"Appareils supportés : Pixel 9/10, Galaxy S25/S26."
)
return@launch
}
modelClient = LanguageModelInference.getClient()
_state.value = InferenceState.ModelReady
} catch (e: Exception) {
_state.value = InferenceState.Error("Erreur initialisation : ${e.message}")
}
}
}
fun sendMessage(userInput: String) {
val client = modelClient ?: return
val startTime = System.currentTimeMillis()
_messages.value = _messages.value + ChatMessage(userInput, isUser = true)
viewModelScope.launch {
_state.value = InferenceState.Generating("")
val sb = StringBuilder()
try {
val options = InferenceOptions.Builder()
.setMaxTokens(512)
.setTemperature(0.7f)
.setTopK(40)
.build()
client.generateResponseAsync(
prompt = buildPrompt(userInput),
options = options,
onPartialResult = { partial ->
sb.append(partial)
_state.value = InferenceState.Generating(sb.toString())
},
onComplete = { _ ->
val latency = System.currentTimeMillis() - startTime
_messages.value = _messages.value + ChatMessage(
content = sb.toString(),
isUser = false,
latencyMs = latency
)
_state.value = InferenceState.ModelReady
},
onError = { e ->
_state.value = InferenceState.Error("Erreur inférence : ${e.message}")
}
)
} catch (e: Exception) {
_state.value = InferenceState.Error(e.message ?: "Erreur inconnue")
}
}
}
private fun buildPrompt(userInput: String): String {
val history = _messages.value.takeLast(6)
val sb = StringBuilder()
history.forEach { msg ->
if (msg.isUser) sb.append("User: ${msg.content}\n")
else sb.append("Assistant: ${msg.content}\n")
}
sb.append("User: $userInput\nAssistant:")
return sb.toString()
}
override fun onCleared() {
super.onCleared()
modelClient?.close()
}
}
```

**Annexe D : Code source complet —** MainActivity.kt

```
// MainActivity.kt
package com.PFE.llmchat
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.PFE.llmchat.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
class MainActivity : AppCompatActivity() {
private lateinit var binding: ActivityMainBinding
private val viewModel: LlmViewModel by viewModels()
private lateinit var chatAdapter: ChatAdapter
override fun onCreate(savedInstanceState: Bundle?) {
super.onCreate(savedInstanceState)
binding = ActivityMainBinding.inflate(layoutInflater)
setContentView(binding.root)
setupRecyclerView()
setupInput()
observeState()
viewModel.initializeModel()
}
private fun setupRecyclerView() {
chatAdapter = ChatAdapter()
binding.rvChat.apply {
layoutManager = LinearLayoutManager(this@MainActivity).apply {
stackFromEnd = true
}
adapter = chatAdapter
}
}
private fun setupInput() {
binding.btnSend.setOnClickListener { sendMessage() }
binding.etInput.setOnEditorActionListener { _, actionId, _ ->
if (actionId == EditorInfo.IME_ACTION_SEND) {
sendMessage(); true
} else false
}
}
private fun sendMessage() {
val text = binding.etInput.text?.toString()?.trim() ?: return
if (text.isEmpty()) return
binding.etInput.text?.clear()
viewModel.sendMessage(text)
}
private fun observeState() {
lifecycleScope.launch {
viewModel.state.collect { state ->
when (state) {
is InferenceState.ModelLoading ->
binding.tvStatus.text = "Chargement de Gemini Nano..."
is InferenceState.ModelReady ->
binding.tvStatus.text = "Gemini Nano prêt"
is InferenceState.Generating ->
binding.tvStatus.text = "Génération en cours..."
is InferenceState.Error ->
binding.tvStatus.text = "${state.message}"
else -> {}
}
}
}
lifecycleScope.launch {
viewModel.messages.collect { messages ->
chatAdapter.submitList(messages)
binding.rvChat.smoothScrollToPosition(messages.size)
}
}
}
}
```

**Annexe E : Layout XML —** activity_main.xml

```
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
xmlns:android="http://schemas.android.com/apk/res/android"
xmlns:app="http://schemas.android.com/apk/res-auto"
android:layout_width="match_parent"
android:layout_height="match_parent">
<TextView
android:id="@+id/tv_status"
android:layout_width="match_parent"
android:layout_height="wrap_content"
android:padding="8dp"
android:text="Initialisation..."
android:textSize="12sp"
app:layout_constraintTop_toTopOf="parent"/>
<androidx.recyclerview.widget.RecyclerView
android:id="@+id/rv_chat"
android:layout_width="match_parent"
android:layout_height="0dp"
android:padding="8dp"
app:layout_constraintTop_toBottomOf="@id/tv_status"
app:layout_constraintBottom_toTopOf="@id/input_layout"/>
<LinearLayout
android:id="@+id/input_layout"
android:layout_width="match_parent"
android:layout_height="wrap_content"
android:orientation="horizontal"
android:padding="8dp"
app:layout_constraintBottom_toBottomOf="parent">
<com.google.android.material.textfield.TextInputEditText
android:id="@+id/et_input"
android:layout_width="0dp"
android:layout_height="wrap_content"
android:layout_weight="1"
android:hint="Posez votre question..."
android:imeOptions="actionSend"
android:inputType="textMultiLine"
android:maxLines="3"/>
<com.google.android.material.button.MaterialButton
android:id="@+id/btn_send"
android:layout_width="wrap_content"
android:layout_height="wrap_content"
android:layout_marginStart="8dp"
android:text="Envoyer"/>
</LinearLayout>
</androidx.constraintlayout.widget.ConstraintLayout>
```

