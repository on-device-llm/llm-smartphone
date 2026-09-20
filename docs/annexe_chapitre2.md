# Annexes — Chapitre 2 : Mise en Œuvre : Déploiement et Analyse des Performances

**Annexes A à E : Code source de l'application Android (ML Kit GenAI / Gemini Nano)**

Le code source complet de l'application de démonstration décrite dans la partie 2 de ce chapitre (sections 2.2 et 2.3) est publié sur le dépôt public du projet, qui constitue la référence à jour et compilable :

```
git clone https://github.com/on-device-llm/llm-smartphone
```

Ce dépôt est privilégié ici à une reproduction intégrale du code, dont la lecture sur papier n'apporte pas de valeur supplémentaire et qui deviendrait obsolète à chaque évolution des bibliothèques ML Kit GenAI (encore en version bêta). Seuls quelques extraits significatifs sont conservés dans les annexes ci-dessous ; les fichiers concernés se trouvent dans le répertoire `prototype-android/` du dépôt :

- `prototype-android/app/build.gradle.kts` (section 2.2.2, Annexe A) : configuration du module, SDK minimum et dépendances ML Kit GenAI.
- `prototype-android/app/src/main/AndroidManifest.xml` (section 2.2.2, Annexe B) : permissions et déclaration de l'activité principale.
- `prototype-android/app/src/main/java/com/pfe/llmchat/LlmViewModel.kt` (section 2.3.1, Annexe C) : gestion de l'état d'inférence, historique de conversation et construction du prompt.
- `prototype-android/app/src/main/java/com/pfe/llmchat/MainActivity.kt` (section 2.3.2, Annexe D) : liaison de l'interface, saisie et observation de l'état.
- `prototype-android/app/src/main/res/layout/activity_main.xml` (section 2.3.3, Annexe E) : disposition de l'écran de conversation.

**Annexe A : Configuration Gradle du projet Android** (`build.gradle.kts`, extrait)

```
android {
    compileSdk = 35
    defaultConfig {
        minSdk = 29
        targetSdk = 35
    }
}
dependencies {
    implementation("com.google.mlkit:genai-common:1.0.0-beta1")
    implementation("com.google.mlkit:genai-inference:1.0.0-beta1")
    // [...] coroutines, lifecycle, RecyclerView, Material
}
```

**Annexe B : Manifeste Android** (`AndroidManifest.xml`, extrait)

```
<!-- Requis pour télécharger le modèle Gemini Nano -->
<uses-permission android:name="android.permission.INTERNET" />
<!-- [...] déclaration de l'activité principale -->
<meta-data
    android:name="com.google.mlkit.genai.ENABLED"
    android:value="true" />
```

**Annexe C : Logique d'inférence** (`LlmViewModel.kt`, extrait)

```
fun initializeModel() {
    viewModelScope.launch {
        _state.value = InferenceState.ModelLoading
        try {
            val availability = LanguageModelInference.checkAvailability()
            if (!availability.isAvailable) {
                _state.value = InferenceState.Error(
                    "Gemini Nano non disponible sur cet appareil. " +
                    "Appareils supportés : Pixel 9/10, Galaxy S25/S26.")
                return@launch
            }
            modelClient = LanguageModelInference.getClient()
            _state.value = InferenceState.ModelReady
        } // [...]
    }
}
```

**Annexe D : Activité principale** (`MainActivity.kt`, extrait)

```
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)
    setupRecyclerView()
    setupInput()
    observeState()
    viewModel.initializeModel()
}
```

**Annexe E : Layout XML** (`activity_main.xml`, extrait)

```
<androidx.constraintlayout.widget.ConstraintLayout ...>
    <TextView
        android:id="@+id/tv_status"
        android:text="Initialisation..." ... />
    <androidx.recyclerview.widget.RecyclerView
        android:id="@+id/rv_chat" ... />
    <!-- [...] zone de saisie (TextInputEditText + bouton Envoyer) -->
</androidx.constraintlayout.widget.ConstraintLayout>
```

Le code complet de chacun de ces fichiers, y compris la classe `ChatAdapter` non reproduite ici, est disponible sur le dépôt indiqué ci-dessus.
