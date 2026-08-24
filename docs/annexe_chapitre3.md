# Annexes — Chapitre 3 : Prototype Minimal de Chatbot Embarqué

**Annexe A : Code source complet —** chatbot.py

```
#!/usr/bin/env python3
"""
chatbot.py — Prototype CLI de chatbot embarqué avec llama.cpp

Cas d'usage couverts :
  1. Chat interactif (Q/R libre)
  2. Résumé de texte
  3. Classification de sentiment

Backend d'inférence :
  Ce prototype pilote directement le binaire natif llama-cli (compilé au
  chapitre 2 via `cmake`/`make`) plutôt que le binding Python llama-cpp-python.
  Ce choix est motivé par un problème de compatibilité documenté et récurrent
  de llama-cpp-python sur Termux/Android (échec de chargement de la
  bibliothèque partagée au runtime — "RuntimeError: Unsupported platform" —
  y compris lorsque la compilation du wheel réussit), alors que le binaire
  llama.cpp natif fonctionne de façon fiable dans ce même environnement.

Usage :
  python chatbot.py --model /path/to/model.gguf --llama-cli ~/llama.cpp/build/bin/llama-cli
  python chatbot.py --model /path/to/model.gguf --task summary
  python chatbot.py --mock   # mode démo sans modèle

Auteur : PFE Master IA — LLMs Embarqués sur Smartphone
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import psutil

from utils import (
    InferenceMetrics, get_ram_usage_mb, get_system_ram_mb,
    format_size, save_metrics, build_chat_prompt, mock_generate,
    safe_cpu_percent,
)

console = Console() if RICH_AVAILABLE else None


# ── Constantes ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un assistant IA embarqué, exécuté localement sur un smartphone "
    "sans connexion internet. Tu réponds de manière concise et précise en français. "
    "Limite tes réponses à 3-4 phrases maximum sauf si l'utilisateur demande plus de détails."
)

TASK_PROMPTS = {
    "chat": "",
    "summary": (
        "Tu es un assistant spécialisé dans le résumé de textes. "
        "Résume le texte fourni en 3-5 points clés, en français, de manière concise."
    ),
    "classification": (
        "Tu es un classificateur de sentiment. Analyse le texte fourni et réponds "
        "uniquement par : [POSITIF], [NÉGATIF], ou [NEUTRE], suivi d'un score de "
        "confiance en pourcentage et d'une justification en une phrase."
    ),
}

BANNER = """
==============================================================
        LLM Embarque -- Prototype CLI (PFE)
   Modeles de langage on-device . llama.cpp (binaire natif)
==============================================================
"""

DEFAULT_LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")


# ── Backend llama-cli (subprocess) ───────────────────────────────────────────

def load_backend(model_path: str, llama_cli_path: str, n_ctx: int = 2048, n_threads: int = 4):
    """Vérifie la disponibilité du binaire llama-cli et du modèle GGUF.

    Contrairement à l'ancienne implémentation basée sur llama-cpp-python
    (qui chargeait le modèle une seule fois en mémoire via `Llama(...)`),
    le backend llama-cli est invoqué en sous-processus indépendant à chaque
    tour de parole (voir generate_response). Il n'y a donc pas de "chargement"
    persistant ici : cette fonction se contente de valider que les deux
    chemins existent avant de démarrer la session.
    """
    if not os.path.exists(llama_cli_path):
        print(f"Binaire llama-cli introuvable : {llama_cli_path}")
        print("   Compiler llama.cpp au préalable (voir chapitre 2, section 1.1.3) :")
        print("   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp")
        print("   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON")
        print("   cmake --build build --config Release -j$(nproc)")
        print("   Puis relancer avec --llama-cli <chemin vers build/bin/llama-cli>")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Modèle introuvable : {model_path}")
        print("   Télécharger un modèle GGUF depuis HuggingFace (voir chapitre 2, section 1.1.4).")
        sys.exit(1)

    model_name = os.path.basename(model_path)
    print(f"Backend : llama-cli natif ({llama_cli_path})")
    print(f"Modèle  : {model_name} ({format_size(model_path)})")
    print(f"Contexte : {n_ctx} tokens | Threads : {n_threads}")

    backend = {
        "llama_cli": llama_cli_path,
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
    }
    return backend, model_name


# Motifs de reconnaissance des statistiques imprimées par llama-cli en fin
# d'exécution (sur stderr), utilisés pour obtenir des métriques précises
# plutôt qu'une simple approximation par horodatage. Le format exact varie
# selon les versions de llama.cpp ("llama_print_timings" ou
# "llama_perf_context_print"), les deux variantes sont donc couvertes.
_PROMPT_EVAL_RE = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_EVAL_RE = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)")


def _parse_llama_cli_stats(stderr_text: str):
    """Tente d'extraire les temps de prefill/decode précis depuis les logs
    de llama-cli. Retourne None si le format n'est pas reconnu, auquel cas
    l'appelant se rabat sur une estimation par horodatage."""
    prompt_match = _PROMPT_EVAL_RE.search(stderr_text)
    eval_match = _EVAL_RE.search(stderr_text)
    if not (prompt_match and eval_match):
        return None
    prefill_ms, prompt_tokens = float(prompt_match.group(1)), int(prompt_match.group(2))
    decode_ms, gen_tokens = float(eval_match.group(1)), int(eval_match.group(2))
    return {
        "prefill_time_s": prefill_ms / 1000.0,
        "prompt_tokens": prompt_tokens,
        "decode_time_s": decode_ms / 1000.0,
        "generated_tokens": gen_tokens,
    }


def generate_response(
    backend: dict,
    prompt: str,
    model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = True,
) -> tuple[str, InferenceMetrics]:
    """Génère une réponse en invoquant llama-cli en sous-processus et mesure
    les performances (débit, latence, empreinte mémoire du sous-processus)."""

    cmd = [
        backend["llama_cli"],
        "-m", backend["model_path"],
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-k", "40",
        "--top-p", "0.95",
        "-c", str(backend["n_ctx"]),
        "-t", str(backend["n_threads"]),
        "--no-display-prompt",
    ]

    cpu_before = safe_cpu_percent(interval=None)
    t_start = time.time()
    first_token_time = None
    full_output = ""

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    # Suivi de la RAM réellement consommée par le sous-processus llama-cli
    # (et non plus par l'interpréteur Python, comme c'était le cas avec le
    # binding llama-cpp-python qui chargeait le modèle dans le même processus).
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps_proc = None
    peak_rss_mb = 0.0

    if stream:
        print("\nAssistant : ", end="", flush=True)

    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            if proc.poll() is not None:
                break
            continue
        if first_token_time is None:
            first_token_time = time.time()
        full_output += ch
        if stream:
            print(ch, end="", flush=True)
        if ps_proc is not None:
            try:
                rss = ps_proc.memory_info().rss / (1024 * 1024)
                peak_rss_mb = max(peak_rss_mb, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass

    stderr_text = proc.stderr.read()
    proc.wait()
    if stream:
        print()

    t_end = time.time()
    cpu_after = safe_cpu_percent(interval=0.1)

    # llama-cli avec --no-display-prompt ne devrait pas ré-imprimer le prompt,
    # mais certaines versions l'ignorent silencieusement : on s'en protège.
    if full_output.startswith(prompt):
        full_output = full_output[len(prompt):]
    full_output = full_output.strip()

    stats = _parse_llama_cli_stats(stderr_text)
    if stats:
        prefill_time = max(stats["prefill_time_s"], 0.001)
        decode_time = max(stats["decode_time_s"], 0.001)
        prompt_tokens = stats["prompt_tokens"]
        generated_tokens = stats["generated_tokens"]
    else:
        # Repli : estimation par horodatage si les stats de llama-cli n'ont
        # pas pu être extraites (format de sortie non reconnu).
        prefill_time = max((first_token_time or t_start) - t_start, 0.01)
        decode_time = max(t_end - (first_token_time or t_start), 0.01)
        prompt_tokens = len(prompt.split()) * 4 // 3  # approximation grossière
        generated_tokens = len(full_output.split()) * 4 // 3 or 1

    metrics = InferenceMetrics(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        total_time_s=t_end - t_start,
        prefill_speed_tps=prompt_tokens / prefill_time,
        decode_speed_tps=generated_tokens / decode_time,
        ram_before_mb=0.0,  # non applicable : le modèle vit dans un sous-processus dédié
        ram_after_mb=peak_rss_mb,
        ram_delta_mb=peak_rss_mb,  # empreinte mémoire pic du sous-processus llama-cli
        cpu_percent=(cpu_before + cpu_after) / 2,
    )

    return full_output, metrics


# ── Modes de tâche ────────────────────────────────────────────────────────────

def run_chat_mode(backend, model_name: str, mock: bool = False):
    """Mode chat interactif."""
    print("\nMode CHAT interactif")
    print("   Tapez votre message et appuyez sur Entrée.")
    print("   Commandes : /résumé, /classify, /stats, /quit\n")
    if not mock:
        print("   Note : chaque tour de parole recharge le modèle (sous-processus")
        print("   llama-cli indépendant) ; un délai de quelques secondes avant la")
        print("   première réponse est donc normal, voir chapitre 3, section 3.10.\n")

    conversation = []
    all_metrics = []

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAu revoir !")
            break

        if not user_input:
            continue

        # Commandes spéciales
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Au revoir !")
            break

        if user_input.lower() == "/stats" and all_metrics:
            m = all_metrics[-1]
            print(m.summary())
            continue

        if user_input.lower().startswith("/résumé "):
            text_to_summarize = user_input[8:]
            prompt = (TASK_PROMPTS["summary"] + "\n\nTexte à résumer :\n" + text_to_summarize)
            conversation_for_prompt = [{"role": "user", "content": prompt}]
        elif user_input.lower().startswith("/classify "):
            text_to_classify = user_input[10:]
            prompt = (TASK_PROMPTS["classification"] + "\n\nTexte : " + text_to_classify)
            conversation_for_prompt = [{"role": "user", "content": prompt}]
        else:
            conversation.append({"role": "user", "content": user_input})
            conversation_for_prompt = conversation

        # Construire le prompt
        full_prompt = build_chat_prompt(
            conversation_for_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        # Générer la réponse
        if mock:
            print("\nAssistant [MOCK] : ", end="", flush=True)
            response, delay = mock_generate(user_input)
            print(response)
        else:
            response, metrics = generate_response(backend, full_prompt, model_name)
            all_metrics.append(metrics)
            save_metrics(metrics)
            print(f"\n   {metrics.decode_speed_tps:.1f} tok/s | "
                  f"{metrics.total_time_s:.1f}s | "
                  f"{metrics.ram_delta_mb:.0f} Mo (pic sous-processus)")

        conversation.append({"role": "assistant", "content": response})
        print()


def run_summary_mode(backend, model_name: str, input_file: Optional[str] = None, mock: bool = False):
    """Mode résumé de texte."""
    print("\nMode RÉSUMÉ de texte")

    if input_file and os.path.exists(input_file):
        with open(input_file) as f:
            text = f.read()
        print(f"   Fichier : {input_file} ({len(text)} caractères)")
    else:
        print("   Entrez le texte à résumer (terminez avec une ligne vide) :")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("Aucun texte fourni.")
        return

    prompt = build_chat_prompt(
        [{"role": "user", "content": f"Résume ce texte en 5 points clés :\n\n{text}"}],
        system_prompt=TASK_PROMPTS["summary"]
    )

    if mock:
        print("\nRésumé [MOCK] :")
        response, _ = mock_generate(text, task="résumé")
        print(response)
    else:
        print("\nRésumé en cours de génération...")
        response, metrics = generate_response(backend, prompt, model_name, stream=True)
        print(metrics.summary())
        save_metrics(metrics)


def run_classification_mode(backend, model_name: str, mock: bool = False):
    """Mode classification de sentiment."""
    print("\nMode CLASSIFICATION de sentiment")
    print("   Entrez le texte à classifier :")

    text = input("Texte : ").strip()
    if not text:
        return

    prompt = build_chat_prompt(
        [{"role": "user", "content": text}],
        system_prompt=TASK_PROMPTS["classification"]
    )

    if mock:
        response, _ = mock_generate(text, task="classification")
        print(f"\nRésultat [MOCK] : {response}")
    else:
        response, metrics = generate_response(
            backend, prompt, model_name, max_tokens=100, stream=False
        )
        print(f"\nRésultat : {response}")
        print(metrics.summary())
        save_metrics(metrics)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prototype CLI — Chatbot LLM embarqué (llama-cli natif)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python chatbot.py --mock                    # Démo sans modèle
  python chatbot.py --model model.gguf        # Chat interactif
  python chatbot.py --model model.gguf --task summary
  python chatbot.py --model model.gguf --task classify
  python chatbot.py --model model.gguf --llama-cli ~/llama.cpp/build/bin/llama-cli
        """
    )
    parser.add_argument("--model", type=str, help="Chemin vers le fichier GGUF")
    parser.add_argument("--llama-cli", type=str, default=DEFAULT_LLAMA_CLI,
                        help=f"Chemin vers le binaire llama-cli (défaut : {DEFAULT_LLAMA_CLI})")
    parser.add_argument("--mock", action="store_true", help="Mode démo sans modèle réel")
    parser.add_argument("--task", choices=["chat", "summary", "classify"],
                        default="chat", help="Tâche à exécuter (défaut: chat)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Taille du contexte")
    parser.add_argument("--threads", type=int, default=4, help="Nombre de threads CPU")
    parser.add_argument("--max-tokens", type=int, default=512, help="Tokens max générés")
    parser.add_argument("--input-file", type=str, help="Fichier texte d'entrée (mode summary)")
    parser.add_argument("--no-stream", action="store_true", help="Désactiver le streaming")

    args = parser.parse_args()

    # Afficher le banner
    print(BANNER)

    # Vérifications
    if not args.mock and not args.model:
        print("Spécifier --model <chemin.gguf> ou --mock pour le mode démo.")
        print("   Exemple : python chatbot.py --mock")
        sys.exit(1)

    # Infos système
    ram = get_system_ram_mb()
    print(f"Système : {ram['total_mb']:.0f} Mo RAM total | "
          f"{ram['available_mb']:.0f} Mo disponibles ({100-ram['percent']:.0f}% libre)")

    if args.mock:
        print("Mode MOCK activé — Réponses simulées (pas de modèle réel chargé)\n")
        backend = None
        model_name = "mock-model"
    else:
        backend, model_name = load_backend(args.model, args.llama_cli, args.n_ctx, args.threads)

    # Lancer la tâche
    if args.task == "chat":
        run_chat_mode(backend, model_name, mock=args.mock)
    elif args.task == "summary":
        run_summary_mode(backend, model_name, args.input_file, mock=args.mock)
    elif args.task == "classify":
        run_classification_mode(backend, model_name, mock=args.mock)


if __name__ == "__main__":
    main()
```

**Annexe B : Code source complet —** benchmark.py

```
#!/usr/bin/env python3
"""
benchmark.py — Script de mesure de performance pour LLMs embarqués

Mesure :
  - Vitesse de prefill (traitement du prompt)
  - Vitesse de decode (génération de tokens)
  - Utilisation RAM
  - Impact thermique (throttling estimé)
  - Comparaison entre configurations

Usage :
  python benchmark.py --model model.gguf
  python benchmark.py --model model.gguf --runs 5
  python benchmark.py --mock
"""

import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Optional

import psutil

from utils import (
    get_ram_usage_mb, get_system_ram_mb, save_metrics, InferenceMetrics,
    safe_cpu_percent,
)

# ── Prompts de benchmark ──────────────────────────────────────────────────────

BENCHMARK_PROMPTS = {
    "short": {
        "prompt": "Quelle est la capitale de la France ?",
        "expected_tokens": 30,
        "description": "Question courte (faible charge)"
    },
    "medium": {
        "prompt": (
            "Explique en 5 points les avantages et inconvénients "
            "de l'intelligence artificielle embarquée sur smartphone."
        ),
        "expected_tokens": 200,
        "description": "Question moyenne (charge modérée)"
    },
    "long": {
        "prompt": (
            "Rédige un tutoriel détaillé expliquant comment installer llama.cpp "
            "sur un smartphone Android via Termux, en incluant toutes les commandes "
            "nécessaires et les dépannages courants."
        ),
        "expected_tokens": 500,
        "description": "Génération longue (forte charge)"
    },
    "reasoning": {
        "prompt": (
            "Un train part de Paris à 8h00 et arrive à Lyon à 10h30. "
            "Un autre train part de Lyon à 9h15 et arrive à Paris à 11h45. "
            "À quelle heure et à quelle distance de Paris se croisent-ils "
            "(distance Paris-Lyon : 512 km) ? Montre tous les calculs."
        ),
        "expected_tokens": 300,
        "description": "Raisonnement arithmétique (test capacité)"
    },
}


@dataclass
class BenchmarkResult:
    prompt_type: str
    description: str
    run_index: int
    prefill_tps: float
    decode_tps: float
    total_time_s: float
    generated_tokens: int
    ram_used_mb: float
    cpu_percent: float
    throttling_detected: bool


def run_single_benchmark(
    llm,
    prompt_key: str,
    run_index: int,
    max_tokens: int = 300,
    prev_decode_tps: Optional[float] = None,
) -> BenchmarkResult:
    """Exécute un benchmark unique et retourne les métriques."""

    prompt_info = BENCHMARK_PROMPTS[prompt_key]
    prompt = prompt_info["prompt"]

    ram_before = get_ram_usage_mb()
    cpu_start = safe_cpu_percent(interval=0.5)
    t_start = time.time()
    first_token_time = None
    token_count = 0

    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,  # Basse température pour reproductibilité
        stream=True,
        stop=["</s>", "<|user|>"],
    ):
        if first_token_time is None:
            first_token_time = time.time()
        token_count += 1
        print(".", end="", flush=True)

    t_end = time.time()
    print()

    ram_after = get_ram_usage_mb()
    cpu_end = safe_cpu_percent(interval=0.1)

    prefill_time = (first_token_time or t_start + 0.1) - t_start
    decode_time = t_end - (first_token_time or t_start + 0.1)
    prompt_tokens = len(prompt.split()) * 4 // 3

    decode_tps = token_count / max(decode_time, 0.01)

    # Détecter le throttling : si le decode ralentit de >15% par rapport au run précédent
    throttling = False
    if prev_decode_tps and decode_tps < prev_decode_tps * 0.85:
        throttling = True

    return BenchmarkResult(
        prompt_type=prompt_key,
        description=prompt_info["description"],
        run_index=run_index,
        prefill_tps=prompt_tokens / max(prefill_time, 0.01),
        decode_tps=decode_tps,
        total_time_s=t_end - t_start,
        generated_tokens=token_count,
        ram_used_mb=ram_after,
        cpu_percent=(cpu_start + cpu_end) / 2,
        throttling_detected=throttling,
    )


def print_results_table(results: list[BenchmarkResult]):
    """Affiche un tableau récapitulatif des résultats."""
    print("\n" + "═" * 80)
    print("  RÉSULTATS DU BENCHMARK")
    print("═" * 80)
    print(f"  {'Type':<12} {'Run':<5} {'Prefill':>10} {'Decode':>10} "
          f"{'Tokens':>8} {'Temps':>8} {'RAM':>10} {'Throttle':>10}")
    print("─" * 80)

    for r in results:
        throttle_str = "⚠️  OUI" if r.throttling_detected else "✅ Non"
        print(f"  {r.prompt_type:<12} {r.run_index:<5} "
              f"{r.prefill_tps:>8.1f}/s  {r.decode_tps:>8.1f}/s  "
              f"{r.generated_tokens:>7}  {r.total_time_s:>6.1f}s  "
              f"{r.ram_used_mb:>8.0f}Mo  {throttle_str:>10}")

    print("─" * 80)

    # Statistiques par type de prompt
    prompt_types = set(r.prompt_type for r in results)
    for pt in sorted(prompt_types):
        group = [r for r in results if r.prompt_type == pt]
        if len(group) > 1:
            decode_vals = [r.decode_tps for r in group]
            print(f"  {pt:<12} μ={statistics.mean(decode_vals):.1f} tok/s  "
                  f"σ={statistics.stdev(decode_vals):.2f}  "
                  f"min={min(decode_vals):.1f}  max={max(decode_vals):.1f}")

    print("═" * 80)


def save_benchmark_results(results: list[BenchmarkResult], model_name: str):
    """Sauvegarde les résultats en JSON."""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results/benchmark_{model_name}_{timestamp}.json"

    data = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": get_system_ram_mb(),
        "results": [asdict(r) for r in results],
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Résultats sauvegardés : {filename}")
    return filename


def run_mock_benchmark(runs: int = 3):
    """Benchmark simulé pour la démonstration."""
    import random
    print("\n🎭 Mode MOCK — Benchmark simulé\n")

    results = []
    for prompt_key, prompt_info in BENCHMARK_PROMPTS.items():
        print(f"  [{prompt_key}] {prompt_info['description']}")
        for i in range(1, runs + 1):
            time.sleep(0.3)  # Simuler le temps d'inférence
            result = BenchmarkResult(
                prompt_type=prompt_key,
                description=prompt_info["description"],
                run_index=i,
                prefill_tps=random.uniform(20, 45),
                decode_tps=random.uniform(10, 20),
                total_time_s=random.uniform(2, 8),
                generated_tokens=random.randint(100, 400),
                ram_used_mb=random.uniform(2000, 2800),
                cpu_percent=random.uniform(60, 95),
                throttling_detected=(i > 2 and random.random() > 0.7),
            )
            results.append(result)
            print(f"    Run {i}/{runs} : {result.decode_tps:.1f} tok/s ✓")

    print_results_table(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM embarqué")
    parser.add_argument("--model", type=str, help="Chemin vers le fichier GGUF")
    parser.add_argument("--mock", action="store_true", help="Mode démo sans modèle")
    parser.add_argument("--runs", type=int, default=3, help="Nombre de répétitions par test")
    parser.add_argument("--prompts", nargs="+",
                        choices=list(BENCHMARK_PROMPTS.keys()) + ["all"],
                        default=["all"], help="Types de prompts à tester")
    parser.add_argument("--threads", type=int, default=4, help="Threads CPU")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Taille du contexte")

    args = parser.parse_args()

    print("=" * 60)
    print("  📊 BENCHMARK — LLM Embarqué sur Smartphone")
    print("=" * 60)

    ram = get_system_ram_mb()
    print(f"\n💻 RAM système : {ram['total_mb']:.0f} Mo total | "
          f"{ram['available_mb']:.0f} Mo disponibles")

    if args.mock:
        run_mock_benchmark(args.runs)
        return

    if not args.model:
        print("❌ Spécifier --model <chemin.gguf> ou utiliser --mock")
        sys.exit(1)

    # Charger le modèle
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python non installé : pip install llama-cpp-python")
        sys.exit(1)

    print(f"\n⏳ Chargement du modèle...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        n_gpu_layers=0,
        verbose=False,
    )
    model_name = os.path.basename(args.model).replace(".gguf", "")
    print(f"✅ Modèle prêt : {model_name}\n")

    # Sélectionner les prompts
    prompt_keys = (
        list(BENCHMARK_PROMPTS.keys())
        if "all" in args.prompts else args.prompts
    )

    # Exécuter les benchmarks
    all_results = []
    for prompt_key in prompt_keys:
        prompt_info = BENCHMARK_PROMPTS[prompt_key]
        print(f"\n🔬 Test [{prompt_key}] — {prompt_info['description']}")
        print(f"   {args.runs} répétition(s)")

        prev_tps = None
        for i in range(1, args.runs + 1):
            print(f"   Run {i}/{args.runs} : ", end="", flush=True)
            result = run_single_benchmark(llm, prompt_key, i, prev_decode_tps=prev_tps)
            prev_tps = result.decode_tps
            all_results.append(result)
            print(f"   ✓ {result.decode_tps:.1f} tok/s | "
                  f"{result.total_time_s:.1f}s | "
                  f"{result.ram_used_mb:.0f} Mo RAM"
                  + (" ⚠️ Throttling!" if result.throttling_detected else ""))

            # Pause entre les runs pour éviter le throttling thermique
            if i < args.runs:
                print("   ⏸️  Pause 10s (refroidissement)...")
                time.sleep(10)

    print_results_table(all_results)
    save_benchmark_results(all_results, model_name)


if __name__ == "__main__":
    main()
```

**Annexe C : Code source complet —** utils.py

```
"""
utils.py — Fonctions utilitaires pour le prototype CLI LLM embarqué.
"""

import time
import psutil
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class InferenceMetrics:
    """Métriques d'une inférence."""
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    prefill_time_s: float       # Temps de traitement du prompt
    decode_time_s: float        # Temps de génération des tokens
    total_time_s: float
    prefill_speed_tps: float    # tokens/s pendant le prefill
    decode_speed_tps: float     # tokens/s pendant le decode
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float
    cpu_percent: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"\n{'─'*50}\n"
            f"  Modèle       : {self.model_name}\n"
            f"  Prompt       : {self.prompt_tokens} tokens\n"
            f"  Généré       : {self.generated_tokens} tokens\n"
            f"  Prefill      : {self.prefill_time_s:.2f}s "
            f"({self.prefill_speed_tps:.1f} tok/s)\n"
            f"  Decode       : {self.decode_time_s:.2f}s "
            f"({self.decode_speed_tps:.1f} tok/s)\n"
            f"  Total        : {self.total_time_s:.2f}s\n"
            f"  RAM delta    : +{self.ram_delta_mb:.0f} Mo "
            f"({self.ram_after_mb:.0f} Mo total)\n"
            f"  CPU usage    : {self.cpu_percent:.1f}%\n"
            f"{'─'*50}"
        )


_cpu_percent_warned = False


def safe_cpu_percent(interval: Optional[float] = None) -> float:
    """Wrapper autour de psutil.cpu_percent() tolérant l'absence d'accès à
    /proc/stat, restreint par le système sur Android non-rooté (Termux).
    Retourne 0.0 (avec un avertissement affiché une seule fois) plutôt que
    de faire planter le programme dans ce cas — la RAM du sous-processus
    llama-cli reste mesurée normalement, seul le CPU% global est affecté."""
    global _cpu_percent_warned
    try:
        return psutil.cpu_percent(interval=interval)
    except (PermissionError, OSError):
        if not _cpu_percent_warned:
            print(
                "\n[Note] CPU% indisponible sur cet appareil "
                "(/proc/stat restreint par Android) — affiché comme 0.0%, "
                "la mesure RAM du sous-processus reste correcte.\n"
            )
            _cpu_percent_warned = True
        return 0.0


def get_ram_usage_mb() -> float:
    """Retourne la RAM utilisée par le processus courant en Mo."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_ram_mb() -> dict:
    """Retourne l'utilisation RAM système."""
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total / (1024 * 1024),
        "used_mb": mem.used / (1024 * 1024),
        "available_mb": mem.available / (1024 * 1024),
        "percent": mem.percent,
    }


def format_size(path: str) -> str:
    """Retourne la taille d'un fichier en format lisible."""
    size = os.path.getsize(path)
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} To"


def save_metrics(metrics: InferenceMetrics, output_file: str = "results/metrics.json"):
    """Sauvegarde les métriques dans un fichier JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Charger l'historique existant
    history = []
    if os.path.exists(output_file):
        with open(output_file) as f:
            history = json.load(f)

    history.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **metrics.to_dict()
    })

    with open(output_file, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def build_chat_prompt(messages: list[dict], system_prompt: str = "") -> str:
    """
    Construit un prompt au format chat compatible avec les modèles instruction.
    Supporte les formats Gemma, LLaMA, ChatML.
    """
    prompt = ""
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"<|user|>\n{content}\n<|assistant|>\n"
        elif role == "assistant":
            prompt += f"{content}\n"

    return prompt


MOCK_RESPONSES = {
    "default": (
        "Je suis un modèle de langage embarqué simulé. "
        "En mode réel, je serais exécuté localement via llama.cpp "
        "sur votre appareil sans connexion internet."
    ),
    "résumé": (
        "Résumé généré localement : Ce texte porte sur les modèles de langage "
        "embarqués sur smartphone. Les points clés sont : (1) la quantification "
        "réduit la taille des modèles de 70 %, (2) INT4 est le meilleur compromis "
        "qualité/performance pour les modèles 3-7B, (3) llama.cpp est le framework "
        "open source de référence pour Android."
    ),
    "classification": "Classification : [POSITIF] — Confiance : 87%",
}


def mock_generate(prompt: str, task: str = "default") -> tuple[str, float]:
    """Génère une réponse simulée avec délai réaliste."""
    import random

    response = MOCK_RESPONSES.get(task, MOCK_RESPONSES["default"])

    # Simuler une latence réaliste (10-15 tok/s sur mobile)
    tokens = len(response.split())
    delay = tokens / random.uniform(10, 15)
    time.sleep(min(delay, 3.0))  # Plafonné à 3s pour la démo

    return response, delay
```

**Annexe D : Dépendances Python —** requirements.txt

```
# Prototype CLI — LLM Embarqué sur Smartphone
# Testé avec Python 3.9, 3.10, 3.11

# Inférence LLM locale (llama.cpp bindings Python)
llama-cpp-python>=0.2.90

# Mesures de performance
psutil>=5.9.0

# Interface CLI améliorée
rich>=13.7.0
prompt_toolkit>=3.0.43

# Optionnel : export/analyse avancée des résultats JSON (non importé par le code actuel,
# à installer séparément si besoin — pandas peut échouer à la compilation sous Termux) :
# pandas>=2.0.0
```

**Annexe E : Synthèse des résultats de validation du prototype**

Résultats bruts obtenus en rejouant le protocole de validation (Parties A à D : questions factuelles et de raisonnement, résumé, classification, observations de session) sur les deux appareils testés. Modèle : LLaMA 3.2 1B Q4_K_M dans les deux cas.

**Galaxy S26 Ultra (Snapdragon 8 Elite) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | Run 1 : 5✅/1⚠️/1❌ — Run 2 : 5✅/0⚠️/2❌ |
| A.2 Raisonnement (5) | Run 1 : 0✅/1⚠️/4❌ — Run 2 : 0✅/1⚠️/4❌ |
| B. Résumé | Run 1 : Bon — Run 2 : Bon — Run 3 : Partiel |
| C. Classification (6) | Run 1 : 6/6 — Run 2 : 6/6 (dont ironie #5 correcte aux deux runs) |
| Saturation du contexte | Confirmée après ~13 échanges (2226 tokens vs 2048) |

**Galaxy A16 (Exynos 1330) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 5✅/0⚠️/2❌ |
| A.2 Raisonnement (5) | 0✅/0⚠️/5❌ |
| B. Résumé | Bon (2 points complets, 2 partiels) |
| C. Classification (6) | 4/6 — échecs sur l'ironie (#5) et le neutre mitigé (#6) |
| Vitesse | Prefill 12,1-52,0 tok/s — Decode 3,9-7,8 tok/s |

Le protocole complet, incluant l'énoncé exact de chaque question, la réponse générée par le modèle et le verdict associé, ainsi que le détail des incidents techniques rencontrés lors de l'installation (compilation `numpy`/`psutil` sous Termux, compatibilité `llama-cpp-python`/Python 3.14, bug `save_metrics()`), est documenté dans le journal de validation associé à ce PFE.

