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

# Seuil de plausibilité pour decode_time_s : au-delà, la mesure est presque
# toujours le symptôme d'une suspension du processus (téléphone verrouillé,
# Doze/app-standby geler l'appel bloquant proc.stdout.read(1)) plutôt qu'une
# inférence réellement longue — voir journal_validation_prototype.md,
# incidents #5/#7 (decode_time_s observé jusqu'à 62 810 s soit 17,4 h).
MAX_PLAUSIBLE_DECODE_S = 120.0


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
# selon les versions de llama.cpp ("llama_print_timings", "llama_perf_context_print",
# ou "slot print_timing: id X | task Y | ..." sur les builds récents basés sur
# le serveur HTTP interne) — les regex ne sont volontairement PAS ancrées à un
# préfixe précis (pas de "^" ni de littéral de préfixe) : elles cherchent
# "prompt eval time"/"eval time" n'importe où dans le texte, ce qui les rend
# robustes à ce genre de changement de préfixe entre builds.
# Nombre avec séparateur de milliers optionnel (ex. "1,234.56"), pour couvrir
# les variantes d'affichage de llama.cpp selon les locales/versions.
#
# IMPORTANT : sur les builds récents (serveur HTTP interne, voir cmd ci-dessous),
# ces lignes ne sont émises DU TOUT sur stderr qu'avec --perf ET -lv 3 (niveau
# de verbosité) — sans ces flags, stderr ne contient aucune statistique quel
# que soit le regex utilisé (vérifié empiriquement sur Galaxy S26 Ultra,
# build b10154-0e4a03622 : 0 ligne de stats sans -lv, 18 lignes propres avec
# -lv 3, plusieurs milliers avec -v/--log-verbose qui active aussi le debug
# complet du chargement des tenseurs — inutile et coûteux en comparaison).
_NUM = r"[\d,]+\.?\d*"
_PROMPT_EVAL_RE = re.compile(rf"prompt eval time\s*=\s*({_NUM})\s*ms\s*/\s*(\d+)\s*tokens?")
_EVAL_RE = re.compile(rf"(?<!prompt )eval time\s*=\s*({_NUM})\s*ms\s*/\s*(\d+)\s*(?:runs?|tokens?)")

# Balise ajoutée par build_chat_prompt() (utils.py) juste avant l'endroit où
# la réponse du modèle doit commencer, pour le tour de parole courant.
_ASSISTANT_TAG = "<|assistant|>"
# Bandeau de vitesse affiché par llama-cli après la réponse, ex.
# "[ Prompt: 207.0 t/s | Generation: 60.6 t/s ]", suivi de "Exiting...".
# DOTALL : une fois qu'on voit "[ Prompt:", tout le reste (bandeau + Exiting...)
# est du bruit de fin de sous-processus, pas de la réponse du modèle.
_TRAILING_STATS_RE = re.compile(r"\n?\[\s*Prompt\s*:.*", re.DOTALL)


def _extract_assistant_reply(full_output: str, prompt: str) -> str:
    """Isole la réponse effective du modèle dans la sortie brute de llama-cli.

    Sur ce build, --no-display-prompt n'empêche ni l'affichage du bandeau de
    démarrage (ASCII art, "available commands", etc.) ni l'écho du prompt
    complet (avec nos balises <|system|>/<|user|>/<|assistant|>) avant la
    réponse — conséquence du mode conversation forcé par -st (voir le
    commentaire sur `cmd` dans generate_response). Sans ce nettoyage,
    full_output contiendrait tout ce bruit en plus de la réponse, qui se
    retrouverait affiché à l'utilisateur ET enregistré tel quel dans
    l'historique de conversation (contaminant les tours suivants).

    On extrait donc tout ce qui suit la DERNIÈRE occurrence de la balise
    <|assistant|> (celle ajoutée pour le tour courant — les tours précédents
    de la conversation peuvent en contenir d'autres, plus tôt dans le texte),
    puis on retire le bandeau de vitesse et le "Exiting..." final."""
    idx = full_output.rfind(_ASSISTANT_TAG)
    if idx != -1:
        full_output = full_output[idx + len(_ASSISTANT_TAG):]
    elif full_output.startswith(prompt):
        # Repli : cas où le prompt est échoué tel quel, sans balise (autre
        # build, ou mode non-conversation qui n'insère pas ce bruit).
        full_output = full_output[len(prompt):]
    full_output = _TRAILING_STATS_RE.sub("", full_output)
    full_output = re.sub(r"\n?Exiting\.\.\.\s*$", "", full_output)
    return full_output.strip()


def _parse_llama_cli_stats(stderr_text: str):
    """Tente d'extraire les temps de prefill/decode précis depuis les logs
    de llama-cli (couvre les préfixes "llama_print_timings" et
    "llama_perf_context_print", singulier/pluriel de "tokens"/"runs", et les
    nombres avec séparateur de milliers). Retourne None si le format n'est
    toujours pas reconnu."""
    prompt_match = _PROMPT_EVAL_RE.search(stderr_text)
    eval_match = _EVAL_RE.search(stderr_text)
    if not (prompt_match and eval_match):
        return None
    prefill_ms = float(prompt_match.group(1).replace(",", ""))
    prompt_tokens = int(prompt_match.group(2))
    decode_ms = float(eval_match.group(1).replace(",", ""))
    gen_tokens = int(eval_match.group(2))
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
    les performances (débit, latence, empreinte mémoire du sous-processus).

    Lève RuntimeError si l'appel a échoué ou si la mesure obtenue n'est pas
    exploitable (sortie vide, code retour non nul, statistiques llama-cli non
    reconnues, ou decode_time_s invraisemblable). L'appelant doit alors
    traiter ce tour comme un échec plutôt que de sauvegarder une métrique
    fabriquée dans results/metrics.json (voir save_metrics)."""

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
        # -no-cnv seul ne suffit pas : sur certains builds, llama-cli entre
        # quand même dans son propre mode conversation dès que le modèle
        # expose un chat template, et reste vivant après le premier tour à
        # lire l'entrée standard — cassant le modèle "un sous-processus
        # indépendant par tour" sur lequel repose toute la mesure de
        # métriques (observé empiriquement : le sous-processus absorbe
        # plusieurs tours et ne se termine jamais, probablement la vraie
        # cause du bug historique de save_metrics() en mode interactif,
        # voir chapitre 3 section 3.8.3). -st/--single-turn est le flag
        # documenté précisément pour ce cas (prompt fourni via -p) :
        # "will not be interactive if first turn is predefined with
        # --prompt" — combiné à -no-cnv en ceinture-bretelles.
        "-no-cnv",
        "-st",
        # Même avec -st, le mode conversation de ce build affiche sa sortie
        # (réponse, bandeau de vitesse, "Exiting...") en écrivant
        # directement sur le TTY plutôt que sur stdout/stderr redirigés —
        # invisible pour un script qui capture ces flux comme ici (observé
        # empiriquement : returncode=0 mais stdout ET stderr vides).
        # --simple-io force une IO basique compatible avec un sous-processus
        # ("better compatibility in subprocesses and limited consoles").
        "--simple-io",
        # Sur ce build (serveur HTTP interne), les statistiques précises de
        # prefill/decode ("slot print_timing: ... prompt eval time = ...")
        # ne sont émises sur stderr qu'à partir du niveau de verbosité 3 —
        # --perf seul ou le niveau par défaut ne produisent AUCUNE ligne de
        # stats (vérifié empiriquement : 0 ligne sans -lv, 18 lignes propres
        # avec -lv 3, contre plusieurs milliers avec -v qui active en plus
        # tout le debug de chargement des tenseurs, inutile ici et coûteux).
        "--perf",
        "-lv", "3",
    ]

    cpu_before = safe_cpu_percent(interval=None)
    t_start = time.time()
    first_token_time = None
    full_output = ""

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        # Sans stdin explicite, le sous-processus hérite du terminal du
        # parent : si -no-cnv échouait pour une raison quelconque (build
        # différent, flag renommé), llama-cli pourrait quand même lire les
        # messages tapés en avance par l'utilisateur au lieu de laisser
        # Python les traiter tour par tour. /dev/null élimine ce risque.
        stdin=subprocess.DEVNULL,
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
    # mais le mode conversation forcé par -st l'échoue quand même (banner,
    # balises <|system|>/<|user|>/<|assistant|>, bandeau de vitesse) : on
    # isole la réponse effective (voir _extract_assistant_reply ci-dessus).
    full_output = _extract_assistant_reply(full_output, prompt)

    # Fix #1 : un code retour non nul ou une sortie vide signalent un échec
    # réel de la génération (crash, argument invalide, binaire tué) — ce
    # n'est pas une réponse valide de longueur nulle, et ne doit donc pas
    # produire une InferenceMetrics fabriquée à partir de zéros/valeurs par
    # défaut.
    if proc.returncode != 0 or not full_output:
        raise RuntimeError(
            f"Échec de génération llama-cli (code retour={proc.returncode}, "
            f"sortie vide={not full_output}). "
            f"Stderr (fin) : {stderr_text[-300:] if stderr_text else '(vide)'}"
        )

    stats = _parse_llama_cli_stats(stderr_text)
    if not stats:
        # Fix #2 : on ne se rabat plus silencieusement sur une estimation
        # par horodatage (prompt.split()*4//3, etc.) — elle est imprécise et,
        # en cas de mise en veille du téléphone (Doze/app-standby) pendant
        # l'attente bloquante sur proc.stdout.read(1), elle peut produire un
        # decode_time_s aberrant (observé : 62 810 s soit 17,4 h — voir
        # journal_validation_prototype.md, incidents #5/#7). On abandonne
        # la mesure plutôt que de l'enregistrer.
        raise RuntimeError(
            "Statistiques llama-cli non reconnues dans stderr (format de "
            "sortie inattendu) — mesure abandonnée plutôt qu'estimée."
        )

    prefill_time = max(stats["prefill_time_s"], 0.001)
    decode_time = max(stats["decode_time_s"], 0.001)
    prompt_tokens = stats["prompt_tokens"]
    generated_tokens = stats["generated_tokens"]

    # Estimation automatique du temps de rechargement du modèle (méthode 1,
    # voir chapitre 3, section 3.8.3, limitation #2) : le prototype relance
    # un sous-processus llama-cli indépendant à chaque tour de parole, ce qui
    # implique un rechargement complet du modèle depuis le stockage avant
    # toute génération. Ce coût n'était jusqu'ici pas isolé, faute d'avoir
    # été chronométré séparément. On l'estime ici sans instrumentation
    # supplémentaire ni chronométrage manuel, à partir de deux horodatages
    # déjà mesurés par ce même sous-processus (t_start, first_token_time) et
    # du temps de prefill déjà extrait précisément des statistiques de
    # llama-cli (stats["prefill_time_s"]) :
    #
    #   load_time_s ≈ (first_token_time - t_start) - prefill_time
    #
    # c'est-à-dire : tout le temps écoulé entre le lancement du sous-processus
    # et l'apparition du premier caractère de la réponse, moins la part de ce
    # délai déjà expliquée par le traitement du prompt (prefill). Cette
    # estimation reste approximative (elle inclut aussi le coût de démarrage
    # du sous-processus lui-même, marginal en comparaison du chargement du
    # modèle) mais elle est automatique, reproductible sur chaque tour, et ne
    # nécessite aucun chronométrage manuel.
    load_time = max((first_token_time - t_start) - prefill_time, 0.0)

    # Fix #3 : garde-fou de plausibilité sur decode_time_s.
    if decode_time > MAX_PLAUSIBLE_DECODE_S:
        raise RuntimeError(
            f"decode_time_s={decode_time:.1f}s dépasse le seuil de "
            f"plausibilité ({MAX_PLAUSIBLE_DECODE_S:.0f}s) — probable "
            f"suspension du processus (Doze/app-standby) pendant la "
            f"génération ; mesure ignorée."
        )

    metrics = InferenceMetrics(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        load_time_s=load_time,
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
            try:
                response, metrics = generate_response(backend, full_prompt, model_name)
            except RuntimeError as e:
                print(f"\n✗ Échec de la génération : {e}")
                print("   Ce tour n'est pas comptabilisé (aucune métrique sauvegardée).")
                continue
            all_metrics.append(metrics)
            save_metrics(metrics)
            print(f"\n   {metrics.decode_speed_tps:.1f} tok/s | "
                  f"{metrics.total_time_s:.1f}s (dont ~{metrics.load_time_s:.1f}s chargement estimé) | "
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
        try:
            response, metrics = generate_response(backend, prompt, model_name, stream=True)
        except RuntimeError as e:
            print(f"\n✗ Échec de la génération : {e}")
            return
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
        try:
            response, metrics = generate_response(
                backend, prompt, model_name, max_tokens=100, stream=False
            )
        except RuntimeError as e:
            print(f"\n✗ Échec de la génération : {e}")
            return
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
