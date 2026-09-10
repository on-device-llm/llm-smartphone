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
