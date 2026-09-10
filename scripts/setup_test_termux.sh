#!/usr/bin/env bash
# =============================================================================
# setup_test_termux.sh — Installation llama.cpp sur Android/Termux (unifié)
# PFE — LLMs Embarqués sur Smartphone
#
# Remplace setup_test_galaxy_a16_termux.sh et setup_test_galaxy_a26_termux.sh :
# un seul script paramétrable pour tous les appareils testés (A16, A26,
# Infinix Hot 60i, etc.), avec un seul benchmark_complet.sh de référence
# (celui de ce dossier) au lieu d'une copie divergente embarquée inline.
#
# Usage :
#   bash setup_test_termux.sh <nom_appareil> [--with-benchmark] [--model URL] [--out FICHIER]
#
# Exemples :
#   bash setup_test_termux.sh Galaxy_A16
#   bash setup_test_termux.sh Galaxy_A26 --with-benchmark
#   bash setup_test_termux.sh Infinix_Hot60i --with-benchmark \
#       --model https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf \
#       --out gemma-2-2b-it-q4_k_m.gguf
# =============================================================================

set -e

# --------------------------------------------------------------------------- #
# 0. ARGUMENTS
# --------------------------------------------------------------------------- #
DEVICE_NAME="${1:?Usage: bash setup_test_termux.sh <nom_appareil> [--with-benchmark] [--model URL] [--out FICHIER]}"
shift

WITH_BENCHMARK=0
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
BENCHMARK_RAW_URL="https://raw.githubusercontent.com/on-device-llm/llm-smartphone/main/scripts/benchmark_complet.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --with-benchmark) WITH_BENCHMARK=1; shift ;;
    --model) MODEL_URL="$2"; shift 2 ;;
    --out) MODEL_FILE="$2"; shift 2 ;;
    *) echo "Argument inconnu : $1"; exit 1 ;;
  esac
done

MODEL_PATH="$HOME/models/$MODEL_FILE"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Installation llama.cpp — Android (Termux)          ║"
echo "║   Appareil : $DEVICE_NAME"
echo "║   PFE Master IA — LLMs Embarqués sur Smartphone       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️  Architecture : $ARCH (attendu : aarch64) — poursuite quand même"
fi

# --------------------------------------------------------------------------- #
# 1. Paquets Termux
# --------------------------------------------------------------------------- #
echo "=== [1/5] Mise à jour Termux + dépendances ==="
pkg update -y && pkg upgrade -y
pkg install -y git cmake clang make python wget

# --------------------------------------------------------------------------- #
# 2. Patch spawn.h (sysroot Termux) — toujours appliqué, sans danger si déjà
#    présent (on ne l'écrase pas). Corrige un bug connu de compilation
#    llama.cpp sous Termux, quel que soit l'appareil.
# --------------------------------------------------------------------------- #
echo ""
echo "=== [2/5] Vérification/correction spawn.h ==="
SPAWN_H="${PREFIX:-/data/data/com.termux/files/usr}/include/spawn.h"
if [ -f "$SPAWN_H" ]; then
    echo "spawn.h déjà présent, on ne touche pas."
else
    python3 - <<'PYEOF'
import os
p = os.environ.get('PREFIX', '/data/data/com.termux/files/usr')
content = """#pragma once
#include <sys/types.h>
#include <sched.h>
#include <signal.h>

#define POSIX_SPAWN_RESETIDS      1
#define POSIX_SPAWN_SETPGROUP     2
#define POSIX_SPAWN_SETSIGDEF     4
#define POSIX_SPAWN_SETSIGMASK    8
#define POSIX_SPAWN_SETSCHEDPARAM 16
#define POSIX_SPAWN_SETSCHEDULER  32

struct __posix_spawnattr;
typedef struct __posix_spawnattr* posix_spawnattr_t;
struct __posix_spawn_file_actions;
typedef struct __posix_spawn_file_actions* posix_spawn_file_actions_t;

#ifdef __cplusplus
extern "C" {
#endif
int posix_spawn(pid_t*, const char*, const posix_spawn_file_actions_t*, const posix_spawnattr_t*, char* const[], char* const[]);
int posix_spawnp(pid_t*, const char*, const posix_spawn_file_actions_t*, const posix_spawnattr_t*, char* const[], char* const[]);
int posix_spawnattr_init(posix_spawnattr_t*);
int posix_spawnattr_destroy(posix_spawnattr_t*);
int posix_spawnattr_setflags(posix_spawnattr_t*, short);
int posix_spawnattr_getflags(const posix_spawnattr_t*, short*);
int posix_spawnattr_setpgroup(posix_spawnattr_t*, pid_t);
int posix_spawnattr_getpgroup(const posix_spawnattr_t*, pid_t*);
int posix_spawnattr_setsigmask(posix_spawnattr_t*, const sigset_t*);
int posix_spawnattr_getsigmask(const posix_spawnattr_t*, sigset_t*);
int posix_spawnattr_setsigdefault(posix_spawnattr_t*, const sigset_t*);
int posix_spawnattr_getsigdefault(const posix_spawnattr_t*, sigset_t*);
int posix_spawn_file_actions_init(posix_spawn_file_actions_t*);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t*);
int posix_spawn_file_actions_addopen(posix_spawn_file_actions_t*, int, const char*, int, mode_t);
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t*, int);
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t*, int, int);
#ifdef __cplusplus
}
#endif
"""
out = p + '/include/spawn.h'
with open(out, 'w') as f:
    f.write(content)
print(f"spawn.h créé : {out}")
PYEOF
fi

# --------------------------------------------------------------------------- #
# 3. Compilation llama.cpp (-j1, RAM limitée sur mobile)
# --------------------------------------------------------------------------- #
echo ""
echo "=== [3/5] Compilation llama.cpp (-j1) ==="
cd "$HOME"
if [ -d "llama.cpp" ]; then
    echo "llama.cpp déjà présent, git pull..."
    cd llama.cpp && git pull
else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    cd llama.cpp
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=OFF
cmake --build build --config Release -j1

echo "Vérification binaires :"
ls -lh build/bin/llama-bench build/bin/llama-cli

# --------------------------------------------------------------------------- #
# 4. Téléchargement du modèle (paramétrable via --model / --out)
# --------------------------------------------------------------------------- #
echo ""
echo "=== [4/5] Téléchargement du modèle ==="
mkdir -p "$HOME/models"
wget -c -O "$MODEL_PATH" "$MODEL_URL"

# --------------------------------------------------------------------------- #
# 5. Benchmark (optionnel, --with-benchmark) — récupère la version de
#    référence dans scripts/ au lieu d'en embarquer une copie divergente.
# --------------------------------------------------------------------------- #
echo ""
if [ "$WITH_BENCHMARK" -eq 1 ]; then
    echo "=== [5/5] Téléchargement + lancement de benchmark_complet.sh ==="
    wget -c -O "$HOME/benchmark_complet.sh" "$BENCHMARK_RAW_URL"
    chmod +x "$HOME/benchmark_complet.sh"
    bash "$HOME/benchmark_complet.sh" "$MODEL_PATH" "$DEVICE_NAME"
else
    echo "=========================================================="
    echo "  SETUP TERMINÉ !"
    echo "  llama-bench : $HOME/llama.cpp/build/bin/llama-bench"
    echo "  Modèle      : $MODEL_PATH"
    echo ""
    echo "  Prochain appel (benchmark) :"
    echo "  bash ~/benchmark_complet.sh \"$MODEL_PATH\" \"$DEVICE_NAME\""
    echo "  (ou relancez ce script avec --with-benchmark)"
    echo "=========================================================="
fi
