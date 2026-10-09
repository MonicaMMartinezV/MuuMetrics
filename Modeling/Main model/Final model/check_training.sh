#!/usr/bin/env bash
# =============================================================
# check_training.sh
# Muestra el estado y las últimas líneas de log del contenedor
# de entrenamiento MÁS RECIENTE (corriendo o ya terminado),
# sin necesidad de acordarse de su nombre exacto.
#
# Uso:
#   ./check_training.sh          -> últimas 30 líneas
#   ./check_training.sh -f       -> sigue el log en vivo (Ctrl+C para salir)
#   ./check_training.sh -n 100   -> últimas 100 líneas
# =============================================================
set -euo pipefail

FOLLOW=false
LINES=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f) FOLLOW=true; shift ;;
        -n) LINES="$2"; shift 2 ;;
        *) shift ;;
    esac
done

NAME=$(docker ps -a --filter "name=muumetrics-training" \
    --format "{{.CreatedAt}}\t{{.Names}}" | sort -r | head -n1 | cut -f2)

if [[ -z "$NAME" ]]; then
    echo "No encontré ningún contenedor 'muumetrics-training*'."
    exit 1
fi

STATUS=$(docker ps -a --filter "name=$NAME" --format "{{.Status}}")
echo "=== Contenedor más reciente: $NAME ==="
echo "=== Estado: $STATUS ==="
echo ""

if $FOLLOW; then
    docker logs -f --tail "$LINES" "$NAME"
else
    docker logs --tail "$LINES" "$NAME"
fi
