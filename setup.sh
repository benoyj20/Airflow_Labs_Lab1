#!/usr/bin/env bash
# Sets up and starts Airflow for this lab.
# Usage: bash setup.sh            first-time setup / start
#        bash setup.sh --reset    wipe the database, logs and outputs, then set up again
set -euo pipefail

cd "$(dirname "$0")"

if ! docker info > /dev/null 2>&1; then
    echo "Docker isn't running. Start Docker Desktop and try again."
    exit 1
fi

if [[ "${1:-}" == "--reset" ]]; then
    echo "Resetting: removing containers, database, logs and pipeline outputs..."
    docker compose down -v
    rm -rf logs working_data
fi

mkdir -p logs plugins config working_data

# Windows has no real Linux user id, so use Airflow's default
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) uid=50000 ;;
    *) uid=$(id -u) ;;
esac
echo "AIRFLOW_UID=$uid" > .env
echo "Wrote .env (AIRFLOW_UID=$uid)"

echo "Initializing Airflow (the first run downloads the images and can take a few minutes)..."
docker compose up airflow-init

echo "Starting Airflow..."
docker compose up -d

echo "Waiting for all services to be healthy (installing packages takes 1-3 minutes)..."
for _ in $(seq 1 60); do
    if ! docker compose ps --format '{{.Status}}' | grep -qE 'starting|unhealthy'; then
        docker compose ps --format 'table {{.Service}}\t{{.Status}}'
        echo
        echo "Airflow is ready: http://localhost:8080 (login airflow2 / airflow2)"
        exit 0
    fi
    sleep 5
done

echo "Services still aren't healthy after 5 minutes. Check: docker compose ps / docker compose logs"
exit 1
