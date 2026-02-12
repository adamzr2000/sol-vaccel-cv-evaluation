#!/usr/bin/env bash
set -e

echo "Stopping docker compose stack..."
docker compose down --remove-orphans

# Remove VNC container if it exists

if docker ps -a --format '{{.Names}}' | grep -qx 'ros-vnc'; then
echo "Removing ros-vnc container..."
docker rm -f ros-vnc >/dev/null
fi

# Remove generated .env file

if [ -f .env ]; then
echo "Removing .env file..."
rm .env
fi

echo "Cleanup complete."
