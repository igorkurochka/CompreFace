#!/bin/bash

# Docker Cleanup Script
# This script removes unused Docker images and containers to free up disk space

set -e  # Exit on any error

echo "🐳 Docker Cleanup Script"
echo "========================"

# Function to print section headers
print_section() {
    echo ""
    echo "📋 $1"
    echo "----------------------------------------"
}

# Function to print disk usage before and after
print_disk_usage() {
    echo "💾 Current disk usage:"
    df -h / | grep -E "(Filesystem|/dev/)"
    echo ""
}

# Function to print Docker disk usage
print_docker_usage() {
    echo "🐳 Docker disk usage:"
    docker system df
    echo ""
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker is not running or not accessible"
    exit 1
fi

print_section "Initial Disk Usage"
print_disk_usage
print_docker_usage

# Store initial disk usage for comparison
INITIAL_DISK_USAGE=$(df / | awk 'NR==2 {print $3}')

print_section "Cleaning Up Stopped Containers"
echo "Removing stopped containers..."
STOPPED_CONTAINERS=$(docker container ls -a --filter "status=exited" --filter "status=created" -q)
if [ -n "$STOPPED_CONTAINERS" ]; then
    echo "Found $(echo "$STOPPED_CONTAINERS" | wc -l) stopped containers"
    docker container prune -f
    echo "✅ Stopped containers removed"
else
    echo "ℹ️  No stopped containers found"
fi

print_section "Cleaning Up Unused Networks"
echo "Removing unused networks..."
UNUSED_NETWORKS=$(docker network ls --filter "type=custom" -q)
if [ -n "$UNUSED_NETWORKS" ]; then
    echo "Found $(echo "$UNUSED_NETWORKS" | wc -l) unused networks"
    docker network prune -f
    echo "✅ Unused networks removed"
else
    echo "ℹ️  No unused networks found"
fi

print_section "Cleaning Up Dangling Images"
echo "Removing dangling images..."
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING_IMAGES" ]; then
    echo "Found $(echo "$DANGLING_IMAGES" | wc -l) dangling images"
    docker image prune -f
    echo "✅ Dangling images removed"
else
    echo "ℹ️  No dangling images found"
fi

print_section "Cleaning Up Unused Volumes"
echo "Removing unused volumes..."
UNUSED_VOLUMES=$(docker volume ls -q)
if [ -n "$UNUSED_VOLUMES" ]; then
    echo "Found $(echo "$UNUSED_VOLUMES" | wc -l) volumes"
    docker volume prune -f
    echo "✅ Unused volumes removed"
else
    echo "ℹ️  No volumes found"
fi

print_section "Full System Cleanup"
echo "Performing full system cleanup (removes all unused images, containers, networks, and volumes)..."
read -p "⚠️  This will remove ALL unused Docker resources. Continue? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker system prune -a -f --volumes
    echo "✅ Full system cleanup completed"
else
    echo "⏭️  Skipping full system cleanup"
fi

print_section "Final Disk Usage"
print_disk_usage
print_docker_usage

# Calculate space freed
FINAL_DISK_USAGE=$(df / | awk 'NR==2 {print $3}')
SPACE_FREED=$((INITIAL_DISK_USAGE - FINAL_DISK_USAGE))

if [ $SPACE_FREED -gt 0 ]; then
    echo "🎉 Cleanup completed successfully!"
    echo "💾 Space freed: ${SPACE_FREED}KB ($(echo "scale=2; $SPACE_FREED/1024/1024" | bc)GB)"
else
    echo "ℹ️  Cleanup completed. No significant space was freed."
fi

echo ""
echo "✨ Docker cleanup script finished!" 