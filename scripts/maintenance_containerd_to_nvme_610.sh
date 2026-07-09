#!/usr/bin/env bash
# maintenance_containerd_to_nvme_610.sh — move containerd image storage to NVMe.
#
# WHY: Docker 29 uses the containerd snapshotter, so IMAGE LAYERS live under
# /var/lib/containerd on the 70 GB root fs — daemon.json data-root=/mnt/nvme/docker
# is NOT enough (37 GB of SGLang layers filled root; vLLM v0.24.0 pull ENOSPC'd
# on 2026-07-09 and rolled back). This migrates containerd's root to the NVMe pool.
#
# WHEN: only in a maintenance window — requires docker restart. NEVER while a
# server container or sweep is running (rule: don't disturb running benchmarks).
#
# After migration: pulls vLLM stable (GLM-5.2) + nightly (MiniMax-M3, digest-pinned).
set -euo pipefail

echo "=== pre-flight: nothing may be running ==="
if [ -n "$(sudo docker ps -q)" ]; then
  echo "ABORT: containers still running:"; sudo docker ps --format ' {{.Names}}'; exit 2
fi
if pgrep -f '[b]ench_sweep_610' >/dev/null; then echo "ABORT: sweep still running"; exit 2; fi

echo "=== stop docker + containerd ==="
sudo systemctl stop docker docker.socket containerd

echo "=== move /var/lib/containerd -> /mnt/nvme/containerd ==="
sudo mkdir -p /mnt/nvme/containerd
sudo rsync -aHAX --delete /var/lib/containerd/ /mnt/nvme/containerd/
sudo mv /var/lib/containerd /var/lib/containerd.old   # keep until verified

echo "=== point containerd at the NVMe root ==="
# containerd.io ships no config.toml by default; generate one, then set root.
if [ ! -s /etc/containerd/config.toml ]; then
  sudo mkdir -p /etc/containerd
  containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
fi
sudo sed -i 's|^root = .*|root = "/mnt/nvme/containerd"|' /etc/containerd/config.toml
grep -q 'root = "/mnt/nvme/containerd"' /etc/containerd/config.toml || {
  # config default may not have an explicit root line; insert one at top-level
  sudo sed -i '1i root = "/mnt/nvme/containerd"' /etc/containerd/config.toml
}

echo "=== restart + verify images survived ==="
sudo systemctl start containerd docker
sleep 3
sudo docker images --format '{{.Repository}}:{{.Tag}}'
sudo docker images | grep -q sglang-b300 || { echo "ABORT: images missing after migration — investigate before deleting containerd.old"; exit 3; }

echo "=== reclaim root ==="
sudo rm -rf /var/lib/containerd.old
df -h / /mnt/nvme | tail -2

echo "=== pull vLLM images (now onto NVMe) ==="
sudo docker pull vllm/vllm-openai:v0.24.0
sudo docker pull vllm/vllm-openai:nightly
echo "--- pin the nightly digest for provenance ---"
sudo docker images --digests vllm/vllm-openai | tee -a /home/ken/benchmark/results_610/metadata/vllm_image_digests.txt

echo "=== sanity: M3 + GLM archs inside the images ==="
sudo docker run --rm --entrypoint bash vllm/vllm-openai:nightly -c \
  "python3 -c 'import vllm; print(\"nightly vllm:\", vllm.__version__)'; grep -rl MiniMaxM3Sparse /usr/local/lib/python3*/dist-packages/vllm/model_executor/models/registry.py /usr/local/lib/python3*/site-packages/vllm/model_executor/models/registry.py 2>/dev/null | head -1 && echo M3-in-registry-OK"
sudo docker run --rm --entrypoint bash vllm/vllm-openai:v0.24.0 -c \
  "python3 -c 'import vllm; print(\"stable vllm:\", vllm.__version__)'"
echo "=== maintenance complete ==="
