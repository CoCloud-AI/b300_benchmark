#!/usr/bin/env bash
# provision_node_610.sh — RECORD of the steps that actually brought this bare
# B300 node (TencentOS Server 4, kernel 6.6.119) up for the driver-610 benchmark
# cycle on 2026-07-08. This is a documented record, not a hands-off installer —
# read each block before re-running on a fresh node. Passwordless sudo assumed;
# Secure Boot disabled (no module signing needed).
#
# Ground truth this node started from: 8x B300 on PCI but NO driver, NO docker,
# NO nvidia-container-toolkit, NO pip/hf CLI, and only 70GB/822GB local disk with
# 4x unmounted 7TB NVMe. Nothing from the 590/595 nodes' stack was present.
set -euo pipefail

### 1. Kernel headers — MUST match the RUNNING kernel exactly ####################
# Trap: `dnf install kernel-devel` grabs 6.6.119-50.12 which mismatches the
# running 49.22 kernel and breaks the DKMS build. Pin the exact version.
sudo dnf install -y "kernel-devel-$(uname -r)" dkms gcc make elfutils-libelf-devel \
                    pkgconf-pkg-config python3-pip dnf-plugins-core

### 2. NVIDIA driver 610.43.02 — open modules via .run installer ################
# The CUDA rhel9 repo's repomd.xml was broken server-side (references a
# primary.xml.gz that 404s on every CDN edge), so `dnf install` from it failed.
# The self-contained .run installer sidesteps repo metadata entirely.
# Newest driver in the repo dir listing was 610.43.02 (branches 590/595/610).
cd /tmp
wget -O NVIDIA-Linux-x86_64-610.43.02.run \
  https://us.download.nvidia.com/XFree86/Linux-x86_64/610.43.02/NVIDIA-Linux-x86_64-610.43.02.run
sudo sh ./NVIDIA-Linux-x86_64-610.43.02.run \
  --silent --dkms --kernel-module-type=open --no-x-check --no-questions
# The silent installer did NOT blacklist nouveau — write it ourselves + rebuild
# initramfs + add kernel cmdline, then REBOOT (nouveau leaves GPUs dirty; only a
# cold init brings nvidia-smi up clean).
printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo grubby --update-kernel=ALL --args="rd.driver.blacklist=nouveau nouveau.modeset=0"
sudo dracut -f
echo ">>> REBOOT NOW, then: nvidia-smi -L  (expect 8x B300 SXM6 AC)"

### 3. NVMe scratch pool — 4x 7TB -> one 28TB striped xfs ########################
# 70GB / root can't hold container images or multi-TB weights. Stripe the NVMe.
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 --chunk=512K --run \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1
sudo mkfs.xfs -f -L nvme-scratch /dev/md0
sudo mkdir -p /mnt/nvme && sudo mount /dev/md0 /mnt/nvme
# reboot-safe:
sudo mdadm --detail --scan | sudo tee /etc/mdadm.conf
echo "UUID=$(sudo blkid -s UUID -o value /dev/md0) /mnt/nvme xfs defaults,nofail,noatime 0 0" | sudo tee -a /etc/fstab
sudo dracut -f
sudo mkdir -p /mnt/nvme/hf-cache /mnt/nvme/docker /mnt/nvme/results-scratch
sudo chown -R ken:ken /mnt/nvme/hf-cache /mnt/nvme/results-scratch

### 3b. NVSwitch fabric manager — MANDATORY on B300 SXM6 (NVL5+) ################
# WITHOUT THIS, nvidia-smi works and shows 8 GPUs, but ANY CUDA context inside a
# container fails with "CUDA error 802: system not yet initialized" and
# `nvidia-smi -q | grep Fabric` shows State=In Progress. The NVSwitch fabric
# must be initialized by nvidia-fabricmanager (version MUST match the driver).
# On this NVL5+ system the fabric manager additionally requires the NVLink
# Subnet Manager (nvlsm) — it aborts with
#   "/opt/nvidia/nvlsm/sbin/nvlsm does not exist ... install nvlsm package".
# Package names: base is nvidia-FABRICMANAGER (no hyphen); the CUDA repo repomd
# was broken so fabricmanager+nscq were installed from direct RPM URLs, while
# nvlsm resolved fine via dnf (DOCA repo present for OFED deps).
cd /tmp
BASE=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64
curl -sS -O $BASE/nvidia-fabricmanager-610.43.02-1.el9.x86_64.rpm
curl -sS -O $BASE/libnvidia-nscq-610.43.02-1.el9.x86_64.rpm
sudo dnf install -y ./nvidia-fabricmanager-610.43.02-1.el9.x86_64.rpm \
                    ./libnvidia-nscq-610.43.02-1.el9.x86_64.rpm
sudo dnf install -y nvlsm          # NVLink Subnet Manager (required on NVL5+)
sudo systemctl enable --now nvidia-fabricmanager
# verify BEFORE trying to serve — must read "Completed" on every GPU:
nvidia-smi -q | grep -A1 '^    Fabric' | grep State | sort | uniq -c
sudo docker run --rm --gpus all lmsysorg/sglang:v0.5.14-cu130-runtime \
  python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())"

### 4. Docker CE + nvidia-container-toolkit #####################################
# TencentOS reports $releasever=4 (invalid for docker's centos repo) -> pin /9/.
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo sed -i 's|/centos/\$releasever/|/centos/9/|g' /etc/yum.repos.d/docker-ce.repo
# nvidia-container-toolkit repo: repo_gpgcheck fails (metadata sig mismatch) -> disable it.
curl -sS https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo sed -i 's/repo_gpgcheck=1/repo_gpgcheck=0/g' /etc/yum.repos.d/nvidia-container-toolkit.repo
# data-root on NVMe BEFORE first start:
sudo mkdir -p /etc/docker
printf '{\n  "data-root": "/mnt/nvme/docker",\n  "log-driver": "json-file",\n  "log-opts": {"max-size": "50m", "max-file": "3"}\n}\n' | sudo tee /etc/docker/daemon.json
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
                    docker-compose-plugin nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || true
sudo systemctl enable --now docker
sudo usermod -aG docker ken   # passwordless docker in NEW shells/tmux (re-login or fresh tmux server)
# smoke test — expect 8 GPUs inside the container:
sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L

### 5. HF download tooling + env ################################################
pip3 install --user -U "huggingface_hub[cli]" hf_transfer
cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mnt/nvme/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

### 6. Pull weights + SGLang image #############################################
sudo docker pull lmsysorg/sglang:v0.5.14-cu130-runtime
bash ~/benchmark/scripts/download_models_610.sh   # run in the `download` tmux session

echo "Provisioning complete. See results_610/metadata/node_info.yaml for the full record."
