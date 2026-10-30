#!/bin/bash
# 下载 ewtherka 复现所需的 17 个公开源头（golem 已在 pulled_golem/）
cd "/c/Users/GEM07/kaggle/s6e10-airline"
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
mkdir -p pulled_members
LOG=pulled_members/download.log
: > "$LOG"

echo "=== datasets ===" | tee -a "$LOG"
while IFS='|' read -r key handle; do
  [ -z "$key" ] && continue
  dir="pulled_members/$key"; mkdir -p "$dir"
  echo "--- dataset $key = $handle ---" | tee -a "$LOG"
  kaggle datasets download "$handle" -p "$dir" --unzip 2>&1 | tail -2 | tee -a "$LOG"
done <<'EOF'
c124|cdeotte/s6e10-124-public-oof-and-test-predictions
tfm|cdeotte/s6e10-tfm-oof-and-test-predictions
abdullah|abdullahsafwan333/oof-dataset
sadam|sadamtorres/ps-s6e10-airline-satisfaction-artifacts
forge|amanatar/s6e10-forge-own-pairs
sachith|sachith7/s6e10-stack-oof-predictions
koumeimaki|koumeimaki/s6e10-route-tfm-nn-oof
gp_tabpfn|goodpjw2008/s6e10-tabpfn-member
gp_extra|goodpjw2008/s6e10-goodpjw2008-extra-members-oof
dmitriy|dmitriykolmikov/s6e10-catboost-all-categorical-oof-and-test
kagan_abl|kagankoral/s6e10-ablation-members-oof
EOF

echo "=== notebook outputs ===" | tee -a "$LOG"
while IFS='|' read -r key handle; do
  [ -z "$key" ] && continue
  dir="pulled_members/$key"; mkdir -p "$dir"
  echo "--- notebook $key = $handle ---" | tee -a "$LOG"
  kaggle kernels output "$handle" -p "$dir" 2>&1 | tail -2 | tee -a "$LOG"
done <<'EOF'
yekenot|yekenot/ps-s6-e10-realmlp-pytabkit
hermengardo|hermengardo/single-catboost-eda-ps6e10
kratosyan|kratosyan/s6e10-xgboost-realmlp-route-id-teacher
kagan_enc|kagankoral/s6e10-encodings-catboost-realmlp-lb-0-961
aman_v5|amanatar/s6e10-the-original-prior-stack-v5
kozy|kozykappa/s6e10-source-transfer-foundation-model-oof
EOF

echo "=== ALL DONE ===" | tee -a "$LOG"
