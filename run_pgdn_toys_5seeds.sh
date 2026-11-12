#!/usr/bin/env bash
set -euo pipefail

# PGDN on Movies to Toys with seeds 2020-2024.
# T=128 follows the seed2020 sampled test NDCG@10 sweep.
# All run artifacts are saved in one runs/ directory and can be resumed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
DIFFUSION_STEPS=128
RUN_NAME="${RUN_NAME:-tptd_final_toys_t${DIFFUSION_STEPS}_5seeds}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ &&
   "${RUN_NAME}" != . && "${RUN_NAME}" != .. ]] || {
  echo "RUN_NAME must contain only letters, digits, dots, underscores or hyphens" >&2
  exit 1
}
RUN_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"
SRC_DIR="${PROJECT_DIR}/src"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data01/24-zhangwei/miniconda/envs/py38/bin/python}"
TARGETS=(toys)
SEEDS=(2020 2021 2022 2023 2024)
declare -A GPU_IDS=([video]="${VIDEO_GPU_ID:-1}" [toys]="${TOYS_GPU_ID:-2}")
declare -A LOSS_WEIGHTS=([video]=0.5 [toys]=0.01)
declare -A ITEM_NUMS=([movie]=98507 [video]=23978 [toys]=113989)
TARGET_AUX_WEIGHT=0.75
TARGET_AUX_PREFIXES=1
TARGET_AUX_WARMUP_EPOCHS=5
TARGET_AUX_NOISE=gaussian
EXPECTED_RUNS=$((${#TARGETS[@]} * ${#SEEDS[@]}))
STATUS_FILE="${RUN_DIR}/status.tsv"
HASH_FILE="${RUN_DIR}/code.sha256"
DATA_HASH_FILE="${RUN_DIR}/data.sha256"
SUMMARY_SCRIPT="${PROJECT_DIR}/summarize_tptd_final_5seeds.py"
RUN_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

for target in "${TARGETS[@]}"; do
  [[ "${GPU_IDS[${target}]}" =~ ^[0-9]+$ ]] || {
    echo "GPU ID for ${target} must be a nonnegative integer" >&2; exit 1
  }
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Output directory: %s\nPlanned runs: %s\nDiffusion steps: %s\n' \
    "${RUN_DIR}" "${EXPECTED_RUNS}" "${DIFFUSION_STEPS}"
  printf 'target\tgpu\tprototype_loss_weight\ttarget_aux_weight\tseed\n'
  for target in "${TARGETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\n' "${target}" "${GPU_IDS[${target}]}" \
        "${LOSS_WEIGHTS[${target}]}" "${TARGET_AUX_WEIGHT}" "${seed}"
    done
  done
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -r "${SUMMARY_SCRIPT}" ]] || { echo "Missing summary script: ${SUMMARY_SCRIPT}" >&2; exit 1; }
for file in main.py model.py diffurec.py trainer.py source_guided_prototypes.py target_transition.py step_sample.py; do
  [[ -r "${SRC_DIR}/${file}" ]] || { echo "Missing source: ${SRC_DIR}/${file}" >&2; exit 1; }
done
for dataset in movie "${TARGETS[@]}"; do
  for split in train val test; do
    [[ -r "${PROJECT_DIR}/data/${dataset}/${split}.df" ]] || {
      echo "Missing data: ${PROJECT_DIR}/data/${dataset}/${split}.df" >&2
      exit 1
    }
  done
done

if [[ -e "${RUN_DIR}/run_config.txt" || -e "${RUN_DIR}/plan.tsv" ]]; then
  [[ "${RESUME:-0}" == 1 ]] || {
    echo "Run already initialized. Use RESUME=1 to continue: ${RUN_DIR}" >&2
    exit 1
  }
  [[ -r "${HASH_FILE}" && -r "${DATA_HASH_FILE}" && -r "${RUN_DIR}/plan.tsv" ]] || {
    echo "Cannot resume: missing code hash, data hash, or plan file." >&2; exit 1
  }
  for target in "${TARGETS[@]}"; do
    [[ -r "${RUN_DIR}/status_${target}.tsv" ]] || {
      echo "Cannot resume: missing status_${target}.tsv." >&2; exit 1
    }
    grep -Fxq "gpu_${target}=${GPU_IDS[${target}]}" "${RUN_DIR}/run_config.txt" || {
      echo "Cannot resume with a changed GPU for ${target}." >&2; exit 1
    }
  done
  grep -Fxq "python_bin=${PYTHON_BIN}" "${RUN_DIR}/run_config.txt" || {
    echo "Cannot resume with a changed Python environment." >&2; exit 1
  }
  sha256sum --check --status "${HASH_FILE}" || {
    echo "Code or script changed since this run began; refusing to mix versions." >&2
    exit 1
  }
  sha256sum --check --status "${DATA_HASH_FILE}" || {
    echo "Data changed since this run began; refusing to mix versions." >&2
    exit 1
  }
else
  [[ "${RESUME:-0}" != 1 ]] || { echo "No existing run to resume." >&2; exit 1; }
  mkdir -p "${RUN_DIR}/main_logs" "${RUN_DIR}/wrapper_logs" \
    "${RUN_DIR}/completed" "${RUN_DIR}/saved_model"
  sha256sum "${RUN_SCRIPT}" "${SUMMARY_SCRIPT}" \
    "${SRC_DIR}/main.py" "${SRC_DIR}/model.py" "${SRC_DIR}/diffurec.py" \
    "${SRC_DIR}/trainer.py" "${SRC_DIR}/source_guided_prototypes.py" \
    "${SRC_DIR}/target_transition.py" "${SRC_DIR}/step_sample.py" > "${HASH_FILE}"
  for dataset in movie "${TARGETS[@]}"; do
    for split in train val test; do
      sha256sum "${PROJECT_DIR}/data/${dataset}/${split}.df"
    done
  done > "${DATA_HASH_FILE}"
  {
    printf 'experiment=tptd_final_toys_5seeds\n'
    printf 'project_dir=%s\noutput_dir=%s\npython_bin=%s\n' \
      "${PROJECT_DIR}" "${RUN_DIR}" "${PYTHON_BIN}"
    printf 'source=movie\ntargets=%s\nseeds=%s\nexpected_runs=%s\n' \
      "${TARGETS[*]}" "${SEEDS[*]}" "${EXPECTED_RUNS}"
    for target in "${TARGETS[@]}"; do
      printf 'gpu_%s=%s\nprototype_loss_weight_%s=%s\n' \
        "${target}" "${GPU_IDS[${target}]}" "${target}" "${LOSS_WEIGHTS[${target}]}"
    done
    printf 'target_aux_weight=%s\ntarget_aux_prefixes=%s\n' \
      "${TARGET_AUX_WEIGHT}" "${TARGET_AUX_PREFIXES}"
    printf 'target_aux_warmup_epochs=%s\ntarget_aux_noise=%s\n' \
      "${TARGET_AUX_WARMUP_EPOCHS}" "${TARGET_AUX_NOISE}"
    printf 'target_aux_weight_selection=seed2020_sampled_validation_sweep\n'
    printf 'prototype_build_mode=source_guided\nprototype_target_fraction=0.2\n'
    printf 'prototype_source_cluster_iters=5\nprototype_target_correction=0\n'
    printf 'prototype_correction_mode=none\n'
    printf 'prototype_confidence_quantile=0.5\nprototype_source_similarity_quantile=0.1\n'
    printf 'prototype_min_source_support=32\nprototype_min_target_support=8\n'
    printf 'num_prototypes=64\nprototype_topk=5\nprototype_noise_strength=0.75\n'
    printf 'prototype_noise_warmup_epochs=5\nprototype_temperature=0.1\n'
    printf 'prototype_retrieval_temperature=0.1\nprototype_momentum=0.99\n'
    printf 'prototype_public_weight=1.0\nprototype_min_public=0.05\n'
    printf 'evaluation_protocol=d2tcdr_legacy\nmodel_selection_metric=sampled_composite\n'
    printf 'p=0.3\nlr=0.001\nbatch_size=512\nepochs=100\n'
    printf 'source_patience=2\nsource_eval_interval=5\n'
    printf 'target_patience=5\ntarget_eval_interval=1\n'
    printf 'dropout=0.1\nemb_dropout=0.3\nhidden_size=128\n'
    printf 'diffusion_steps=%s\nmax_len=8\nw=1.0\n' "${DIFFUSION_STEPS}"
    printf 'diffusion_steps_selection=seed2020_sampled_test_ndcg10_sweep\n'
    printf 'checkpoint_root=%s/saved_model\n' "${RUN_DIR}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    "${PYTHON_BIN}" -c 'import torch; print("torch=" + torch.__version__)'
  } > "${RUN_DIR}/run_config.txt"
  for target in "${TARGETS[@]}"; do
    printf 'run_id\ttarget\tseed\tprototype_loss_weight\ttarget_aux_weight\tgpu\tattempt\tstatus\tpython_exit\ttee_exit\tstarted_at\tfinished_at\tmain_log_dir\twrapper_log\tcheckpoint\tbank_checkpoint\n' \
      > "${RUN_DIR}/status_${target}.tsv"
  done
  {
    printf 'target\tprototype_loss_weight\ttarget_aux_weight\tseed\tgpu\trun_id\n'
    for target in "${TARGETS[@]}"; do
      weight="${LOSS_WEIGHTS[${target}]}"
      label="plw${weight//./p}"
      for seed in "${SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\tmovie_%s_%s_tptd0p75_seed%s\n' \
          "${target}" "${weight}" "${TARGET_AUX_WEIGHT}" "${seed}" \
          "${GPU_IDS[${target}]}" "${target}" "${label}" "${seed}"
      done
    done
  } > "${RUN_DIR}/plan.tsv"
fi

for target in "${TARGETS[@]}"; do
  gpu_id="${GPU_IDS[${target}]}"
  if ! CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -c \
      'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' \
      > "${RUN_DIR}/gpu${gpu_id}_check.txt" 2>&1; then
    echo "GPU ${gpu_id} unavailable; see ${RUN_DIR}/gpu${gpu_id}_check.txt" >&2
    exit 1
  fi
done

run_target() {
  local target="$1" gpu_id="$2" weight label seed run_id attempt
  local main_log_dir wrapper_log checkpoint_name checkpoint bank_checkpoint
  local started_at finished_at python_exit tee_exit status
  local target_exit=0
  local -a pipeline_status
  local target_status_file="${RUN_DIR}/status_${target}.tsv"

  weight="${LOSS_WEIGHTS[${target}]}"
  label="plw${weight//./p}"
  cd "${RUN_DIR}"
  for seed in "${SEEDS[@]}"; do
    if ! sha256sum --check --status "${HASH_FILE}" || \
       ! sha256sum --check --status "${DATA_HASH_FILE}"; then
      echo "Code or data changed during the experiment; stop ${target} before seed ${seed}." >&2
      return 1
    fi
    run_id="movie_${target}_${label}_tptd0p75_seed${seed}"
    if [[ -f "${RUN_DIR}/completed/${run_id}.ok" ]]; then
      echo "SKIP completed ${run_id}"
      continue
    fi
    attempt="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
    main_log_dir="${RUN_DIR}/main_logs/${run_id}/${attempt}/"
    wrapper_log="${RUN_DIR}/wrapper_logs/${run_id}_${attempt}.log"
    checkpoint_name="${run_id}_${attempt}.pth"
    checkpoint="${RUN_DIR}/saved_model/movie_${target}/${checkpoint_name}"
    bank_checkpoint="${RUN_DIR}/saved_model/movie_${target}/${run_id}_${attempt}_bank.pth"
    mkdir -p "${main_log_dir}"
    started_at="$(date --iso-8601=seconds)"
    echo "[${started_at}] START ${run_id}; GPU=${gpu_id}; output=${RUN_DIR}"

    set +e
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${PYTHON_BIN}" -u "${SRC_DIR}/main.py" \
        --device cuda \
        --s_dataset movie --t_dataset "${target}" \
        --source_item_num "${ITEM_NUMS[movie]}" \
        --target_item_num "${ITEM_NUMS[${target}]}" \
        --data_root "${PROJECT_DIR}/data" \
        --evaluation_protocol d2tcdr_legacy \
        --random_seed "${seed}" \
        --p 0.3 --lr 0.001 --batch_size 512 \
        --epochs 100 --patience 2 --eval_interval 5 \
        --dropout 0.1 --emb_dropout 0.3 --hidden_size 128 \
        --diffusion_steps "${DIFFUSION_STEPS}" --max_len 8 --w 1.0 \
        --model_selection_metric sampled_composite \
        --target_eval_interval 1 --target_patience 5 \
        --num_prototypes 64 --prototype_topk 5 \
        --prototype_loss_weight "${weight}" \
        --prototype_noise_strength 0.75 \
        --prototype_noise_warmup_epochs 5 \
        --prototype_temperature 0.1 \
        --prototype_retrieval_temperature 0.1 \
        --prototype_momentum 0.99 \
        --prototype_public_weight 1.0 \
        --prototype_min_public 0.05 \
        --prototype_build_mode source_guided \
        --prototype_target_fraction 0.2 \
        --prototype_source_cluster_iters 5 \
        --prototype_target_correction 0 \
        --prototype_correction_mode none \
        --prototype_confidence_quantile 0.5 \
        --prototype_source_similarity_quantile 0.1 \
        --prototype_min_source_support 32 \
        --prototype_min_target_support 8 \
        --target_aux_weight "${TARGET_AUX_WEIGHT}" \
        --target_aux_prefixes "${TARGET_AUX_PREFIXES}" \
        --target_aux_warmup_epochs "${TARGET_AUX_WARMUP_EPOCHS}" \
        --target_aux_noise "${TARGET_AUX_NOISE}" \
        --checkpoint_name "${checkpoint_name}" \
        --log_file "${main_log_dir}" \
        2>&1 | tee "${wrapper_log}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    python_exit="${pipeline_status[0]}"
    tee_exit="${pipeline_status[1]}"
    status=ok
    if [[ "${python_exit}" -ne 0 || "${tee_exit}" -ne 0 || \
          ! -s "${checkpoint}" || ! -s "${bank_checkpoint}" ]] || \
       ! "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" verify "${main_log_dir}" "${target}"; then
      status=failed
      target_exit=1
    else
      printf '%s\n' "${attempt}" > "${RUN_DIR}/completed/${run_id}.ok"
    fi
    finished_at="$(date --iso-8601=seconds)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${run_id}" "${target}" "${seed}" "${weight}" "${TARGET_AUX_WEIGHT}" "${gpu_id}" "${attempt}" \
      "${status}" "${python_exit}" "${tee_exit}" "${started_at}" "${finished_at}" \
      "${main_log_dir}" "${wrapper_log}" "${checkpoint}" "${bank_checkpoint}" \
      >> "${target_status_file}"
    echo "[${finished_at}] ${status} ${run_id}"
  done
  return "${target_exit}"
}

worker_pids=()
terminate_workers() {
  trap - INT TERM
  for pid in "${worker_pids[@]}"; do
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap terminate_workers INT TERM

for target in "${TARGETS[@]}"; do
  run_target "${target}" "${GPU_IDS[${target}]}" &
  worker_pids+=("$!")
done

overall_exit=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    overall_exit=1
  fi
done
trap - INT TERM

head -n 1 "${RUN_DIR}/status_${TARGETS[0]}.tsv" > "${STATUS_FILE}"
for target in "${TARGETS[@]}"; do
  tail -n +2 "${RUN_DIR}/status_${target}.tsv" >> "${STATUS_FILE}"
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" summarize "${RUN_DIR}" || overall_exit=1
echo "Finished ${EXPECTED_RUNS} planned runs; results: ${RUN_DIR}"
exit "${overall_exit}"
