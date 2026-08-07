#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_python="/root/miniconda3/envs/wam/bin/python"
default_visualiser="/home/geek/share3/demo-7.29/projects/state_visualiser"

usage() {
  cat <<EOF
用法：
  ./run_pipeline.sh --dataset DATASET --output OUTPUT [选项]

必选参数：
  --dataset PATH          原始 LeRobot v2.1 数据集目录（只读）
  --output PATH           pipeline 的全部输出目录，不能位于原始数据目录内

选项：
  --overwrite             覆盖已有的分段 parquet
  --reuse-summary         不重新自动检测；使用 OUTPUT/split_summary.json 中人工修改的 cut_times
  --max-episodes N        仅处理前 N 条，用于快速测试
  --python PATH           Python 解释器（默认优先使用 wam 环境）
  --state-visualiser PATH State Visualiser 项目目录
  --start-visualiser      pipeline 完成后立即启动可视化服务
  --port PORT             可视化端口，默认 8000
  -h, --help              显示帮助

输出结构：
  OUTPUT/split/                       训练使用的逐 episode JSON
  OUTPUT/parquet/                     五段 parquet
  OUTPUT/cut_points.csv
  OUTPUT/split_summary.json
  OUTPUT/cut_point_distributions/     四张分布图与统计
  OUTPUT/visualiser_workspace/        State Visualiser 工作区
EOF
}

dataset=""
output=""
python_bin="${PYTHON_BIN:-}"
state_visualiser="${STATE_VIS_APP_DIR:-$default_visualiser}"
overwrite=0
reuse_summary=0
start_visualiser=0
max_episodes=""
port="${PORT:-8000}"

while (($#)); do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "错误：--dataset 缺少路径" >&2; exit 2; }
      dataset="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "错误：--output 缺少路径" >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "错误：--python 缺少路径" >&2; exit 2; }
      python_bin="$2"
      shift 2
      ;;
    --state-visualiser)
      [[ $# -ge 2 ]] || { echo "错误：--state-visualiser 缺少路径" >&2; exit 2; }
      state_visualiser="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "错误：--port 缺少数值" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --max-episodes)
      [[ $# -ge 2 ]] || { echo "错误：--max-episodes 缺少数值" >&2; exit 2; }
      max_episodes="$2"
      shift 2
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --reuse-summary)
      reuse_summary=1
      shift
      ;;
    --start-visualiser)
      start_visualiser=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$dataset" ]] || { echo "错误：必须指定 --dataset" >&2; usage >&2; exit 2; }
[[ -n "$output" ]] || { echo "错误：必须指定 --output" >&2; usage >&2; exit 2; }
[[ -f "$dataset/meta/info.json" ]] || {
  echo "错误：$dataset 不是有效的 LeRobot 数据集" >&2
  exit 2
}

dataset="$(realpath "$dataset")"
output="$(realpath -m "$output")"
if [[ "$output" == "$dataset" || "$output" == "$dataset/"* ]]; then
  echo "错误：输出目录不能是原始数据目录或其子目录：$output" >&2
  exit 2
fi
mkdir -p "$output"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-trajectory-splitter}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "$python_bin" ]]; then
  if [[ -x "$default_python" ]]; then
    python_bin="$default_python"
  else
    python_bin="$(command -v python3)"
  fi
fi
[[ -x "$python_bin" ]] || { echo "错误：Python 不可执行：$python_bin" >&2; exit 2; }
"$python_bin" -c "import matplotlib, numpy, pandas, pyarrow" || {
  echo "错误：Python 环境缺少依赖，请安装 $project_dir/requirements.txt" >&2
  exit 2
}

if ((reuse_summary == 0)); then
  echo "[1/4] 自动检测切点并生成 parquet 分段"
  split_args=(
    "$project_dir/split_trajectories.py"
    --dataset "$dataset"
    --output "$output"
  )
  ((overwrite)) && split_args+=(--overwrite)
  [[ -n "$max_episodes" ]] && split_args+=(--max-episodes "$max_episodes")
  "$python_bin" "${split_args[@]}"
else
  echo "[1/4] 使用已有 split_summary.json（保留人工 cut_times）"
  [[ -f "$output/split_summary.json" ]] || {
    echo "错误：找不到 $output/split_summary.json" >&2
    exit 2
  }
fi

echo "[2/4] 生成 split JSON，并同步人工 cut_times 对应的 frame"
"$python_bin" "$project_dir/sync_manual_cut_times.py" \
  --dataset "$dataset" \
  --summary "$output/split_summary.json" \
  --cut-points "$output/cut_points.csv" \
  --visualiser-segments "$output/visualiser_segments.json" \
  --split-output "$output/split" \
  --parquet-output "$output/parquet"

echo "[3/4] 生成四个切点的分布图和统计"
"$python_bin" "$project_dir/plot_cut_distributions.py" \
    --input "$output/cut_points.csv" \
    --output "$output/cut_point_distributions"

echo "[4/4] 生成 State Visualiser 工作区"
"$python_bin" "$project_dir/prepare_visualiser_workspace.py" \
  --dataset "$dataset" \
  --segments "$output/visualiser_segments.json" \
  --workspace "$output/visualiser_workspace"

echo
echo "Pipeline 完成"
echo "  原始数据：$dataset"
echo "  split JSON：$output/split"
echo "  分布图：$output/cut_point_distributions"
echo "  可视化工作区：$output/visualiser_workspace"

visualiser_command=(
  "$project_dir/start_visualiser.sh"
  --dataset "$dataset"
  --output "$output"
  --state-visualiser "$state_visualiser"
  --port "$port"
)
if ((start_visualiser)); then
  exec "${visualiser_command[@]}"
fi

printf "启动可视化："
printf " %q" "${visualiser_command[@]}"
printf "\n"
