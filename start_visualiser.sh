#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${STATE_VIS_DATASET:-/home/geek/share3/agilex_make_breakfast_380}"
output="${STATE_VIS_OUTPUT:-$project_dir/output}"
workspace="${STATE_VIS_WORKSPACE:-}"
state_visualiser="${STATE_VIS_APP_DIR:-/home/geek/share3/demo-7.29/projects/state_visualiser}"
port="${PORT:-8000}"

usage() {
  cat <<EOF
用法：
  ./start_visualiser.sh --dataset DATASET --output PIPELINE_OUTPUT [选项]

选项：
  --workspace PATH        指定可视化工作区，默认 OUTPUT/visualiser_workspace
  --state-visualiser PATH State Visualiser 项目目录
  --port PORT             服务端口，默认 8000
  -h, --help              显示帮助
EOF
}

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
    --workspace)
      [[ $# -ge 2 ]] || { echo "错误：--workspace 缺少路径" >&2; exit 2; }
      workspace="$2"
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

[[ -f "$dataset/meta/info.json" ]] || {
  echo "错误：$dataset 不是有效的 LeRobot 数据集" >&2
  exit 2
}
[[ -x "$state_visualiser/start.sh" ]] || {
  echo "错误：找不到 State Visualiser 启动脚本：$state_visualiser/start.sh" >&2
  exit 2
}

dataset="$(realpath "$dataset")"
output="$(realpath -m "$output")"
workspace="${workspace:-$output/visualiser_workspace}"
workspace="$(realpath -m "$workspace")"
segments="$output/visualiser_segments.json"
dataset_name="$(basename "$dataset")"
state_dir="$workspace/.state_visualiser/$dataset_name"

if [[ "$workspace" == "$dataset" || "$workspace" == "$dataset/"* ]]; then
  echo "错误：可视化工作区不能位于原始数据目录内：$workspace" >&2
  exit 2
fi
if [[ ! -f "$state_dir/segments.json" || ! -f "$state_dir/config.json" ]]; then
  [[ -f "$segments" ]] || { echo "错误：找不到 $segments" >&2; exit 2; }
  "$project_dir/prepare_visualiser_workspace.py" \
    --dataset "$dataset" \
    --segments "$segments" \
    --workspace "$workspace"
fi

if [[ -z "${STATE_VIS_ALLOWED_DATA_ROOT:-}" ]]; then
  allowed_root="$dataset"
  while [[ "$workspace" != "$allowed_root" && "$workspace" != "$allowed_root/"* ]]; do
    parent="$(dirname "$allowed_root")"
    [[ "$parent" != "$allowed_root" ]] || break
    allowed_root="$parent"
  done
  export STATE_VIS_ALLOWED_DATA_ROOT="$allowed_root"
fi

export STATE_VIS_DATASET="$dataset"
export STATE_VIS_ANNOTATED_DATASET="$workspace/processed_$dataset_name"
export STATE_VIS_ANNOTATIONS="$state_dir/segments.json"
export PORT="$port"

echo "State Visualiser"
echo "  数据集：$STATE_VIS_DATASET"
echo "  工作区：$workspace"
echo "  标注：$STATE_VIS_ANNOTATIONS"
echo "  地址：http://localhost:$PORT"
exec "$state_visualiser/start.sh"
