# Trajectory Splitter

该项目根据四个动作事件把 LeRobot v2.1 轨迹切成五段，并自动生成训练使用的
split JSON、切点分布图以及 State Visualiser 工作区。所有代码都位于本目录，
原始数据只读，输出统一写入运行时指定的目录。

## 切点规则

1. 右臂全部 motor 中最早出现的第一次突变；
2. right_gripper_percent 第三次闭合；
3. 第二个切点之后，左臂全部 motor 中最早出现的第一次突变；
4. 从末尾向前扫描，左臂第 2 个不同 motor 开始大幅变化。

信号先做 5 帧滑动平均。普通 motor 突变阈值为 0.05；第4点以各 motor
末尾8帧的中位数为基线，大幅变化阈值为 0.12。

切分前还会做采集连续性质检。对 `state[t] -> state[t+1]`，若任一关节的绝对
变化量至少为 0.3，且 12 维 state 变化量与同步 action 关节变化量的欧氏距离
之比至少为 5，则判定为录制跳接并剔除该 episode。使用单关节阈值可避免多个
关节的正常快速运动在 12 维欧氏距离中累加成误报；同步 action 比值可放行 action
同时变化的真实快速动作。剔除记录写入 `cut_points.csv` 的 `rejected` 行以及
`split_summary.json` 的 `rejected` 列表；质检剔除不会中断其余 episode 的
pipeline。

## 一条命令运行完整 pipeline

    cd /home/geek/share3/demo-7.29/projects/trajectory_splitter

    ./run_pipeline.sh \
      --dataset /path/to/lerobot_dataset \
      --output /path/to/pipeline_output

输入和输出均可自行指定。输出不能放进原始数据目录，避免修改原始数据。
输出目录已有分段 parquet 时，显式添加 --overwrite：

    ./run_pipeline.sh \
      --dataset /path/to/lerobot_dataset \
      --output /path/to/pipeline_output \
      --overwrite

快速测试前5条：

    ./run_pipeline.sh \
      --dataset /path/to/lerobot_dataset \
      --output /tmp/trajectory_splitter_test \
      --max-episodes 5

可以用 --python /path/to/python 指定环境；默认优先使用服务器上的 wam
环境。完整参数见：

    ./run_pipeline.sh --help

## 输出目录

    pipeline_output/
    ├── split/                         # 逐 episode 的训练 JSON
    ├── parquet/                       # episode_xxxxxx/segment_01..05.parquet
    ├── cut_points.csv
    ├── split_summary.json
    ├── visualiser_segments.json
    ├── cut_point_distributions/       # 4张PNG及CSV/JSON统计
    └── visualiser_workspace/          # State Visualiser 工作区

## 人工修改 cut_times 后重新同步

如果人工编辑了 pipeline_output/split_summary.json 中的 cut_times，不要重新
执行自动检测；使用同一个 pipeline 的 --reuse-summary：

    ./run_pipeline.sh \
      --dataset /path/to/lerobot_dataset \
      --output /path/to/pipeline_output \
      --reuse-summary

它会按真实 timestamp 找最近帧，同步 cut_frames、split JSON、相关 parquet、
分布图和可视化工作区，并在 manual_sync_backups/ 中备份修改前文件。

## State Visualiser

完整 pipeline 会自动创建适配工作区。启动方式：

    ./start_visualiser.sh \
      --dataset /path/to/lerobot_dataset \
      --output /path/to/pipeline_output \
      --port 8000

也可以在 pipeline 完成后立即启动：

    ./run_pipeline.sh \
      --dataset /path/to/lerobot_dataset \
      --output /path/to/pipeline_output \
      --start-visualiser

浏览器访问 http://localhost:8000。启动脚本会把数据集、工作区、标注文件和
允许访问的公共根目录传给
/home/geek/share3/demo-7.29/projects/state_visualiser，因此不需要修改
State Visualiser 的业务代码。

## 单独运行组件

统一 pipeline 已覆盖常用流程；需要调试时，各 Python 脚本仍可独立使用：

    /root/miniconda3/envs/wam/bin/python split_trajectories.py --help
    /root/miniconda3/envs/wam/bin/python sync_manual_cut_times.py --help
    /root/miniconda3/envs/wam/bin/python plot_cut_distributions.py --help
    /root/miniconda3/envs/wam/bin/python prepare_visualiser_workspace.py --help

## 生成 subtask 训练数据集

`vla_data_process/` 会读取规则切分生成的逐 episode JSON，把每个有效 subtask
物理裁成独立的 LeRobot v2.1 episode，同时裁剪三路视频并重建元数据。详细命令见
`vla_data_process/README.md`。

## 去除开头静止段

先只检测切点并检查 CSV：

    python trim_initial_stationary.py \
      --source /path/to/lerobot_dataset \
      --detect-only \
      --report /path/to/trim_points.csv

确认后直接自动检测并导出，或者编辑 CSV 中的 `trim_frame` 后通过 `--cuts` 使用
人工切点：

    python trim_initial_stationary.py \
      --source /path/to/lerobot_dataset \
      --output /path/to/trimmed_dataset \
      --cuts /path/to/trim_points.csv \
      --workers 2

导出会同步裁剪 Parquet 和全部视频，重置帧索引与时间戳，并重建 LeRobot v2.1
元数据。自动切点只依据 observation 中的实际关节和夹爪运动，不使用可能提前
变化的 action。原始数据始终只读，输出目录必须不存在且不能位于原始数据目录内。

## 完整早餐数据清洗流程

一条命令先去掉开头静止帧，再筛除插面包过程中有长时间内部停顿的 episode：

    /root/miniconda3/envs/lingbotvla-v2/bin/python clean_breakfast_dataset.py \
      --source /path/to/source_dataset \
      --trimmed-output /path/to/trimmed_dataset \
      --output /path/to/cleaned_dataset \
      --workers 2

异常规则使用右臂末端 XYZ（7 帧平滑）。松夹前最高 Z 点到松夹之间，末端速度
低于 10.5 mm/s 且连续至少 1 秒的内部停顿会使整个 episode 被排除。延续至松夹
的尾部停顿不计，episode 前 0.2 秒内开始的停顿也不计。清洗后的 episode 会重新
连续编号，Parquet、视频和元数据保持一致；被排除记录位于
`meta/removed_hesitation_episodes.csv`，旧编号可通过 `meta/source_mapping.jsonl`
追溯。原始数据不会被修改；cleaned 数据集成功导出并校验后，中间的
`trimmed-output` 会被自动删除。若异常筛除失败，则保留中间数据方便重跑。

只检测、不导出数据集：

    /root/miniconda3/envs/lingbotvla-v2/bin/python remove_hesitation_episodes.py \
      --source /path/to/trimmed_dataset \
      --detect-only \
      --report /path/to/hesitation_candidates.csv
