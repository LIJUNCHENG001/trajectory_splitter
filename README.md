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
