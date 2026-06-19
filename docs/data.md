**Series**
series/ 是多机场基础观测序列数据。

- 文件组成：train.npy、val.npy、test.npy、metadata.json
- 数组形状：[time, airport, variable]
- 机场顺序默认是：["ZGSZ", "ZGGG", "VHHH", "VMMC"]
- 变量顺序默认是：
  ["sknt", "wind_dir_sin", "wind_dir_cos", "tmpf", "alti", "dwpf", "relh", "gust", "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_norm"]
- 其中前 8 个是每个机场各自的观测变量，后 5 个是共享时间编码
- 时间分辨率：当前这版基础构建脚本默认是 1h
- 默认时间划分：
  - 训练集：2020-2023
  - 验证集：2024
  - 测试集：2025
- 默认完整年份下的 shape 一般是：
  - train.npy: [35064, 4, 13]
  - val.npy: [8784, 4, 13]
  - test.npy: [8760, 4, 13]
- 单位说明：
  - sknt、gust 最终都统一保存为 m/s
  - wind_dir_sin、wind_dir_cos 是无量纲
  - relh 是 %
  - tmpf、dwpf、alti 保持项目当前原始观测单位



**Series_15min**

series_15min/ 是在 series/ 基础上做的 15 分钟插值版本，仍然是多机场观测序列。

- 文件组成：train.npy、val.npy、test.npy、metadata.json

- 额外文件：train_original_mask.npy、val_original_mask.npy、test_original_mask.npy

- 数组形状仍然是：[time, airport, variable]

- 机场顺序、变量顺序与 series/ 完全一致

- 时间分辨率：15min

- 默认完整年份下的 shape 一般是：

  - train.npy: [140256, 4, 13]
  - val.npy: [35136, 4, 13]
  - test.npy: [35040, 4, 13]

- 插值规则

  原始整点严格保留

  - sknt：PCHIP
  - wind_dir_sin/cos：分别 PCHIP 后单位圆归一化
  - tmpf：PCHIP
  - dwpf：通过插值 tmpf-dwpf 重建
  - relh：优先由 tmpf/dwpf 重算，失败时回退插值
  - alti：PCHIP
  - gust：线性插值
  - 时间特征：按 15 分钟时间戳重生成

- *_original_mask.npy

  含义：

  - shape 是 [time]
  - True 表示这个时间点是原始真实观测点
  - False 表示这个时间点是插值生成点

**Series_15min_cubic**

series_15min_cubic/ 是另一套 15 分钟观测序列目录，用于保存新的 cubic
生成版本。

- 文件组成：train.npy、val.npy、test.npy、metadata.json

- 额外文件：train_original_mask.npy、val_original_mask.npy、test_original_mask.npy

- 数组形状仍然是：[time, airport, variable]

- 机场顺序、变量顺序与 series/、series_15min/ 一致

- 时间分辨率：15min

- 与 series_15min/ 共享同一份输入输出契约、mask 语义和窗口构造规则

- 与 series_15min/ 的区别仅在于它来自另一套 cubic 生成流程，因此目录名和
  metadata.source 必须显式写为 series_15min_cubic

- *_original_mask.npy

  含义：

  - shape 是 [time]
  - True 表示这个时间点是原始真实观测点
  - False 表示这个时间点是 cubic 生成点

**EC**
EC/ 是 ERA5 网格数据目录。

- 文件组成：train.npy、val.npy、test.npy

- EC 数据本体  

  - 形状是 [time, channel, lat, lon]，通道顺序是 ["t2m", "d2m", "sp", "u10", "v10"]。
  - 含义分别是：
    - t2m：2 米气温
    - d2m：2 米露点温度
    - sp：地面气压
    - u10：10 米 U 风
    - v10：10 米 V 风
  - 空间网格：项目当前默认按 9 x 9 网格处理
  - 时间分辨率：原始按小时使用
  - 网格范围：
    - lat_start=24.00, lat_end=22.00
    - lon_start=112.50, lon_end=114.50

  所以 EC 网格坐标轴是：

  - 纬度 latitudes = [24.00, 23.75, 23.50, 23.25, 23.00, 22.75, 22.50, 22.25, 22.00]
  - 经度 longitudes = [112.50, 112.75, 113.00, 113.25, 113.50, 113.75, 114.00, 114.25, 114.50]

  两个机场的确切经纬度  

  - 深圳宝安机场 ZGSZ: 22.55, 114.10
  - 广州白云机场 ZGGG: 23.39, 113.30
    
