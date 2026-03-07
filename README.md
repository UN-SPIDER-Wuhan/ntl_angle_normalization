# NTL Prophet

基于 VIIRS 夜光遥感数据的时序处理项目，包含数据预处理、观测角度归一化、Prophet 拟合补全，以及将时序文本还原为逐日 GeoTIFF 影像的完整流程。

本项目适合以下场景：

- 处理 VNP46A1 / VNP46A2 夜光产品
- 对像元级夜光时间序列进行质量控制与角度归一化
- 使用 Prophet 对缺测或异常波动进行时序拟合与补全
- 将最终 txt 格式结果恢复为可在 GIS 软件中读取的 GeoTIFF

## 1. 项目功能

项目当前包含四类核心能力：

### 1. 数据预处理

- 从 VNP46A2 / VNP46A1 的 HDF5 文件中提取关键图层
- 支持研究区 Shapefile 裁剪
- 支持跨瓦片自动镶嵌
- 生成标准化像元时序文本文件

对应模块与入口：

- notebook: [1_preprocessing_data.ipynb](1_preprocessing_data.ipynb)
- code: [functions/preprocessing.py](functions/preprocessing.py)

### 2. 角度归一化

- 读取预处理后的像元时序 txt
- 基于传感器天顶角执行归一化拟合
- 支持异常值过滤、并行处理、结果可视化

对应模块与入口：

- notebook: [2_angle normalization.ipynb](2_angle%20normalization.ipynb)
- code: [functions/angle_normalization.py](functions/angle_normalization.py)
- code: [functions/timeseries_analysis.py](functions/timeseries_analysis.py)

### 3. Prophet 拟合与补全

- 对归一化后的时间序列进行 Prophet 拟合
- 支持 PSO 搜索 Prophet 关键参数
- 输出拟合后时序文本和精度报告

对应模块与入口：

- notebook: [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb)
- code: [functions/prophet_pipeline.py](functions/prophet_pipeline.py)

### 4. 文本转影像

- 将 Prophet 输出的 txt 结果转换为逐日 GeoTIFF
- 支持使用模板影像继承空间参考
- 也支持无模板模式，但需要手动提供宽高、仿射变换和坐标系

对应模块：

- code: [functions/text_to_img.py](functions/text_to_img.py)

## 2. 项目结构

```text
ntl_prophet/
├── README.md
├── LICENSE
├── requirements.txt
├── __init__.py
├── 1_preprocessing_data.ipynb
├── 2_angle normalization.ipynb
├── 3_prophet_params_and_run.ipynb
├── datasets/
└── functions/
    ├── __init__.py
    ├── preprocessing.py
    ├── angle_normalization.py
    ├── prophet_pipeline.py
    ├── text_to_img.py
    └── timeseries_analysis.py
```

## 3. 推荐环境

建议使用以下环境：

- Python 3.10 或 3.11
- Windows 10/11
- Conda 环境管理

## 4. 安装方式

### 方式一：使用 Conda 创建环境（推荐）

```bash
conda create -n ntl_prophet python=3.10 -y
conda activate ntl_prophet
pip install -r requirements.txt
```

如果安装 GDAL 或 Rasterio 失败，优先尝试：

```bash
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

### 方式二：直接使用 pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 5. 数据处理流程

推荐按照以下顺序运行：

### Step 1. 预处理原始 HDF5 数据

打开 [1_preprocessing_data.ipynb](1_preprocessing_data.ipynb)，配置以下路径：

- VNP46A2 数据目录
- VNP46A1 数据目录
- 输出目录
- Shapefile 路径
- 时间范围与瓦片过滤条件

该步骤会输出标准化时间序列文本，例如：

- processed 目录中的 ntl_timeseries.txt

### Step 2. 执行角度归一化

打开 [2_angle normalization.ipynb](2_angle%20normalization.ipynb)，配置：

- 输入 txt 所在目录
- 输出目录
- 数据集名称列表
- 并行进程数
- 3-sigma 异常值过滤参数

该步骤会输出：

- 归一化后的时序 txt
- 拟合参数
- R² 等统计结果
- 可选可视化图表

### Step 3. 执行 Prophet 拟合

打开 [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb)，配置：

- 输入 txt 路径
- 输出 txt 路径
- DOY 范围
- PSO 参数
- Prophet 参数
- 并行参数

该步骤会输出：

- Prophet 拟合后的 txt
- 各类精度统计信息

### Step 4. 将 txt 转为逐日 GeoTIFF

在 [3_prophet_params_and_run.ipynb](3_prophet_params_and_run.ipynb) 的最后一个代码单元中，已经添加 txt 转影像调用。

输出结果为：

- 每一天一个 tif 文件
- 文件名格式为 YYYYMMDD.tif

## 6. 输入输出格式说明

### 6.1 时序 txt 格式

项目使用的像元时序文本通常为以下格式：

```text
point1:lng,lat(lefttop):YYYYMMDD,Zenith,NTLValue;YYYYMMDD,Zenith,NTLValue;...
point2:lng,lat(lefttop):YYYYMMDD,Zenith,NTLValue;...
```

其中：

- point1、point2 表示像元编号
- lng,lat(lefttop) 表示该像元左上角坐标
- 每条记录依次为 日期、天顶角、夜光值

### 6.2 GeoTIFF 输出

文本转影像时，程序按像元编号顺序回填到二维网格，并将每天的数据写出为单波段 GeoTIFF。

推荐使用模板影像，原因是：

- 可以直接继承宽高
- 可以直接继承 transform
- 可以直接继承 CRS
- 不容易出现空间错位

无模板模式也可以使用，但必须显式提供：

- width
- height
- transform
- crs

## 7. 核心代码接口

### 7.1 预处理

主要函数位于 [functions/preprocessing.py](functions/preprocessing.py)：

- complete_ntl_preprocessing_pipeline
- stage1_extract_and_pair
- stage2_generate_time_series
- clip_rasters_by_shapefile
- mosaic_tiles_by_date

### 7.2 角度归一化

主要函数位于 [functions/angle_normalization.py](functions/angle_normalization.py)：

- run_angle_normalization
- readFile
- visScatterAndFitCurve
- visTimeSeries

### 7.3 Prophet 拟合

主要函数位于 [functions/prophet_pipeline.py](functions/prophet_pipeline.py)：

- run_prophet_pipeline

### 7.4 文本转影像

主要函数位于 [functions/text_to_img.py](functions/text_to_img.py)：

- txt_to_daily_geotiffs

示例：

```python
from functions.text_to_img import txt_to_daily_geotiffs

tif_files = txt_to_daily_geotiffs(
    txt_path=r".\output\ntl_timeseries_angle_prophet_pso.txt",
    output_dir=r".\output\prophet_daily_tif",
    start_date="20180101",
    end_date="20190101",
    template_tif=r".\template.tif",
)
```

## 8. 依赖说明

本项目依赖大致分为四类：

- 数值计算与数据处理：numpy、pandas、scipy
- 可视化：matplotlib
- 时序建模：prophet
- 遥感与 GIS：GDAL、rasterio、affine

另有辅助依赖：

- tqdm：显示进度条
- notebook / ipykernel：运行 Jupyter Notebook

## 9. 使用建议

### 9.1 关于模板影像

如果你已经有研究区裁剪后的任意一张参考 tif，建议始终使用模板模式进行 txt 转影像。

只有在你明确知道以下信息时，才建议无模板模式：

- 栅格列数与行数
- 左上角坐标
- 像元分辨率
- 投影坐标系

### 9.2 关于并行

项目多个阶段支持并行处理，但并不是线程数越高越好。

建议：

- 小数据量：4 到 8 个 worker
- 中等数据量：8 到 16 个 worker
- Windows 下若出现资源占用或句柄问题，可适当降低 worker 数量

## 10. 引用与致谢

如果你后续公开论文或项目页面，建议补充数据来源、研究区说明、结果示例以及方法参考文献。

## 11. 仓库发布建议

当前项目目录名建议保持为 ntl_prophet，这样与包导入名、notebook 中的路径推断逻辑以及 GitHub 仓库展示都保持一致。

上传 GitHub 前建议确认：

- notebook 中的数据路径仍指向相对目录
- datasets 下的大体量原始数据不要直接提交
- output、processed 等结果目录按需保留或继续由 .gitignore 忽略
- 如果你计划公开发布，补充项目截图、研究区说明和数据来源会更完整

