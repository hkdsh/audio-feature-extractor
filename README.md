# 音频特征提取脚本

基于 Python 的批量音频特征提取工具，支持 RMS、动态范围、基频、共振峰、谐波/辅音、七段频率能量比等多类特征，适用于语音/歌声分析及毕业论文实验等场景。

## 功能概览

- **预处理**：50Hz 高通滤波、EBU R128 响度归一化（-23 LUFS）、静音切除  
- **基础特征**：RMS、动态范围、谱流量、频谱质心/带宽/滚降、SEC、F95、浊音比、起振/衰减时间等  
- **基频与共振峰**：F0 统计、Jitter/Shimmer、共振峰 F1–F4、Singers Formant 等  
- **谐波与辅音**：H1-H2、H1-A3、HNR、HFC、辅音检测  
- **频率能量比**：七段频带（60–150Hz、150–350Hz、…、6000–10000Hz）能量比  
- **批处理**：支持多核并行（`joblib`），从目录批量读取 WAV 并输出 CSV  

## 环境要求

- Python 3.9（推荐）
- 见 `requirements.txt` 中的依赖

## 安装

```bash
# 建议使用 conda 创建独立环境
conda create -n audio_feature python=3.9
conda activate audio_feature

# 安装依赖
pip install -r requirements.txt
```

主要依赖：`librosa`、`numpy`、`pandas`、`scipy`、`torch`、`joblib`、`tqdm`、`pyloudnorm`、`praat-parselmouth`。

## 配置

在 **`config.py`** 中修改运行参数与路径：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `input_folder` | 待分析音频所在目录（留空则运行测试音频逻辑） | `""` |
| `output_csv` | 输出特征 CSV 文件名 | `"audio_features.csv"` |
| `file_pattern` | 输入目录下的文件匹配模式 | `"*.wav"` |
| `sr` | 采样率 (Hz) | `48000` |
| `n_jobs` | 并行任务数（-1 表示使用所有 CPU 核心） | `-1` |

其他参数（帧长、基频范围、共振峰数量等）也在 `config.py` 中，可按需调整。

## 使用方法

1. **设置输入目录**  
   在 `config.py` 中设置 `input_folder` 为你的 WAV 所在目录，例如：  
   `self.input_folder = r"D:\audio\wavs"`

2. **运行主程序**  
   ```bash
   python main_extractor.py
   ```

3. **查看结果**  
   程序会在当前目录生成 `config.py` 中指定的 CSV（默认 `audio_features.csv`），每行一个音频文件，列为各特征名。

若未设置 `input_folder` 或目录不存在，程序会提示并在测试音频上跑一遍特征提取（可用于检查环境与特征列表）。

## 项目结构

```
├── README.md
├── requirements.txt
├── config.py              # 统一配置（路径、采样率、并行等）
├── main_extractor.py      # 主入口：预处理 + 批量特征提取 + CSV 输出
├── basic_features.py      # 基础特征（RMS、动态范围、频谱等）
├── pitch_formant_features.py   # 基频、共振峰、Jitter/Shimmer
├── harmonic_consonant_features.py  # 谐波、辅音
└── frequency_energy_ratio_features.py  # 七段频率能量比
```

## 输出说明

- 输出为 CSV，第一列为 `filename`（音频文件名），其余列为数值特征。  
- 若某文件处理失败，会多一列 `error` 记录错误信息，并在控制台给出失败文件列表。  
- 特征名与上述功能模块对应（如 `RMS_mean`、`F0_center_frequency`、`Energy_Ratio_60_150Hz_mean` 等）。

## 许可与用途

本仓库为毕业论文实验用音频分析脚本，按需修改 `config.py` 与各特征模块即可适配不同数据集与实验设计。
