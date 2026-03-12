"""
音频特征提取配置文件 (config.py)
支持 CPU/GPU 自动检测，与 main_extractor 及各特征模块一致。
"""

import torch


class AudioConfig:
    """音频分析配置类，供 main_extractor / basic_features / pitch_formant / harmonic_consonant / frequency_energy_ratio 使用。"""

    def __init__(self):
        # ---------- 设备 ----------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()

        # ---------- 音频基础参数 (与 librosa / STFT 一致) ----------
        self.sr = 48000  # 采样率 (Hz)，与 main 预处理一致
        self.frame_length = 960  # 帧长样本数，约 20ms @ 48kHz
        self.hop_length = 240   # 帧移样本数，约 5ms @ 48kHz
        self.n_fft = 2048       # FFT 点数

        # ---------- 基频 (Pitch) ----------
        self.f0_min = 60.0   # 最小基频 (Hz)
        self.f0_max = 800.0  # 最大基频 (Hz)

        # ---------- 共振峰 ----------
        self.n_formants = 5       # 提取共振峰数量
        self.max_formant_hz = 7000  # 最大共振峰频率 (Hz)

        # ---------- 动态范围 (basic_features) ----------
        self.dr_low_percentile = 10   # 动态范围低百分位
        self.dr_high_percentile = 90  # 动态范围高百分位

        # ---------- 可选/扩展参数 (当前脚本未使用，预留) ----------
        self.flux_threshold = 0.1    # 谱流量阈值
        self.energy_threshold = 0.01 # 辅音能量阈值
        self.zcr_threshold = 0.1     # 过零率阈值
        self.jitter_window = 5       # Jitter 窗口
        self.shimmer_window = 5      # Shimmer 窗口

        # ---------- 并行与批处理 ----------
        self.n_jobs = -1  # 并行任务数，-1 表示使用所有 CPU 核心

        # ---------- 路径 (main_extractor 使用) ----------
        self.input_folder = ""   # 输入音频目录，留空则 main 走测试音频逻辑
        self.output_csv = "audio_features.csv"  # 输出 CSV 文件名
        self.file_pattern = "*.wav"  # 输入目录下的文件匹配模式

    def __repr__(self):
        return f"AudioConfig(device={self.device}, sr={self.sr}, GPU={self.use_gpu})"

    def to_dict(self):
        """转换为字典，便于打印或持久化"""
        return {
            "device": str(self.device),
            "use_gpu": self.use_gpu,
            "sr": self.sr,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "n_formants": self.n_formants,
            "max_formant_hz": self.max_formant_hz,
            "dr_low_percentile": self.dr_low_percentile,
            "dr_high_percentile": self.dr_high_percentile,
            "n_jobs": self.n_jobs,
            "input_folder": self.input_folder,
            "output_csv": self.output_csv,
            "file_pattern": self.file_pattern,
        }


# 全局配置实例，各模块统一 from config import CONFIG
CONFIG = AudioConfig()

if __name__ == "__main__":
    print(CONFIG)
    print("\n配置详情:")
    for k, v in CONFIG.to_dict().items():
        print(f"  {k}: {v}")
