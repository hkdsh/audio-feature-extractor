import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import scipy.signal as signal
import pyloudnorm as pyln

warnings.filterwarnings('ignore')

from config import CONFIG
from basic_features import BasicFeatureExtractor
from pitch_formant_features import PitchFormantExtractor
from harmonic_consonant_features import HarmonicConsonantExtractor
from frequency_energy_ratio_features import FrequencyEnergyRatioExtractor  # 导入新模块


class AudioFeatureExtractor:
    """完整的音频特征提取器"""

    def __init__(self, config=CONFIG):
        self.config = config

        # 初始化各个子提取器
        self.basic_extractor = BasicFeatureExtractor(config)
        self.pitch_formant_extractor = PitchFormantExtractor(config)
        self.harmonic_consonant_extractor = HarmonicConsonantExtractor(config)
        self.frequency_energy_ratio_extractor = FrequencyEnergyRatioExtractor(config)  # 初始化新提取器

        print(f"特征提取器初始化完成")
        print(f"使用设备: {config.device}")
        print(f"采样率: {config.sr} Hz")



    def preprocess_audio(self, audio_path, target_sr=48000):
        try:
            # 1. 加载音频
            audio, sr = librosa.load(audio_path, sr=target_sr, dtype=np.float32, mono=True)

            # 2. 鲁棒化滤波 (50Hz 高通)
            b, a = signal.butter(4, 50, btype='highpass', fs=sr) 
            audio = signal.filtfilt(b, a, audio)
            
            # ====================================================
            # [新增] 3. EBU R128 响度归一化 (Target: -23 LUFS)
            # ====================================================
            try:
                # 创建测量计
                meter = pyln.Meter(sr) 
                # 测量当前响度
                loudness = meter.integrated_loudness(audio)
                
                # 归一化到 -23 LUFS (广播标准) 或 -14 LUFS (流媒体标准)
                # 建议干声分析使用 -23 LUFS，保留更多动态余量
                target_lufs = -23.0 
                audio = pyln.normalize.loudness(audio, loudness, target_lufs)
                
                # 防止归一化后爆音 (Clip protection)
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / np.max(np.abs(audio)) * 0.99
                    
            except Exception as e_norm:
                # 如果音频太短或全静音，测量可能会失败，降级为峰值归一化
                print(f"LUFS归一化失败，降级为峰值归一化: {e_norm}")
                audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9

            # ====================================================

            # 4. 静音去除 (保持原有逻辑)
            non_silent_intervals = librosa.effects.split(audio, top_db=60)
            
            # [修正] 建议不要直接 concatenate，而是返回 intervals 让后续处理，
            # 但如果你坚持要拼接，至少归一化要在拼接前做。
            if len(non_silent_intervals) > 0:
                audio_non_silent = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
            else:
                audio_non_silent = audio

            return audio_non_silent, sr

        except Exception as e:
            print(f"预处理音频时发生错误: {e}")
            return None, None

    def extract_features(self, audio_path, verbose=True):
        """
        从单个音频文件提取所有特征

        参数:
            audio_path: 音频文件路径
            verbose: 是否显示进度信息

        返回:
            features: 包含所有特征的字典
        """
        if verbose:
            print(f"\n处理文件: {audio_path}")

        try:
            # 预处理音频
            y, sr = self.preprocess_audio(audio_path)
            if y is None:
                raise ValueError("音频预处理失败")

            # 初始化特征字典
            features = {'filename': os.path.basename(audio_path)}

            # 1. 提取基础特征
            if verbose:
                print("  - 提取基础特征...")
            basic_features = self.basic_extractor.extract_all_basic_features(y)
            features.update(basic_features)

            # 2. 提取基频和共振峰特征
            if verbose:
                print("  - 提取基频和共振峰特征...")
            pitch_formant_features = self.pitch_formant_extractor.extract_all_pitch_formant_features(y, sr)
            features.update(pitch_formant_features)

            # 3. 提取谐波和辅音特征
            if verbose:
                print("  - 提取谐波和辅音特征...")
            harmonic_consonant_features = self.harmonic_consonant_extractor.extract_all_harmonic_consonant_features(y,
                                                                                                                    sr)
            features.update(harmonic_consonant_features)

            # 4. 提取七段频率能量比特征
            if verbose:
                print("  - 提取七段频率能量比特征...")
            frequency_energy_features = self.frequency_energy_ratio_extractor.extract_all_features(y, sr)
            features.update(frequency_energy_features)

            if verbose:
                print(f"  ✓ 成功提取 {len(features)} 个特征")

            return features

        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            return {'filename': os.path.basename(audio_path), 'error': str(e)}

    def extract_features_batch(self, audio_paths, n_jobs=None, verbose=True):
        """
        批量提取音频特征 (支持并行处理)

        参数:
            audio_paths: 音频文件路径列表
            n_jobs: 并行任务数 (None表示使用配置中的值)
            verbose: 是否显示进度

        返回:
            DataFrame: 包含所有特征的数据框
        """
        if n_jobs is None:
            n_jobs = self.config.n_jobs

        print(f"\n开始批量处理 {len(audio_paths)} 个音频文件")
        print(f"并行任务数: {n_jobs if n_jobs > 0 else '全部CPU核心'}")

        # 使用joblib并行处理
        if n_jobs == 1:
            # 单线程处理 (便于调试)
            results = []
            for audio_path in tqdm(audio_paths, desc="提取特征"):
                features = self.extract_features(audio_path, verbose=False)
                results.append(features)
        else:
            # 多线程并行处理
            results = Parallel(n_jobs=n_jobs)(  # 使用并行提取
                delayed(self.extract_features)(path, verbose=False)
                for path in tqdm(audio_paths, desc="提取特征")
            )

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 检查错误
        if 'error' in df.columns:
            error_count = df['error'].notna().sum()
            if error_count > 0:
                print(f"\n警告: {error_count} 个文件处理失败")
                print("失败的文件:")
                print(df[df['error'].notna()][['filename', 'error']])

        print(f"\n完成! 共提取 {len(df.columns) - 1} 个特征")

        return df

    def extract_from_directory(self, directory, pattern="*.wav", output_csv=None, n_jobs=None):
        """
        从目录中提取所有音频文件的特征

        参数:
            directory: 音频文件目录
            pattern: 文件匹配模式 (默认: *.wav)
            output_csv: 输出CSV文件路径 (None表示不保存)
            n_jobs: 并行任务数

        返回:
            DataFrame: 特征数据框
        """
        # 查找所有音频文件
        audio_dir = Path(directory)
        audio_paths = list(audio_dir.glob(pattern))

        if len(audio_paths) == 0:
            print(f"错误: 在 {directory} 中未找到匹配 {pattern} 的文件")
            return None

        print(f"找到 {len(audio_paths)} 个音频文件")

        # 批量提取特征
        df = self.extract_features_batch([str(p) for p in audio_paths], n_jobs=n_jobs)



        # 保存结果
        if output_csv is not None:
            df.to_csv(output_csv, index=False)
            print(f"\n结果已保存到: {output_csv}")

        # for audio_path in audio_paths:
        #     y, sr = librosa.load(audio_path, sr=self.config.sr)
        #     print(f"加载的音频时长: {len(y) / sr} 秒")  # 在此处加载并打印每个音频的时长
        return df

    def get_feature_names(self):
        """返回所有特征名称"""
        # 生成一个测试音频来获取特征名称
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.config.sr))
        features = self.extract_features_from_array(y, self.config.sr, verbose=False)
        return list(features.keys())

    def extract_features_from_array(self, y, sr, verbose=False):
        """
        从numpy数组提取特征 (用于实时处理)

        参数:
            y: 音频信号数组
            sr: 采样率
            verbose: 是否显示进度

        返回:
            features: 特征字典
        """
        features = {}

        # 重采样到目标采样率
        if sr != self.config.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.config.sr)
            sr = self.config.sr

        # 提取所有特征
        features.update(self.basic_extractor.extract_all_basic_features(y))
        features.update(self.pitch_formant_extractor.extract_all_pitch_formant_features(y, sr))
        features.update(self.harmonic_consonant_extractor.extract_all_harmonic_consonant_features(y, sr))
        features.update(self.frequency_energy_ratio_extractor.extract_all_features(y, sr))  # 提取新特征

        return features


def main():
    """主函数 - 示例用法"""

    # 创建提取器
    extractor = AudioFeatureExtractor()

    # 示例1: 处理单个文件
    # print("\n" + "=" * 60)
    # print("示例1: 处理单个音频文件")
    # print("=" * 60)
    #
    # audio_file = "example_audio.wav"  # 替换为实际文件路径
    #
    # if os.path.exists(audio_file):
    #     features = extractor.extract_features(audio_file)
    #     print("\n提取的特征:")
    #     for key, value in list(features.items())[:10]:  # 显示前10个特征
    #         print(f"  {key}: {value}")
    #     print(f"  ... (共 {len(features)} 个特征)")
    # else:
    #     print(f"文件不存在: {audio_file}")

    # 示例2: 批量处理目录
    print("\n" + "=" * 60)
    print("批量处理目录中的音频文件")
    print("=" * 60)

    audio_directory = CONFIG.input_folder  # 从 config 读取输入文件夹路径
    output_csv = CONFIG.output_csv

    if audio_directory and os.path.exists(audio_directory):
        df = extractor.extract_from_directory(
            directory=audio_directory,
            pattern=getattr(CONFIG, "file_pattern", "*.wav"),
            output_csv=output_csv,
            n_jobs=-1  # 使用所有CPU核心
        )

        if df is not None:
            print("\n特征统计:")
            print(df.describe())
    else:
        if not audio_directory:
            print("未设置输入文件夹。请在 config.py 中设置 input_folder 为您的音频目录路径。")
        else:
            print(f"目录不存在: {audio_directory}")
        print("\n创建测试音频并处理...")  # 可选，生成并处理测试音频

        # 生成测试音频
        sr = CONFIG.sr
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波

        # 从数组提取特征
        features = extractor.extract_features_from_array(y, sr)

        print("\n测试音频特征提取结果 (部分):")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value:.4f}")

        # 打印所有可用特征
        print("\n" + "=" * 60)
        print("所有可用特征列表:")
        print("=" * 60)

        feature_names = extractor.get_feature_names()
        print(f"共 {len(feature_names)} 个特征:")

        # 按类别分组显示
        categories = {
            '基础特征': ['RMS', 'Dynamic', 'Spectral', 'Onset', 'Attack', 'Decay', 'Centroid', 'Bandwidth', 'Rolloff',
                         'SEC', 'F95', 'Voicing'],
            '基频特征': ['F0', 'Jitter'],
            '共振峰特征': ['F1', 'F2', 'F3', 'F4'],
            '谐波特征': ['H1', 'H2', 'A3', 'HNR', 'HFC'],
            '辅音特征': ['Consonant'],
            '能量特征': ['energy', 'E_total'],
            '频率能量比特征': ['Energy_Ratio_60_150Hz_mean', 'Energy_Ratio_150_350Hz_mean',
                               'Energy_Ratio_350_700Hz_mean',
                               'Energy_Ratio_700_1500Hz_mean', 'Energy_Ratio_1500_3000Hz_mean',
                               'Energy_Ratio_3000_6000Hz_mean',
                               'Energy_Ratio_6000_10000Hz_mean'],
            '其他': []
        }

        for category, keywords in categories.items():
            if category == '其他':
                # 其他未分类的特征
                categorized = set()
                for cat_features in [f for k, features_list in categories.items() if k != '其他' for f in feature_names
                                     if any(kw in f for kw in categories[k])]:
                    categorized.add(cat_features)
                other_features = [f for f in feature_names if f not in categorized and f != 'filename']
                if len(other_features) > 0:
                    print(f"\n{category} ({len(other_features)}):")
                    for name in sorted(other_features)[:5]:
                        print(f"  - {name}")
                    if len(other_features) > 5:
                        print(f"  ... 还有 {len(other_features) - 5} 个")
            else:
                category_features = [f for f in feature_names if any(kw in f for kw in keywords)]
                if len(category_features) > 0:
                    print(f"\n{category} ({len(category_features)}):")
                    for name in sorted(category_features)[:5]:
                        print(f"  - {name}")
                    if len(category_features) > 5:
                        print(f"  ... 还有 {len(category_features) - 5} 个")


if __name__ == "__main__":
    main()
