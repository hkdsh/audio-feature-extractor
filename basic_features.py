"""
基础音频特征提取模块
包含: RMS, 动态范围, 谱流量, 频谱特征等
支持GPU加速
"""

import numpy as np
import librosa
import torch
from config import CONFIG


class BasicFeatureExtractor:
    """基础特征提取器"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = config.device
        
    def extract_rms_features(self, y):
        """
        提取RMS响度特征
        返回: mean, std, var, p95, variation_rate
        """
        rms = librosa.feature.rms(
            y=y, 
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length
        )[0]


        # RMS变化率
        rms_variation = np.abs(np.diff(rms))

        return {
            'RMS_mean': float(np.mean(rms)),
            'RMS_std': float(np.std(rms)),
            'RMS_var': float(np.var(rms)),
            'RMS_p95': float(np.percentile(rms, 95)),
            'RMS_variation_mean': float(np.mean(rms_variation)),
            'RMS_variation_std': float(np.std(rms_variation))
        }
    
    def extract_dynamic_range(self, y):
        """
        计算动态范围 (10-90%)
        DR = 20 * log10(P90 / P10)
        """
        # 转换为dB
        y_abs = np.abs(y)
        y_abs = y_abs[y_abs > 0]  # 过滤零值

        if len(y_abs) == 0:
            return {'Dynamic_Range_10_90': 0.0}
        
        p10 = np.percentile(y_abs, self.config.dr_low_percentile)
        p90 = np.percentile(y_abs, self.config.dr_high_percentile)

        if p10 > 0:
            dr_db = 20 * np.log10(p90 / p10)
        else:
            dr_db = 0.0
            
        return {'Dynamic_Range_10_90': float(dr_db)}
    
    def extract_spectral_flux(self, y):
        """
        提取谱流量特征
        """
        flux = librosa.onset.onset_strength(
            y=y,
            sr=self.config.sr,
            hop_length=self.config.hop_length
        )

        return {
            'Spectral_Flux_mean': float(np.mean(flux)),
            'Spectral_Flux_std': float(np.std(flux)),
            'Spectral_Flux_p95': float(np.percentile(flux, 95))
        }
    
    def extract_onset_features(self, y):
        """
        提取Onset Strength和Onset Rate
        """
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=self.config.sr,
            hop_length=self.config.hop_length
        )


        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.config.sr,
            hop_length=self.config.hop_length
        )

        # Onset times
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=self.config.sr,
            hop_length=self.config.hop_length
        )
        
        # Onset rate (events per second)
        duration = len(y) / self.config.sr
        onset_rate = len(onset_times) / duration if duration > 0 else 0
        
        return {
            'Onset_Strength_mean': float(np.mean(onset_env)),
            'Onset_Strength_std': float(np.std(onset_env)),
            'Onset_Strength_p95': float(np.percentile(onset_env, 95)),
            'Onset_Rate': float(onset_rate)
        }
    
    def extract_attack_decay_time(self, y):
        """
        计算Attack Time和Decay Time (修复版：支持连音 Legato)
        """
        rms = librosa.feature.rms(y=y, hop_length=self.config.hop_length)[0]

        # 归一化
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        
        # 找到所有能量峰值
        from scipy.signal import find_peaks
        # height=0.3 稍微降低一点峰值门槛，防止漏掉弱音
        peaks, _ = find_peaks(rms, height=0.3, distance=10)
        
        attack_times = []
        decay_times = []
        
        for peak in peaks:
            # --- 修复 Attack 计算逻辑 ---
            start_idx = peak
            # 往回走 (Walk backwards)
            while start_idx > 0:
                # 停止条件1: 能量低于阈值 (碰到了真正的静音)
                if rms[start_idx] <= 0.1:
                    break
                
                # 停止条件2: 碰到波谷 (Valley)
                # 如果当前点(start_idx)比它右边那个点(start_idx+1)能量要大，
                # 说明我们已经走过了最低点，正在爬上一个音符的Decay坡度。
                # 加一个微小的缓冲 0.001 防止噪音抖动导致的误判
                if start_idx < peak and rms[start_idx] > rms[start_idx + 1] + 0.001:
                    # 找到了波谷，这里就是起音点
                    start_idx += 1 # 回退一步到最低点
                    break
                
                start_idx -= 1
            
            attack_time = (peak - start_idx) * self.config.hop_length / self.config.sr
            attack_times.append(attack_time)
            
            # --- 修复 Decay 计算逻辑 (原理同上) ---
            end_idx = peak
            while end_idx < len(rms) - 1:
                # 停止条件1: 能量低于阈值
                if rms[end_idx] <= 0.1:
                    break
                
                # 停止条件2: 碰到波谷 (下一个音符开始了)
                if end_idx > peak and rms[end_idx] > rms[end_idx - 1] + 0.001:
                    end_idx -= 1 # 回退一步到最低点
                    break
                    
                end_idx += 1
                
            decay_time = (end_idx - peak) * self.config.hop_length / self.config.sr
            decay_times.append(decay_time)
        
        if len(attack_times) == 0:
            return {
                'Attack_Time_median': 0.0,
                'Attack_Time_p75': 0.0,
                'Decay_Time_median': 0.0,
                'Decay_Time_p75': 0.0
            }
        
        return {
            'Attack_Time_median': float(np.median(attack_times)),
            'Attack_Time_p75': float(np.percentile(attack_times, 75)),
            'Decay_Time_median': float(np.median(decay_times)),
            'Decay_Time_p75': float(np.percentile(decay_times, 75))
        }
    
    def extract_spectral_centroid(self, y):
        """
        提取频谱质心及其变化率
        """
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )[0]

        # Centroid Variability
        centroid_diff = np.abs(np.diff(centroid))
        
        return {
            'Spectral_Centroid_mean': float(np.mean(centroid)),
            'Spectral_Centroid_std': float(np.std(centroid)),
            'Centroid_Variability_STD': float(np.std(centroid)),
            'Centroid_Variability_IQR': float(np.percentile(centroid, 75) - np.percentile(centroid, 25)),
            'Centroid_Variability_mean': float(np.mean(centroid_diff))
        }
    
    def extract_spectral_bandwidth(self, y):
        """提取频谱带宽"""
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )[0]

        return {
            'Spectral_Bandwidth_mean': float(np.mean(bandwidth)),
            'Spectral_Bandwidth_std': float(np.std(bandwidth))
        }
    
    def extract_spectral_rolloff(self, y):
        """提取频谱滚降点 (Spectral Rolloff at 85%)"""
        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=self.config.sr,
            roll_percent=0.85,  # 85% 能量频率点
            hop_length=self.config.hop_length
        )[0]

        # [修复] 修正命名，从 Slope 改为 Rolloff
        return {
            'Spectral_Rolloff_mean': float(np.mean(rolloff)),
            'Spectral_Rolloff_std': float(np.std(rolloff))
        }
    
    def extract_spectral_flatness(self, y):
        """
        提取频谱平坦度 (SEC - Spectral Envelope Compactness)
        """
        flatness = librosa.feature.spectral_flatness(
            y=y,
            hop_length=self.config.hop_length
        )[0]

        return {
            'SEC_mean': float(np.mean(flatness)),
            'SEC_std': float(np.std(flatness))
        }
    
    def extract_f95(self, y):
        """
        计算F95频率 (95%能量分布的频率)
        """
        # 计算功率谱
        D = np.abs(librosa.stft(y, n_fft=self.config.n_fft))
        power = D ** 2
        
        # 沿时间轴平均
        avg_power = np.mean(power, axis=1)
        
        # 计算累积能量
        cumsum = np.cumsum(avg_power)
        total = cumsum[-1]
        
        if total > 0:
            # 找到95%能量对应的频率bin
            f95_idx = np.where(cumsum >= 0.95 * total)[0][0]
            # 转换为Hz
            freqs = librosa.fft_frequencies(sr=self.config.sr, n_fft=self.config.n_fft)
            f95_hz = freqs[f95_idx]
        else:
            f95_hz = 0.0
        
        return {'F95': float(f95_hz)}
    
    def extract_voicing_ratio(self, y):
        """
        计算浊音比 (使用频谱平坦度的逆)
        """
        flatness = librosa.feature.spectral_flatness(
            y=y,
            hop_length=self.config.hop_length
        )[0]
        
        # Voicing ratio = 1 - flatness (平坦度越低，浊音成分越多)
        voicing_ratio = 1 - flatness
        
        return {
            'Voicing_Ratio_mean': float(np.mean(voicing_ratio)),
            'Voicing_Ratio_std': float(np.std(voicing_ratio))
        }
    
    def extract_all_basic_features(self, y):
        """
        提取所有基础特征
        """
        features = {}
        
        # 依次提取各类特征
        features.update(self.extract_rms_features(y))
        features.update(self.extract_dynamic_range(y))
        features.update(self.extract_spectral_flux(y))
        features.update(self.extract_onset_features(y))
        features.update(self.extract_attack_decay_time(y))
        features.update(self.extract_spectral_centroid(y))
        features.update(self.extract_spectral_bandwidth(y))
        features.update(self.extract_spectral_rolloff(y))
        features.update(self.extract_spectral_flatness(y))
        features.update(self.extract_f95(y))
        features.update(self.extract_voicing_ratio(y))
        
        return features


# 测试代码
if __name__ == "__main__":
    # 生成测试音频
    duration = 2.0
    t = np.linspace(0, duration, int(CONFIG.sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    
    extractor = BasicFeatureExtractor()
    features = extractor.extract_all_basic_features(y)
    
    print("基础特征提取结果:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
