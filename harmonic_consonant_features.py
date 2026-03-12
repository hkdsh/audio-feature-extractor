"""
谐波和辅音特征提取模块 - 最终修复版
包含:
1. H1-H2, H1-A3 (带搜索半径优化)
2. HNR (仅浊音段)
3. HFC
4. 辅音检测 (带 Gap Filling 缝合逻辑，防止计数虚高)
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from config import CONFIG

class HarmonicConsonantExtractor:
    """谐波和辅音特征提取器"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
    def extract_all_harmonic_consonant_features(self, y, sr):
        """提取所有谐波和辅音特征"""
        features = {}
        
        # ==========================================
        # 1. 基础特征准备
        # ==========================================
        # STFT
        S = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        # Parselmouth Sound 对象
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        
        # ==========================================
        # 2. 谐波特征 (H1-H2, H1-A3, HFC)
        # ==========================================
        # 提取 Pitch & Formant
        pitch = sound.to_pitch(time_step=self.config.hop_length/sr, pitch_floor=self.config.f0_min, pitch_ceiling=self.config.f0_max)
        f0_values = pitch.selected_array['frequency']
        formant = sound.to_formant_burg(time_step=self.config.hop_length/sr, max_number_of_formants=self.config.n_formants, maximum_formant=self.config.max_formant_hz)
        
        h1_h2_list = []
        h1_a3_list = []
        hfc_list = []
        
        # 对齐帧数
        n_frames = min(S.shape[1], len(f0_values), formant.get_number_of_frames())
        
        for t in range(n_frames):
            curr_f0 = f0_values[t]
            
            # HFC (High Frequency Content) - 对所有非静音帧计算
            # 简单的能量判断
            mag_sq = S[:, t]**2
            e_sum = np.sum(mag_sq)
            if e_sum > 1e-9:
                hfc_list.append(np.sum(mag_sq * freqs) / e_sum)

            # 仅在浊音段计算 H1-H2, H1-A3
            if curr_f0 > self.config.f0_min:
                curr_time = formant.get_time_from_frame_number(t + 1)
                
                # 获取频率点
                f_h1 = curr_f0
                f_h2 = 2 * curr_f0
                f_a3 = formant.get_value_at_time(3, curr_time) # F3
                
                # 插值获取振幅
                amp_h1 = self._get_interp_amp(S[:, t], freqs, f_h1)
                amp_h2 = self._get_interp_amp(S[:, t], freqs, f_h2)
                
                if amp_h1 > 1e-6 and amp_h2 > 1e-6:
                    diff_h1h2 = 20*np.log10(amp_h1) - 20*np.log10(amp_h2)
                    h1_h2_list.append(diff_h1h2)
                    
                    # --- H1-A3 逻辑 (保留原版优秀的搜索逻辑) ---
                    if f_a3 and not np.isnan(f_a3):
                        # 在 F3 附近 +/- 0.5*F0 的范围内搜索最大的谐波峰值
                        bin_width = freqs[1] - freqs[0]
                        search_radius_hz = curr_f0 * 0.5 
                        radius_bins = int(search_radius_hz / bin_width)
                        
                        # 找到 F3 对应的中心 Bin
                        idx_f3 = np.argmin(np.abs(freqs - f_a3))
                        
                        # 确定搜索范围
                        start = max(0, idx_f3 - radius_bins)
                        end = min(len(freqs), idx_f3 + radius_bins + 1)
                        
                        # 取该范围内最大的能量值作为 A3
                        if end > start:
                            amp_a3 = np.max(S[start:end, t])
                        else:
                            amp_a3 = self._get_interp_amp(S[:, t], freqs, f_a3)

                        if amp_a3 > 1e-6:
                            diff_h1a3 = 20*np.log10(amp_h1) - 20*np.log10(amp_a3)
                            h1_a3_list.append(diff_h1a3)

        features['H1_H2_diff'] = float(np.mean(h1_h2_list)) if h1_h2_list else 0.0
        features['H1_A3_diff'] = float(np.mean(h1_a3_list)) if h1_a3_list else 0.0
        features['HFC_mean'] = float(np.mean(hfc_list)) if hfc_list else 0.0

        # ==========================================
        # 3. HNR (修复版：仅计算浊音段)
        # ==========================================
        try:
            # 1. 计算 HNR 对象
            hnr_obj = call(sound, "To Harmonicity (cc)", 0.01, self.config.f0_min, 0.1, 1.0)
            hnr_values = hnr_obj.values[0] # 获取每一帧的 HNR 值 (dB)
            
            # 2. 获取对应的 Pitch 对象用于掩码 (显式指定 time_step = 0.01 与 HNR 默认一致)
            # 注意：To Harmonicity 默认步长通常是 0.01s，但也可能随参数变。
            # 为了最安全，我们让 pitch 的步长跟随 hnr 对象的 dx
            hnr_dt = hnr_obj.dx
            pitch_for_mask = sound.to_pitch(
                time_step=hnr_dt, 
                pitch_floor=self.config.f0_min, 
                pitch_ceiling=self.config.f0_max
            )
            pitch_mask_values = pitch_for_mask.selected_array['frequency']
            
            # 对齐帧数 (取交集长度)
            min_len = min(len(hnr_values), len(pitch_mask_values))
            hnr_values = hnr_values[:min_len]
            pitch_mask_values = pitch_mask_values[:min_len]
            
            # 3. [修复核心] 创建掩码：只有 pitch > 0 且 HNR 不是 NaN 的地方才算数
            valid_mask = (pitch_mask_values > 0) & (~np.isnan(hnr_values))
            
            if np.sum(valid_mask) > 0:
                valid_hnr = hnr_values[valid_mask]
                features['HNR_mean'] = float(np.mean(valid_hnr))
            else:
                features['HNR_mean'] = 0.0 
        except Exception as e:
            print(f"HNR计算警告: {e}")
            features['HNR_mean'] = 0.0
        
        # ==========================================
        # 4. 辅音检测 (Count 修复核心)
        # ==========================================
        rms = librosa.feature.rms(y=y, frame_length=self.config.frame_length, hop_length=self.config.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=self.config.frame_length, hop_length=self.config.hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.config.n_fft, hop_length=self.config.hop_length)[0]
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.config.hop_length)
        
        # 对齐长度
        min_len = min(len(rms), len(zcr), len(flatness), len(flux))
        rms = rms[:min_len]
        zcr = zcr[:min_len]
        flatness = flatness[:min_len]
        flux = flux[:min_len]

        # --- 动态阈值计算 ---
        active_mask = rms > 0.001
        if np.sum(active_mask) > 10:
            thresh_zcr = np.percentile(zcr[active_mask], 50)
            thresh_flat = np.percentile(flatness[active_mask], 50)
            thresh_flux = np.percentile(flux[active_mask], 50)
            
            thresh_zcr_high = thresh_zcr * 1.5
            thresh_flat_high = thresh_flat * 1.5
            thresh_flux_high = thresh_flux * 2.0
        else:
            thresh_zcr_high = 0.05
            thresh_flat_high = 0.15
            thresh_flux_high = 1.0

        rms_norm = rms / (np.max(rms) + 1e-9)
        
        # --- 判定条件 ---
        
        # 条件1：能量适中
        # [调整] 下限提高到 0.01 以过滤呼吸声
        is_energy_good = (rms_norm > 0.01) & (rms_norm < 0.5)

        # 条件2：高频/噪声特征
        is_high_freq = (zcr > thresh_zcr_high) | (flatness > thresh_flat_high)

        # 条件3：强瞬态 (仅用于爆破音判定)
        is_transient = flux > thresh_flux_high

        # 核心逻辑分支
        is_fricative = is_high_freq & is_energy_good
        is_plosive = (flatness > thresh_flat_high) & is_transient & is_energy_good
        
        # 保护逻辑：排除假阳性元音
        is_vowel_core = (rms_norm > 0.6) & (flatness < 0.1)
        
        # 原始 Mask
        is_consonant_raw = (is_fricative | is_plosive) & (~is_vowel_core)
        
        # --- [关键修复] 缝合逻辑 (Gap Filling) ---
        # 如果两个 True 之间隔了不到 4 帧 (约 20ms)，就填上
        # 这能把 "爆破-停顿-送气" 缝合成一个完整的辅音
        min_gap_frames = 8 
        is_consonant_smooth = self._smooth_boolean_array(is_consonant_raw, min_gap_frames)
        
        # --- 统计逻辑 ---
        durations = []
        count = 0
        for flag in is_consonant_smooth:
            if flag:
                count += 1
            else:
                if count > 0:
                    dur_sec = count * self.config.hop_length / sr
                    # 范围：15ms 到 400ms (上限调低到400ms防止误判呼吸)
                    if 0.015 <= dur_sec <= 0.4:
                        durations.append(dur_sec)
                    count = 0
        if count > 0:
            dur_sec = count * self.config.hop_length / sr
            if 0.015 <= dur_sec <= 0.4:
                durations.append(dur_sec)

        if durations:
            features['Consonant_Duration_mean'] = float(np.mean(durations))
            features['Consonant_Duration_std'] = float(np.std(durations))
            features['Consonant_Duration_p75'] = float(np.percentile(durations, 75))
            features['Consonant_Count'] = len(durations)
            features['Articulation_Rate'] = len(durations) / (len(y)/sr) if len(y) > 0 else 0.0
        else:
            features['Consonant_Duration_mean'] = 0.0
            features['Consonant_Duration_std'] = 0.0
            features['Consonant_Duration_p75'] = 0.0
            features['Consonant_Count'] = 0
            features['Articulation_Rate'] = 0.0
            
        return features

    def _get_interp_amp(self, spectrum, freqs, target_freq):
        """频谱插值获取振幅"""
        idx = np.argmin(np.abs(freqs - target_freq))
        if idx <= 0 or idx >= len(spectrum)-1: return spectrum[idx]
        return np.max(spectrum[idx-1:idx+2])

    def _smooth_boolean_array(self, arr, max_gap):
        """填补布尔数组中的微小空隙 (False gaps)"""
        n = len(arr)
        output = arr.copy()
        
        i = 0
        while i < n:
            if not arr[i]: # 这是一个 False (空隙)
                # 寻找这个空隙的终点
                j = i + 1
                while j < n and not arr[j]:
                    j += 1
                gap_len = j - i
                
                # 检查这个 gap 是否被两端的 True 包围
                # 只有当 gap 左边是辅音，右边也是辅音，且 gap 很短时，才缝合
                if i > 0 and j < n: 
                    if gap_len <= max_gap:
                        output[i:j] = True # 填补为 True
                
                i = j
            else:
                i += 1
        return output