---
marp: true
math: mathjax
---

# 基於時序卷積網路之單FMCW雷達應用於非接觸式即時生命特徵監控 

---

## 緒論

市面上有許多監測心率及呼吸率的產品，依監測方式可分為接觸式及非接觸式兩種。
接觸式裝置的精確度通常較非接觸式來得高，然而也會有長期穿戴不適等問題。

#### 接觸式
- 智慧手錶/手環、夾式血氧儀

#### 非接觸式
- **無線雷達**、雷射測距儀、熱成像攝影

## 此次研究使用的是**調頻連續波(FMCW)雷達**

---

# 心率/呼吸率提取方法

---
- 資料前處理
  1. FMCW 雷達 ->時域訊號
  2. 時域訊號->Range FFT->頻域訊號 
  3. 頻域訊號->靜態雜波濾除->動態物體頻率與距離
  4. 用拍頻訊號之相位差提取心率及呼吸率
- 時序卷積網路
- Transformer Encoder
- Regressor

---

# 調頻連續波雷達 Frequency Modulated Continuous Waveform Radar

---

![FMCW](/slides/img/TCN/img4.png)

---

## 連續調變波

- 應用鎖相迴路（Phase-Locked Loop，PLL）作為訊號產生器。
- 透過不斷調整 PLL 裡的壓控振盪器(Voltage-Controlled Oscillator, VCO) 的頻率來生成頻率連續調變的信號

---

![WAVE](/slides/img/TCN/img3.png)

---

## 訊號表示

- 發射訊號:  $x_T(t) = A_T\cos(2\pi f_T(t)t + \Phi_T(t))$, $A_T$為發射的傳輸能量大小
- 接收訊號:  $x_R(t) = A_R\cos(2\pi f_R(t)t + \Phi_R(t))$, $A_R$為接收的傳輸能量大小

#### 我們將 LNA 加強後的訊號與發射訊號做混頻處理，經過混頻後的訊號稱為拍頻訊號

- 拍頻訊號: $x_T(t) \cdot x_R(t) = A_T\cos(2\pi f_T(t)t + \Phi_T(t)) \cdot A_R\cos(2\pi f_R(t)t + \Phi_R(t))$

#### 將混頻完的訊號取低頻部分，即為我們所需的基頻訊號

- 基頻訊號: 
  $$x(t) = \frac{1}{2}A_TA_R \cdot \cos (2\pi (f_T(t)-f_R(t))t + (\Phi_T(t) - \Phi_R(t)))$$
  $\qquad\qquad\quad\ = A\cos (2\pi f_b(t)t + \Phi_b(t)t)$. $\Phi_b$為拍頻訊號相位隨時間的變化
