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

---

## 數位訊號處理器

**將類比訊號取樣來獲得數位訊號**

- 取樣表示
  $$x[n, m] = A\cos (2\pi f_b(nT_f+mT_s)nT_f + \Phi_b(nT_f+mT_s))$$

- $T_f$ 快速時間 (Fast-time): 針對每個 chirp 裡的時間做取樣
- $T_s$ 慢速時間 (Slow-time): 針對不同 chrip 之間的時間間隔做取樣

---

## 距離推導

FMCW 雷達發射訊號的頻率會隨著時間變化，假設一個 chirp 的起始頻率為$f_s$、截止頻率為$f_e$、週期為$T_s$、頻寬為$B$、斜率為$S$，則可以表示成以下式子: 

- $B = F_e-F_s$
- $S = \frac{B}{T_s}$

毫米波以光速$C$傳遞，因此可以透過前面得到的拍頻訊號經由下式推導出距離$d(t)$

- $t_d = \frac{2d(t)}{C}$
- $f_b = St_d = \frac{B}{T_s} \cdot \frac{2d(t)}{C}$

一維快速傅立葉轉換將協助我們取得$f_b$, 從而算得$2d(t)$。

---

## 一維快速傅立葉轉換 Range FFT

我們得到經取樣過的拍頻訊號後，將每一個快速時間取樣點做 Range FFT，經過 FFT 的表示如下:

$$X_m[K] = \sum^{N-1}_{n=0}x[n,m]e^{-j\frac{2\pi nk}{N}}, k = 0, ..., N-1$$

其中 $K$ 為 FFT 指標(index)，$N$ 為每個 chirp 裡的取樣數。當第 $K$ 個達到最大值時，即可透過距離分辨率換算物體距離。其換算的公式如下:

$$d = \frac{t_d\cdot C}{2} = f_bT_c\cdot \frac{C}{2B}$$

其中$C=3\times 10^8$為光速，$B$為 chirp 之頻寬。 $\frac{C}{2B}$定義為距離分辨率。

---

## 靜態雜波濾除 Clutter removal

- 利用平滑處理來濾出環境中的靜態背景物件
  
$$X'_m[k] = \sigma X_m[k]+ (1-\sigma )X_{m-1}[k], 0\leq \sigma \leq 1$$

- $\sigma$為平滑系數，$\sigma$越大，平滑效果越好，但較多的目標訊號也會被平滑掉；$\sigma$越小，平滑效果越差，較多的靜態雜波會被保留。
- 最後，將平滑處理後的靜態雜波圖與原本未經處理的訊號相減，便能得到濾除靜態雜波後的結果。其表示如下:

$$Y_m[K] = X_m[k]-X'_m[k]$$

從這些$Y_m[K]$找出最大值的頻率，結合推導的距離分辨率便能計算出物體距離，其表示如下: 

$$k_{\max} = \arg_k\max |Y_m[K]|$$
$$d = f_bT_c\cdot \frac{C}{2B} = k_{\max}\cdot \frac{C}{2B}$$

---

## 相位提取

- 我們可以將拍頻訊號以胸壁運動振幅表示為以下：
$$x(t) = A\cos (2\pi f_b(t)t+\phi(t))$$
$$ = A\cos(2\pi(\frac{2B}{T_sC}\cdot (d_0+\Delta d(t)))t+(\phi_0+\Delta\phi(t)))$$
- 其中$d_0$及$\phi_0$為胸壁距離雷達的初始位置相位，$\Delta d(t)$及$\Delta\phi(t)$為心跳及呼吸而造成的胸壁運動振幅與相位。
- $\Delta d(t)$亦可表示成以下：
$$\Delta d(t) = A_{hr}\sin(2\pi f_{hr}t)+A_{rr}\sin(2\pi f_{rr}t)$$
- 其中$A_{hr}, A_{rr}$分別為心跳及呼吸之振幅，$f_{hr},f_{rr}$分別為心跳及呼吸之頻率。

---

## 相位計算 

- 透過拍頻訊號的相位差來提取心跳及呼吸頻率：$\Delta\phi(t) = 4\pi\frac{\Delta d(t)}{\lambda}$
- 取出$Y_m[k_{\max}]$ 的實部及虛部做$\arctan$: $\phi_m =\arctan(\frac{Im(Y_m[k_{\max}])}{Re(Y_m[k_{\max}])})$
- 若兩個連續相位差的結果超過$-\pi$ ~ $\pi$之間，補上$-2\pi$~$2\pi$的落差值，將相位差範圍限制在$-\pi$ ~ $\pi$之間。
$$
\phi_{b,m} =
\begin{cases}
\phi_m + 2\pi, & \phi_m - \phi_{m-1} > \pi \\
\phi_m - 2\pi, & \phi_m - \phi_{m-1} < -\pi \\
\phi_m, & \text{otherwise}
\end{cases}
$$

- 結合此頁第一式：$\phi_{b,m} = \frac{4\pi}{\lambda}[A_{hr}\sin(2\pi f_{hr}t)+A_{rr}\sin(2\pi f_{rr}t)]$
- 此時的$\phi_{b,m}$為胸壁造成的相位差，其包含了呼吸及心率兩種相位。

---

## 相位分離

- 串聯兩個雙二階的 IIR 濾波器，每個雙二階的 IIR 濾波器的結構為直接 II 階型(Direct-form II)，其差分方程如下:
$$y[n] = b_0\omega[n]+b_1\omega[n-1]+b_2\omega[n-2]$$
$$\text{where}\ \  \omega[n] = x[n]-a_1\omega[n-1]-a_2\omega[n-2]$$

- 將截止頻率設定為頻率$(0.1-0.6\text{Hz})$較符合人體呼吸的合理範圍頻率，則可得到呼吸相位$\Phi_{rr,m}$;同理，也能將截止頻率設定為頻率$(0.8-2\text{Hz})$較符合人體心跳的合理範圍頻率，也能得到心率相位$\Phi_{hr,m}$

---

## 循環暫存器

- 蒐集一段時間的$\Phi_{rr,m}$，並放入長度為𝐿的循環暫存器(Circular Buffer)內，確保每次取樣皆能更新呼吸率。
- 將循環暫存器內所有相位做$L$點的 FFT 即可得到呼吸頻率$f_{rr}$，其表示如下:
$$\Phi_{m, rr}[l] = \sum^{L-1}_{n=0}\Phi[n]e^{-j\frac{2\pi ml}{L}}, l = 0, ..., L-1$$
$$f_{rr} = arc_l\max|\Phi_{m, rr}[l]|\cdot\frac{1}{T_sL}$$

- 在心率估計上，則同理呼吸率
