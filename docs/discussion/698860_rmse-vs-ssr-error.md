# rmse_vs_ssr_error

- 投稿者: jose mata
- 投稿日時: 2026-05-11 20:08:44.625000
- 投票数: 4
- コメント数: 1（取得数: 1）
- トピックID: `698860`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698860](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698860)

## 本文

<p><em>I don't know if it will be of any use, however I'm sharing it just in case it might be helpful to someone.</em>
`"""</p>
<p>General Equation for RMSE:</p>
<p>RMSE = sqrt( (1/n) * Σ(y_i - ŷ_i)² )</p>
<p>Definition of the Sum of Squared Residuals (SSR):</p>
<p>SSR = Σ(y_i - ŷ_i)²</p>
<p>Simplified Form (for n = 14151):</p>
<p>RMSE = 8.41e-3 * sqrt(SSR)</p>
<p>Interpretation:</p>
<p>Since k = 1/sqrt(14151) ≈ 8.41e-3, the RMSE is defined as</p>
<p>the product of the statistical scale factor and the linear magnitude of the total error (L2 norm).</p>
<p>"""</p>
<p>import numpy as np</p>
<p>import matplotlib.pyplot as plt</p>
<p>ssr = np.linspace(0, 3700000, 1000)</p>
<p>k = 8.41e-3 </p>
<p>y = k * np.sqrt(ssr)</p>
<p>plt.figure(figsize=(10, 6))</p>
<p>plt.plot(ssr, y, color='black', linewidth=2, label='RMSE')</p>
<p>plt.fill_between(ssr, 8, 16, color='r', alpha=0.3, label='Critical region')</p>
<p>plt.fill_between(ssr, 2, 8, color='g', alpha=0.3, label='Optimal region')</p>
<p>plt.fill_between(ssr, 0.9, 2, color='b', alpha=0.3, label='Excellent region')</p>
<p>plt.fill_between(ssr, 0, 0.9, color='y', alpha=0.3, label='Overfitting')</p>
<p>plt.title('RMSE vs SSR (n = 14151)')</p>
<p>plt.xlabel('SSR (ft²)')</p>
<p>plt.ylabel('RMSE (ft)')</p>
<p>plt.xlim(0, 3700000)</p>
<p>plt.ylim(0, 18)</p>
<p>plt.legend(loc='upper left')</p>
<p>plt.grid(True, linestyle='--', alpha=0.6)</p>
<p>plt.show()`</p>

## コメント

### コメント 1 — jose mata

- 投稿日時: 2026-05-11 20:09:23.287000
- 投票数: 0
- コメントID: `3456410`

<p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F28215536%2Ff5ba33574eadc12bc6ee7371cbc56475%2Fdownload%20(1).png?generation=1778530159836179&alt=media" alt=""></p>
