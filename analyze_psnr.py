import pandas as pd
import numpy as np

# Load evaluation results
df = pd.read_csv('evaluation_results.csv')

print('=' * 50)
print('CURRENT PSNR PERFORMANCE ANALYSIS')
print('=' * 50)

# PSNR statistics
psnr_values = df['psnr'].dropna()
print(f'Average PSNR: {psnr_values.mean():.2f} ± {psnr_values.std():.2f} dB')
print(f'Median PSNR: {psnr_values.median():.2f} dB')
print(f'PSNR Range: {psnr_values.min():.2f} - {psnr_values.max():.2f} dB')
print(f'Images with PSNR > 15 dB: {(psnr_values > 15).sum()} ({(psnr_values > 15).sum()/len(psnr_values)*100:.1f}%)')
print(f'Images with PSNR < 10 dB: {(psnr_values < 10).sum()} ({(psnr_values < 10).sum()/len(psnr_values)*100:.1f}%)')

# IoU statistics
iou_values = df['iou'].dropna()
print(f'\nAverage IoU: {iou_values.mean():.4f} ± {iou_values.std():.4f}')
print(f'IoU Range: {iou_values.min():.4f} - {iou_values.max():.4f}')

# ROI percentage
roi_values = df['roi_percentage'].dropna()
print(f'\nAverage ROI %: {roi_values.mean():.2f}%')
print(f'Images with 0% ROI: {(roi_values == 0).sum()} ({(roi_values == 0).sum()/len(roi_values)*100:.1f}%)')

print('\n' + '=' * 50)