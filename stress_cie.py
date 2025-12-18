import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def rgb_to_cie_xy(r, g, b):
    r = np.array(r) / 255.0
    g = np.array(g) / 255.0
    b = np.array(b) / 255.0

    def gamma_correction(channel):
        return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)

    r_lin = gamma_correction(r)
    g_lin = gamma_correction(g)
    b_lin = gamma_correction(b)

    X = r_lin * 0.4124 + g_lin * 0.3576 + b_lin * 0.1805
    Y = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722
    Z = r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505

    denom = X + Y + Z
    x = np.where(denom == 0, 0, X / denom)
    y = np.where(denom == 0, 0, Y / denom)

    return x, y, Y


df = pd.read_excel('Standard s-RGB.xlsx')

stress = df['Stress'].values

r = df['R'].values
g = df['G'].values
b = df['B'].values


t = (stress - stress.min()) / (stress.max() - stress.min())

r_spline = CubicSpline(t, r)
g_spline = CubicSpline(t, g)
b_spline = CubicSpline(t, b)
stress_spline = CubicSpline(t, stress)

t_dense = np.linspace(0, 1, 500)

t_combined = np.unique(np.concatenate([t, t_dense]))

r_dense = r_spline(t_combined)
g_dense = g_spline(t_combined)
b_dense = b_spline(t_combined)
stress_dense = stress_spline(t_combined)

print(stress_dense)

lut_df = pd.DataFrame({
    'R': r_dense,
    'G': g_dense,
    'B': b_dense,
    'Stress': stress_dense
})

cie_x, cie_y, Y = rgb_to_cie_xy(r_dense, g_dense, b_dense)

cie_df = pd.DataFrame({
    'Stress': stress_dense,
    'CIE_x': cie_x,
    'CIE_y': cie_y,
    'Y': Y
})

cie_df.to_csv('Stress_CIE_xy.csv', index=False)



