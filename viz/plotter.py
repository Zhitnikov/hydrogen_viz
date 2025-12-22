from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm


def create_orbital_figure(density: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int, l: int, m: int,
                          sliced: bool = False):
    max_val = np.max(density)
    vol_data = density / max_val if max_val > 0 else density

    # Если включен режим среза, обнуляем половину данных
    if sliced:
        vol_data = vol_data.copy()
        vol_data[Y < 0] = 0  # Срезаем по оси Y
        vol_data[Y > 1] = 0  # Срезаем по оси Y

    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orb_char = orbital_names.get(l, '?')

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol_data.flatten(),
        isomin=0.03, isomax=0.5,
        opacity=0.15,
        surface_count=25,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False
    ))

    fig.update_layout(
        title=f"Орбиталь {n}{orb_char} (m={m}) {'[СРЕЗ]' if sliced else ''}",
        scene=dict(bgcolor="black"),
        paper_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def create_orbital_figure_matplotlib(density: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int, l: int, m: int):
    fig = Figure(figsize=(6, 6), dpi=100, facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    max_val = np.max(density)
    vol_data = density / max_val if max_val > 0 else density

    mask = vol_data > 0.05
    x_vis, y_vis, z_vis, v_vis = X[mask], Y[mask], Z[mask], vol_data[mask]

    if len(x_vis) > 0:
        # Рандомизированная выборка точек в зависимости от плотности вероятности
        max_points = 100000  # Максимальное количество точек для отрисовки
        
        if len(x_vis) > max_points:
            # Вероятностная выборка: вероятность отрисовки точки пропорциональна её плотности
            # Нормализуем плотности для использования в качестве весов
            v_normalized = v_vis / np.max(v_vis)
            # Добавляем минимальную вероятность для всех точек
            probabilities = 0.1 + 0.9 * v_normalized
            probabilities = probabilities / np.sum(probabilities)
            
            # Выбираем точки случайным образом с учетом вероятностей
            indices = np.random.choice(len(x_vis), size=max_points, replace=False, p=probabilities)
            
            x_vis = x_vis[indices]
            y_vis = y_vis[indices]
            z_vis = z_vis[indices]
            v_vis = v_vis[indices]
        
        # Увеличение контрастности: применяем гамма-коррекцию для усиления контраста
        v_enhanced = np.power(v_vis, 0.5)  # Гамма-коррекция для увеличения контраста
        v_normalized = (v_enhanced - np.min(v_enhanced)) / (np.max(v_enhanced) - np.min(v_enhanced) + 1e-10)
        
        depth = x_vis + y_vis + z_vis 
        d_min, d_max = np.min(depth), np.max(depth)
        norm_depth = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.ones_like(depth)

        # Увеличенная яркость и контрастность
        brightness_coeff = 0.4 + norm_depth * 1.2
        alphas = 0.4 + 0.5 * v_normalized  # Альфа зависит от плотности
        
        # Изменение цветовой схемы: используем плазму для более ярких и контрастных цветов
        cmap = cm.get_cmap('plasma')
        colors = cmap(v_normalized)
        # Нормализуем альфа-канал, чтобы значения были в диапазоне 0-1
        final_alphas = alphas * brightness_coeff
        final_alphas = np.clip(final_alphas, 0.0, 1.0)  # Ограничиваем значения 0-1
        colors[:, 3] = final_alphas

        # Адаптивный размер маркеров в зависимости от количества точек (маленькие точки)
        marker_size = max(3, min(15, 50000 / len(x_vis)))
        
        ax.scatter(x_vis, y_vis, z_vis, c=colors, 
                    s=marker_size, marker='o', edgecolors='none', depthshade=False)
        
        ax.scatter([0], [0], [0], color='red', s=45, edgecolors='white', linewidth=0.8)
        
        limit = np.max(np.abs(x_vis)) * 1.15
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    else:
        ax.text(0, 0, 0, "Нет данных", color='white', ha='center')

    ax.set_axis_off()
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orb_char = orbital_names.get(l, '?')
    ax.set_title(f"Орбиталь {n}{orb_char}", color='white', fontsize=12, y=0.95)
    
    return fig
