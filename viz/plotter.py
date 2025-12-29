from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm, colors as mcolors

BOHR_TO_ANGSTROM = 0.529177
# Полуширина среза в борнах (тонкий слой)
SLICE_HALF_BOHR = 0.5


def create_orbital_figure(density: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int, l: int, m: int,
                          sliced: bool = False):
    max_val = np.max(density)
    vol_data = density / max_val if max_val > 0 else density

    if sliced:
        vol_data = vol_data.copy()
        vol_data[np.abs(Y) > SLICE_HALF_BOHR] = 0

    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orb_char = orbital_names.get(l, '?')

    # Переводим координаты в ангстремы для подписей осей
    x_ang = (X * BOHR_TO_ANGSTROM).flatten()
    y_ang = (Y * BOHR_TO_ANGSTROM).flatten()
    z_ang = (Z * BOHR_TO_ANGSTROM).flatten()

    fig = go.Figure(
        data=go.Volume(
            x=x_ang,
            y=y_ang,
            z=z_ang,
            value=vol_data.flatten(),
            isomin=0.03,
            isomax=0.5,
            opacity=0.15,
            surface_count=25,
            colorscale="Viridis",
            cmin=0,
            cmax=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            colorbar=dict(
                title="Плотность",
                tickformat=".2f",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                thickness=16,
                len=0.75,
            ),
        )
    )

    fig.update_layout(
        title=f"Орбиталь {n}{orb_char} (m={m}) {'[СРЕЗ]' if sliced else ''}",
        scene=dict(
            bgcolor="black",
            xaxis_title="x, Å",
            yaxis_title="y, Å",
            zaxis_title="z, Å",
            aspectmode="cube",
        ),
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
        max_points = 100000
        
        if len(x_vis) > max_points:
            v_normalized = v_vis / np.max(v_vis)
            probabilities = 0.1 + 0.9 * v_normalized
            probabilities = probabilities / np.sum(probabilities)
            indices = np.random.choice(len(x_vis), size=max_points, replace=False, p=probabilities)
            x_vis = x_vis[indices]
            y_vis = y_vis[indices]
            z_vis = z_vis[indices]
            v_vis = v_vis[indices]
        
        v_enhanced = np.power(v_vis, 0.5)
        v_normalized = (v_enhanced - np.min(v_enhanced)) / (np.max(v_enhanced) - np.min(v_enhanced) + 1e-10)
        
        depth = x_vis + y_vis + z_vis 
        d_min, d_max = np.min(depth), np.max(depth)
        norm_depth = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.ones_like(depth)

        brightness_coeff = 0.4 + norm_depth * 1.2
        alphas = 0.4 + 0.5 * v_normalized
        
        cmap = cm.get_cmap('plasma')
        colors = cmap(v_normalized)
        final_alphas = alphas * brightness_coeff
        final_alphas = np.clip(final_alphas, 0.0, 1.0)
        colors[:, 3] = final_alphas

        marker_size = max(3, min(15, 50000 / len(x_vis)))

        mask_slice = y_vis <= 0
        x_vis = x_vis[mask_slice]
        y_vis = y_vis[mask_slice]
        z_vis = z_vis[mask_slice]
        colors = colors[mask_slice]

        x_vis_ang = x_vis * BOHR_TO_ANGSTROM
        y_vis_ang = y_vis * BOHR_TO_ANGSTROM
        z_vis_ang = z_vis * BOHR_TO_ANGSTROM
        
        ax.scatter(
            x_vis_ang,
            y_vis_ang,
            z_vis_ang,
            c=colors,
            s=marker_size,
            marker='o',
            edgecolors='none',
            depthshade=False,
        )
        
        ax.scatter([0], [0], [0], color='red', s=45, edgecolors='white', linewidth=0.8)
        
        limit = np.max(np.abs(x_vis_ang)) * 1.15
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.02, label="Отн. плотность")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")
    else:
        ax.text(0, 0, 0, "Нет данных", color='white', ha='center')

    ax.set_axis_off()
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orb_char = orbital_names.get(l, '?')
    ax.set_title(f"Орбиталь {n}{orb_char} (коорд. в Å)", color='white', fontsize=12, y=0.95)
    
    return fig
