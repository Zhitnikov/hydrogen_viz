from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm

def create_orbital_figure(density: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, n: int, l: int, m: int):
    max_val = np.max(density)
    vol_data = density / max_val if max_val > 0 else density
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orb_char = orbital_names.get(l, '?')
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol_data.flatten(),
        isomin=0.04, isomax=0.5,
        opacity=0.15,
        surface_count=25,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='red', opacity=1.0, symbol='circle'),
        name='Ядро'
    ))

    fig.update_layout(
        title=f"Орбиталь {n}{orb_char} (m={m})",
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
        depth = x_vis + y_vis + z_vis 
        d_min, d_max = np.min(depth), np.max(depth)
        norm_depth = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.ones_like(depth)

        brightness_coeff = 0.6 + norm_depth * 0.8
        alphas = 0.3 * brightness_coeff
        
        cmap = cm.get_cmap('viridis')
        colors = cmap(v_vis)
        colors[:, 3] = alphas

        ax.scatter(x_vis, y_vis, z_vis, c=colors, 
                    s=130, marker='o', edgecolors='none', depthshade=False)
        
        ax.scatter([0], [0], [0], color='red', s=45, edgecolors='white', linewidth=0.8)
        
        limit = np.max(np.abs(x_vis)) * 1.02
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
