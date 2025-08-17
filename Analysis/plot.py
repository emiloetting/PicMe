import numpy as np
import plotly.express as px
from PIL import Image
import base64
import io



def img_to_base64_robust(img_array, size=(100, 100)):
    try:
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return None
            
        img = Image.fromarray(img_array, mode='RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

if __name__ == "__main__":
    rgb_ims = np.load("Analysis/reduced_images_rgb.npy")
    reduced_embeds = np.load("Analysis/dim_reduced_clip_embeds.npy")
    labels = np.load("Analysis/cluster_labels.npy")
    centroids_2d = np.load("Analysis/cluster_centroids.npy")

    fig = px.scatter(
        x=reduced_embeds[:, 0],
        y=reduced_embeds[:, 1],
        color=labels.astype(str),
        title="2D CLIP-embeddings colored via KMeans-clustering",
        opacity=0.6
    )

    # Alle Punkte gleich groß machen
    fig.update_traces(marker=dict(size=1.0))  # Sehr winzig

    x_range = max(reduced_embeds[:, 0]) - min(reduced_embeds[:, 0])
    y_range = max(reduced_embeds[:, 1]) - min(reduced_embeds[:, 1])
    
    image_scale = 0.03
    image_size_x = x_range * image_scale
    image_size_y = y_range * image_scale

    num_images = min(len(centroids_2d), len(rgb_ims))
    
    for i in range(num_images):
        cx, cy = centroids_2d[i]
        img_base64 = img_to_base64_robust(rgb_ims[i], size=(80, 80))
        
        if img_base64 is not None:
            fig.add_layout_image(
                dict(
                    source=img_base64,
                    xref="x",
                    yref="y",
                    x=cx,
                    y=cy,
                    xanchor="center",
                    yanchor="middle",
                    sizex=image_size_x,
                    sizey=image_size_y,
                    sizing="contain",
                    layer="above"
                )
            )

    fig.update_layout(
        showlegend=False,
        width=1400,
        height=1000,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.write_html("Analysis/cluster_plot.html")