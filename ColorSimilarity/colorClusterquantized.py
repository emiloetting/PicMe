import cv2	
import	numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from scipy.spatial import cKDTree   # For mapping color signatures to quantized LAB colors, might replace with ANNOY
from itertools import product


cwd = os.getcwd()
image = "my_test_file.jpg"
filepath  = os.path.join(cwd, 'ColorSimilarity',  image)
filepath1 = os.path.join(cwd, 'ColorSimilarity', "Schwan.jpeg")
filepath2 = os.path.join(cwd, 'ColorSimilarity', "Schwan.jpeg")




def extract_color_signature(img_path: str, n_clusters: int) -> tuple:
    """
    Extracts the color signature of an image using k-means clustering in the Lab color space.
    The function normalizes the pixel values, applies k-means clustering, and returns the cluster centers and their weights.

    The cluster centers are converted back to BGR color space for visualization.
    The weights are normalized to sum to 1.

    Arguments:
        img_path [str]: Path to the image file.

    Returns:
        [centers (LAB) [np.array], weights[np.array]]
    """
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    pixels = np.float32(img.reshape(-1, 3))

    #Werte normalisieren
    pixels[:, 0] /= 100.0  
    pixels[:, 1:] += 128.0  
    pixels[:, 1:] /= 255.0	    
    # 8 cluster
    k=n_clusters
    # Abbruchkritereien (komische opencv kacke, "+"" weil beide kriterien gelten, 100 Durchläufe, 0.1 epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    # k=8 cluster, keine Labels, Abbruchkriterien von oben, 10 Durchläufe mit anderen Zentren, zufällige center
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Gewichte berechnen
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(k)
    for i, label in enumerate(unique_labels):
        weights[label] = counts[i]
    weights = weights / np.sum(weights)  # Normalisierung

    
    return [centers, weights]


def visualize_color_signature(centers, weights, figsize=(10, 4)):
    """
    Visualisiert eine Farbsignatur.
    
    Parameters:
    -----------
    centers : numpy.ndarray
        Die k repräsentativen Farben
    weights : numpy.ndarray
        Die relative Häufigkeit jeder Farbe
    """
    # Sortiere Farben nach Gewicht (absteigend)
    sorted_indices = np.argsort(weights)[::-1]
    sorted_centers = centers[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Erstelle Farbbalken
    plt.figure(figsize=figsize)
    
    # Jede Farbe als Balken darstellen
    for i, (center, weight) in enumerate(zip(sorted_centers, sorted_weights)):
        # Konvertiere BGR zu RGB für matplotlib
        rgb_color = center[::-1] / 255.0  # BGR zu RGB und auf [0, 1] normalisieren
        plt.bar(i, weight, color=rgb_color, width=0.8)
        
        # Prozentsatz anzeigen
        plt.text(i, weight + 0.01, f"{weight*100:.1f}%", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Farbcluster')
    plt.ylabel('Relative Häufigkeit')
    plt.title('Farbsignatur des Bildes')
    plt.xticks([])  # x-Achse ohne Zahlen
    plt.ylim(0, max(weights) * 1.2)  # Ein wenig Platz über dem höchsten Balken
    plt.show()


def compare_images(weights1, centers1, weights2, centers2):
    """
    Vergleicht zwei Farbsignaturen mithilfe der Earth Mover's Distance (EMD).

    Die Funktion berechnet die EMD zwischen zwei Farbsignaturen, die durch ihre Gewichte und Zentren dargestellt werden.
    Die Gewichte und Zentren werden in numpy-Arrays umgewandelt, bevor die EMD berechnet wird.

    Arguments:
    
            weights1 [np.array]: Gewichte der ersten Farbsignatur.
            centers1 [np.array]: Zentren der ersten Farbsignatur.
            weights2 [np.array]: Gewichte der zweiten Farbsignatur.
            centers2 [np.array]: Zentren der zweiten Farbsignatur.

    Returns:
            [distance [float]]: Die berechnete EMD zwischen den beiden Farbsignaturen.
    """
    weights1 = np.array(weights1, dtype=np.float32)
    weights2 = np.array(weights2, dtype=np.float32)
    centers1 = np.array(centers1, dtype=np.float32)
    centers2 = np.array(centers2, dtype=np.float32)


    signature1 = np.column_stack((weights1, centers1))
    signature2 = np.column_stack((weights2, centers2))


    distance, _,_ = cv2.EMD(signature1,signature2,cv2.DIST_L1)

    return distance

def get_quantized_LAB(l_bins: int, a_bins: int, b_bins: int) -> np.ndarray:
    """
    Erzeugt eine quantisierte LAB-Farbtabelle.

    Parameters:
    -----------
    l_bins : int
        Anzahl der L-Bins
    a_bins : int
        Anzahl der a-Bins
    b_bins : int
        Anzahl der b-Bins

    Returns:
    --------
    quantized_lab : numpy.ndarray
        Quantisierte LAB-Farbtabelle
    """
    
    l = np.linspace(0, 100, l_bins)
    a = np.linspace(-128, 127, a_bins)
    b = np.linspace(-128, 127, b_bins)

    quantized_lab = np.array(list(product(l, a, b)))    # build cartesian product of l, a, b
    
    # make sure values are of type int
    quantized_lab = quantized_lab.astype(np.uint8)

    return quantized_lab


if __name__ == "__main__":
    centers, weights = extract_color_signature(filepath)
    visualize_color_signature(centers, weights)
    centers1, weights1 = extract_color_signature(filepath1)
    centers2, weights2 = extract_color_signature(filepath2)
    dauer = timeit.timeit(lambda : compare_images(weights1, centers1, weights2, centers2), number=500_000)
    print(dauer)

