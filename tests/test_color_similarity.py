import numpy as np
import sqlite3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ColorSimilarity.colorClusterquantized import quantized_image, get_bin_centers


def test_quantized_image_basic():
    """Test basic quantized image functionality"""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]  # BGR values
    
    hist = quantized_image(img, l_bins=3, a_bins=3, b_bins=3, normalization='L1')
    
    assert hist is not None
    assert hist.shape == (27,)  # 3*3*3 = 27 bins
    assert np.isclose(np.sum(hist), 1.0)  # L1 normalized


def test_quantized_image_normalization():
    """Test different normalizations"""
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    hist_l1 = quantized_image(img, l_bins=2, a_bins=2, b_bins=2, normalization='L1')
    hist_l2 = quantized_image(img, l_bins=2, a_bins=2, b_bins=2, normalization='L2')
    
    assert np.isclose(np.sum(hist_l1), 1.0)  # L1 sum = 1
    assert np.isclose(np.linalg.norm(hist_l2), 1.0)  # L2 norm = 1


def test_get_bin_centers():
    """Test bin centers calculation"""
    centers = get_bin_centers(l_bins=2, a_bins=2, b_bins=2)
    
    assert centers.shape == (8, 3)  # 2*2*2 = 8 centers, 3 channels (LAB)
    assert centers.dtype in [np.float32, np.float64]


def test_database_creation_and_retrieval():
    """Test database creation and data retrieval"""
    db_path = "test_color_db.db"
    
    # Create demo db
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE color_db (
            ann_index INTEGER PRIMARY KEY,
            path TEXT,
            L1_embedding BLOB,
            image_size INTEGER DEFAULT 0
        )
    ''')
    
    # input single data
    test_embedding = np.random.rand(27).astype(np.float64)
    c.execute(
        "INSERT INTO color_db (ann_index, path, L1_embedding, image_size) VALUES (?, ?, ?, ?)",
        (0, "/test/path.jpg", test_embedding.tobytes(), 1024)
    )
    
    conn.commit()
    conn.close()
    
    # check single grab
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT L1_embedding, path FROM color_db WHERE ann_index = 0")
    result = c.fetchone()
    
    assert result is not None
    embedding_bytes, path = result
    embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
    assert embedding.shape == (27,)
    assert path == "/test/path.jpg"
    
    conn.close()
    os.remove(db_path)


def test_database_multiple_entries():
    """Test retrieving multiple database entries"""
    db_path = "test_multi_db.db"
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE color_db (
            ann_index INTEGER PRIMARY KEY,
            path TEXT,
            L1_embedding BLOB,
            image_size INTEGER DEFAULT 0
        )
    ''')
    
    # check multiple inserts
    for i in range(5):
        test_embedding = np.random.rand(27).astype(np.float64)
        c.execute(
            "INSERT INTO color_db (ann_index, path, L1_embedding, image_size) VALUES (?, ?, ?, ?)",
            (i, f"/test/path{i}.jpg", test_embedding.tobytes(), 1024)
        )
    
    conn.commit()
    
    # check multiple returns
    indices = [0, 2, 4]
    placeholders = ','.join(['?' for _ in indices])
    query = f"SELECT L1_embedding, path FROM color_db WHERE ann_index IN ({placeholders})"
    
    c.execute(query, indices)
    results = c.fetchall()
    
    assert len(results) == 3
    
    conn.close()
    os.remove(db_path)


def test_histogram_properties():
    """Test histogram properties are maintained"""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    hist_l1 = quantized_image(img, l_bins=5, a_bins=9, b_bins=9, normalization='L1')
    hist_l2 = quantized_image(img, l_bins=5, a_bins=9, b_bins=9, normalization='L2')
    
    # shaps
    assert hist_l1.shape == (405,) 
    assert hist_l2.shape == (405,)
    
    # Test normalization
    assert np.isclose(np.sum(hist_l1), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(hist_l2), 1.0, atol=1e-6)
    
    # only pos vals
    assert np.all(hist_l1 >= 0)
    assert np.all(hist_l2 >= 0)


def test_histogram_differences():
    """Test that different images produce different histograms"""
    # Make 2 demo imgs
    img1 = np.full((32, 32, 3), [255, 255, 255], dtype=np.uint8)  # White
    img2 = np.full((32, 32, 3), [0, 0, 0], dtype=np.uint8)        # Black
    
    hist1 = quantized_image(img1, l_bins=3, a_bins=3, b_bins=3, normalization='L1')
    hist2 = quantized_image(img2, l_bins=3, a_bins=3, b_bins=3, normalization='L1')
    
    # diff hists
    assert not np.array_equal(hist1, hist2)
    
    # normalizations
    assert np.isclose(np.sum(hist1), 1.0)
    assert np.isclose(np.sum(hist2), 1.0)


def test_bin_centers_properties():
    """Test bin centers have correct properties"""
    centers = get_bin_centers(l_bins=3, a_bins=3, b_bins=3)
    
    # correct amnt centers
    assert centers.shape == (27, 3)
    
    # L vals in [0, 100]
    assert np.all(centers[:, 0] >= 0) and np.all(centers[:, 0] <= 100)


def test_database_schema():
    """Test correct database schema creation"""
    db_path = "test_schema_db.db"
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE color_db (
            ann_index INTEGER PRIMARY KEY,
            path TEXT,
            L1_embedding BLOB,
            image_size INTEGER DEFAULT 0
        )
    ''')
    
    # Schene check
    c.execute("PRAGMA table_info(color_db);")
    columns = c.fetchall()
    
    # 4 cols
    assert len(columns) == 4
    
    # Check col names + types
    column_info = {col[1]: col[2] for col in columns}  
    assert column_info['ann_index'] == 'INTEGER'
    assert column_info['path'] == 'TEXT'
    assert column_info['L1_embedding'] == 'BLOB'
    assert column_info['image_size'] == 'INTEGER'
    
    conn.close()
    os.remove(db_path)


def test_embedding_roundtrip():
    """Test embedding storage and retrieval roundtrip"""
    # Create test embedding
    original_embedding = np.random.rand(405).astype(np.float64)
    
    # Convert to bytes and back
    embedding_bytes = original_embedding.tobytes()
    retrieved_embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
    
    # Should be identical
    assert np.array_equal(original_embedding, retrieved_embedding)
    assert original_embedding.shape == retrieved_embedding.shape
    assert original_embedding.dtype == retrieved_embedding.dtype