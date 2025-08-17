import pytest
import sqlite3
import cv2
import numpy as np
import pickle
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
ssim_dir = os.path.join(parent_dir, 'SSIM')

# Add both directories to Python path
sys.path.insert(0, parent_dir)
sys.path.insert(0, ssim_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"SSIM dir: {ssim_dir}")
print(f"SSIM exists: {os.path.exists(ssim_dir)}")

try:
    from create_hash_database import DatabaseBuilder, build_complete_database
    from hash import Hash, HashDatabase, hamming_distance, get_similar_images, insert_hashes
    from ssim import get_ssim, get_ssim_single, get_ssim_multiple
except ImportError as e:
    from SSIM.create_hash_database import DatabaseBuilder, build_complete_database
    from SSIM.hash import Hash, HashDatabase, hamming_distance, get_similar_images, insert_hashes
    from SSIM.ssim import get_ssim, get_ssim_single, get_ssim_multiple

# Fixtures
@pytest.fixture
def temp_db():
    """Create temporary database"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)



@pytest.fixture
def test_image():
    """Create a test image"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img_path = f.name
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    yield img_path
    if os.path.exists(img_path):
        os.unlink(img_path)


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images"""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ext in enumerate(['.jpg', '.png']):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(temp_dir, f'test_{i}{ext}'), img)
        yield temp_dir


@pytest.fixture
def db_with_data(temp_db):
    """Create database with test data"""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE image_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            hash TEXT,
            created_at TEXT,
            image_32x32 BLOB
        )
    ''')
    
    # Insert test data
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    test_data = [
        (1, '/path/to/image1.jpg', '1010101010101010101010101010101010101010101010101010101010101010', '2024-01-01T12:00:00', pickle.dumps(test_img)),
        (2, '/path/to/image2.jpg', '1010101010101011101010101010101010101010101010101010101010101010', '2024-01-01T12:00:00', pickle.dumps(test_img)),
        (3, '/path/to/image3.jpg', None, '2024-01-01T12:00:00', pickle.dumps(test_img)),
    ]
    
    cursor.executemany('''
        INSERT INTO image_hashes (id, image_path, hash, created_at, image_32x32)
        VALUES (?, ?, ?, ?, ?)
    ''', test_data)
    
    conn.commit()
    conn.close()
    yield temp_db


# create_hash_database.py tests
def test_database_builder_init(temp_db):
    """Test DatabaseBuilder.__init__"""
    builder = DatabaseBuilder(temp_db)
    assert builder.db_path == temp_db


def test_create_database(temp_db):
    """Test DatabaseBuilder.create_database"""
    builder = DatabaseBuilder(temp_db)
    builder.create_database()
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_hashes'")
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None


def test_add_images_from_directory(temp_image_dir):
    """Test DatabaseBuilder.add_images_from_directory"""
    builder = DatabaseBuilder("dummy.db")
    image_paths = list(builder.add_images_from_directory(temp_image_dir))
    
    assert len(image_paths) == 2  
    for path in image_paths:
        assert os.path.exists(path)




def test_add_images(temp_db, temp_image_dir):
    """Test DatabaseBuilder.add_images"""
    builder = DatabaseBuilder(temp_db)
    builder.create_database()
    builder.add_images(temp_image_dir)
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM image_hashes")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 2


def test_create_32x32_image(test_image):
    """Test DatabaseBuilder.create_32x32_image"""
    builder = DatabaseBuilder("dummy.db")
    result = builder.create_32x32_image(test_image)
    
    assert result is not None
    assert result.shape == (32, 32, 3)


@patch('create_hash_database.DatabaseBuilder.create_database')
@patch('create_hash_database.DatabaseBuilder.add_images')
def test_build_complete_database(mock_add_images, mock_create_db):
    """Test build_complete_database function"""
    build_complete_database("dummy.db", "/test/dir")
    
    mock_create_db.assert_called_once()
    mock_add_images.assert_called_once_with("/test/dir")


#hash.py tests

def test_hamming_distance():
    """Test hamming_distance function"""
    distance = hamming_distance("10101010", "11111111")
    assert distance == 4  
    
    hash1 = "1010101010101010101010101010101010101010101010101010101010101010"
    hash2 = "1010101010101011101010101010101010101010101010101010101010101010"
    distance = hamming_distance(hash1, hash2)
    assert distance == 1  # 1 bit differs


def test_hash_init(temp_db):
    """Test Hash.__init__"""
    hash_obj = Hash(temp_db)
    assert hash_obj.db_path == temp_db


def test_compute_hash(test_image):
    """Test Hash.compute_hash"""
    hash_obj = Hash("dummy.db")
    result = hash_obj.compute_hash(test_image)
    
    assert result is not None
    assert len(result) == 64 
    assert all(c in '01' for c in result)


def test_add_hash_column(db_with_data):
    """Test Hash.add_hash_column"""
    hash_obj = Hash(db_with_data)
    hash_obj.add_hash_column()
    
    conn = sqlite3.connect(db_with_data)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(image_hashes)")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()
    
    assert 'hash' in columns





def test_get_images_without_hash(db_with_data):
    """Test Hash.get_images_without_hash"""
    hash_obj = Hash(db_with_data)
    results = hash_obj.get_images_without_hash()
    
    # Should return image with id=3 (has NULL hash)
    assert len(results) == 1
    assert results[0][0] == 3


def test_update_hash(db_with_data):
    """Test Hash.update_hash"""
    hash_obj = Hash(db_with_data)
    test_hash = "1111000011110000"
    hash_obj.update_hash(3, test_hash)
    
    conn = sqlite3.connect(db_with_data)
    cursor = conn.cursor()
    cursor.execute("SELECT hash FROM image_hashes WHERE id = 3")
    result = cursor.fetchone()
    conn.close()
    
    assert result[0] == test_hash


@patch('hash.Hash.compute_hash')
@patch('hash.Hash.get_images_without_hash')
def test_calc_and_add_hashes(mock_get_images, mock_compute, db_with_data):
    """Test Hash.calc_and_add_hashes"""
    mock_get_images.return_value = [(3, '/path/to/image3.jpg')]
    mock_compute.return_value = "1111000011110000"
    
    hash_obj = Hash(db_with_data)
    with patch.object(hash_obj, 'update_hashes') as mock_update:
        hash_obj.calc_and_add_hashes()
        mock_update.assert_called_once()


def test_update_hashes(db_with_data):
    """Test Hash.update_hashes"""
    hash_obj = Hash(db_with_data)
    hash_updates = [("1111000011110000", 3)]
    hash_obj.update_hashes(hash_updates)
    
    conn = sqlite3.connect(db_with_data)
    cursor = conn.cursor()
    cursor.execute("SELECT hash FROM image_hashes WHERE id = 3")
    result = cursor.fetchone()
    conn.close()
    
    assert result[0] == "1111000011110000"


def test_hash_database_init(temp_db):
    """Test HashDatabase.__init__"""
    hash_db = HashDatabase(temp_db)
    assert hash_db.db_path == temp_db


def test_get_all_hashes(db_with_data):
    """Test HashDatabase.get_all_hashes"""
    hash_db = HashDatabase(db_with_data)
    hashes = hash_db.get_all_hashes()
    
    assert len(hashes) == 3
    assert hashes[1] == '1010101010101010101010101010101010101010101010101010101010101010'
    assert hashes[2] == '1010101010101011101010101010101010101010101010101010101010101010'


def test_find_similar(db_with_data):
    """Test HashDatabase.find_similar"""
    hash_db = HashDatabase(db_with_data)
    query_hash = '1010101010101010101010101010101010101010101010101010101010101010'  
    results = hash_db.find_similar(query_hash, max_distance=2)
    
    # Should find images 1 and 2 (distances 0 and 1)
    assert len(results) >= 1
    assert results[0]['distance'] >= 0  


@patch('hash.Hash.compute_hash')
@patch('hash.HashDatabase.find_similar')
def test_get_similar_images(mock_find_similar, mock_compute_hash, test_image):
    """Test get_similar_images function"""
    mock_compute_hash.return_value = '1010101010101010'
    mock_find_similar.return_value = [{'id': 1, 'distance': 0}]
    
    results = get_similar_images(test_image, "dummy.db")
    
    assert results == [{'id': 1, 'distance': 0}]
    mock_compute_hash.assert_called_once_with(test_image)


@patch('hash.Hash.calc_and_add_hashes')
def test_insert_hashes(mock_calc_and_add):
    """Test insert_hashes function"""
    insert_hashes("dummy.db")
    mock_calc_and_add.assert_called_once()



# ssim.py tests

@patch('ssim.get_similar_images')
def test_get_ssim_single(mock_get_similar, test_image, db_with_data):
    """Test get_ssim_single function"""
    mock_get_similar.return_value = [
        {'id': 1, 'distance': 0},
        {'id': 2, 'distance': 1}
    ]
    
    results = get_ssim_single(test_image, db_with_data)
    
    assert isinstance(results, list)
    mock_get_similar.assert_called_once()


def test_get_ssim_multiple_basic(test_image, db_with_data):
    """Test multiple image SSIM - basic functionality"""
    results = get_ssim_multiple([test_image], db_with_data)
    assert isinstance(results, list)


@patch('ssim.get_ssim_single')
@patch('ssim.get_ssim_multiple')
def test_get_ssim(mock_multiple, mock_single, test_image):
    """Test get_ssim function"""
    # Test single image input
    mock_single.return_value = ['result1', 'result2']
    result = get_ssim(test_image, "dummy.db")
    mock_single.assert_called_once_with(test_image, "dummy.db")
    assert result == ['result1', 'result2']
    
    # Reset mocks and test multiple images input
    mock_single.reset_mock()
    mock_multiple.return_value = ['result3', 'result4']
    result = get_ssim([test_image, test_image], "dummy.db")
    mock_multiple.assert_called_once_with([test_image, test_image], "dummy.db")
    assert result == ['result3', 'result4']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])