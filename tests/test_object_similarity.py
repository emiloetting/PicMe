import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
from annoy import AnnoyIndex
from unittest.mock import patch, MagicMock
import torch
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
object_dir = os.path.join(parent_dir, 'ObjectSimilarity')

# Add both directories to Python path
sys.path.insert(0, parent_dir)
sys.path.insert(0, object_dir)

try:
    from create_embeddings import image_embeddings_with_paths, create_ann
    from similar_image import get_image_embedding, get_best_images
except ImportError as e:

    from ObjectSimilarity.create_embeddings import image_embeddings_with_paths, create_ann
    from ObjectSimilarity.similar_image import get_image_embedding, get_best_images


@pytest.fixture
def test_dir():
    """Create a temporary directory with test images"""
    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir()
    
    formats = [("test1.jpg", "JPEG"), ("test2.png", "PNG"), ("test3.bmp", "BMP")]
    created_files = []
    
    for filename, format_type in formats:
        img = Image.new('RGB', (50, 50), color='red' if 'test1' in filename else 'blue')
        file_path = images_dir / filename
        img.save(file_path, format_type)
        created_files.append(file_path)
    
    # non-image file
    txt_file = images_dir / "not_an_image.txt"
    txt_file.write_text("This is not an image")
    created_files.append(txt_file)
    
    yield temp_dir
    

    # remove individual files first
    for file_path in created_files:
        try:
            if file_path.exists():
                file_path.unlink()
        except PermissionError:
            pass  
    
    # remove directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass
    
    # Cleanup test files
    for file in ["500k.ann", "test_pictures_paths.json"]:
        try:
            if Path(file).exists():
                Path(file).unlink()
        except PermissionError:
            pass


@patch('create_embeddings.clip.load')
def test_image_embeddings_with_paths(mock_clip_load, test_dir):
    """Test image_embeddings_with_paths function"""
    # Setup mock CLIP model
    mock_model = MagicMock()
    mock_preprocess = MagicMock()
    mock_clip_load.return_value = (mock_model, mock_preprocess)
    
    mock_preprocess.return_value = MagicMock()
    mock_model.encode_image.return_value = torch.tensor([[1.0] * 512])
    
    images_dir = Path(test_dir) / "images"
    
    results = list(image_embeddings_with_paths(str(images_dir)))
    
    assert len(results) >= 3  
    
    if len(results) > 0:
        for path, embedding in results:
            assert isinstance(path, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (512,)
            assert Path(path).exists()
            assert any(ext in path.lower() for ext in ['.jpg', '.png', '.bmp'])


@patch('create_embeddings.image_embeddings_with_paths')
@patch('create_embeddings.AnnoyIndex')
def test_create_ann(mock_annoy_index, mock_image_embeddings, test_dir):
    """Test create_ann function"""
    # Setup mock data
    mock_embeddings = [
        ("/path/to/image1.jpg", np.array([1.0] * 512)),
        ("/path/to/image2.jpg", np.array([2.0] * 512)),
        ("/path/to/image3.jpg", np.array([3.0] * 512))]
    mock_image_embeddings.return_value = mock_embeddings
    
    # Setup mock AnnoyIndex
    mock_index = MagicMock()
    mock_annoy_index.return_value = mock_index
    
    images_dir = Path(test_dir) / "images"
    result = create_ann(str(images_dir))
    
    mock_annoy_index.assert_called_once_with(512, 'angular')
    assert mock_index.add_item.call_count == 3
    mock_index.build.assert_called_once_with(20)
    mock_index.save.assert_called_once_with("500k.ann")
    
    assert isinstance(result, dict)
    assert len(result) == 3
    
    for i in range(3):
        assert i in result
        assert result[i] == f"/path/to/image{i+1}.jpg"


@patch('similar_image.clip.load')
def test_get_image_embedding(mock_clip_load, test_dir):
    """Test get_image_embedding function"""
    # Setup mock CLIP model
    mock_model = MagicMock()
    mock_preprocess = MagicMock()
    mock_clip_load.return_value = (mock_model, mock_preprocess)
    
    expected_embedding = np.array([1.0] * 512)
    mock_preprocess.return_value = MagicMock()
    mock_model.encode_image.return_value = torch.tensor([expected_embedding])
    
    # Create a test image 

    test_image_dir = Path(test_dir) / "test_images"
    test_image_dir.mkdir()
    
    test_image_path = test_image_dir / "test_image.jpg"
    img = Image.new('RGB', (50, 50), color='green')
    img.save(test_image_path, "JPEG")
    

    result = get_image_embedding(str(test_image_path))
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (512,)
    mock_clip_load.assert_called_once_with("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with invalid path
    result_invalid = get_image_embedding("/nonexistent/path.jpg")
    assert result_invalid is None


@patch('similar_image.get_image_embedding')
@patch('similar_image.AnnoyIndex')
def test_get_best_images_single_input(mock_annoy_index, mock_get_embedding, test_dir):
    """Test get_best_images function with single input image"""
    # Setup mock data
    mock_embedding = np.array([1.0] * 512)
    mock_get_embedding.return_value = mock_embedding
    
    mock_index = MagicMock()
    mock_index.get_nns_by_vector.return_value = [0, 1, 2]
    mock_annoy_index.return_value = mock_index
    
    # Create test JSON mapping
    test_mapping = {
        "0": "/path/to/image1.jpg",
        "1": "/path/to/image2.jpg", 
        "2": "/path/to/image3.jpg"}
    json_path = Path(test_dir) / "test_mapping.json"

    with open(json_path, 'w') as f:
        json.dump(test_mapping, f)
    
    ann_path = Path(test_dir) / "test_index.ann"
    ann_path.touch()
    
    result = get_best_images(
        input_images="/path/to/query.jpg",
        index_to_path_json=str(json_path),
        annfile=str(ann_path),
        num_results=3
    )
    
    assert len(result) == 3
    assert result == ["/path/to/image1.jpg", "/path/to/image2.jpg", "/path/to/image3.jpg"]
    mock_annoy_index.assert_called_once_with(512, 'angular')
    mock_index.load.assert_called_once_with(str(ann_path))
    mock_index.get_nns_by_vector.assert_called_once_with(mock_embedding, 3)


@patch('similar_image.get_image_embedding')
@patch('similar_image.AnnoyIndex')
def test_get_best_images_multiple_inputs(mock_annoy_index, mock_get_embedding, test_dir):
    """Test get_best_images function with multiple input images"""
    # Setup mock data
    mock_embeddings = [np.array([1.0] * 512), np.array([2.0] * 512)]
    mock_get_embedding.side_effect = mock_embeddings
    
    mock_index = MagicMock()
    mock_index.get_nns_by_vector.return_value = [0, 1]
    mock_annoy_index.return_value = mock_index
    
    # Create test JSON mapping
    test_mapping = {"0": "/path/to/result1.jpg", "1": "/path/to/result2.jpg"}

    json_path = Path(test_dir) / "test_mapping.json"
    with open(json_path, 'w') as f:
        json.dump(test_mapping, f)
    
    ann_path = Path(test_dir) / "test_index.ann"
    ann_path.touch()
    
    input_images = ["/path/to/query1.jpg", "/path/to/query2.jpg"]
    result = get_best_images(
        input_images=input_images,
        index_to_path_json=str(json_path),
        annfile=str(ann_path),
        num_results=2
    )
    


    assert len(result) == 2
    assert result == ["/path/to/result1.jpg", "/path/to/result2.jpg"]
    assert mock_get_embedding.call_count == 2
    
    # Verify that the mean embedding was used
    called_args = mock_index.get_nns_by_vector.call_args[0]
    expected_mean = np.mean(mock_embeddings, axis=0)
    np.testing.assert_array_equal(called_args[0], expected_mean)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])