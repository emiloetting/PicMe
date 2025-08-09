import cv2
import numpy as np
from PIL import Image
import sqlite3
from typing import List, Tuple, Dict
from pathlib import Path
from scipy.spatial.distance import hamming
from tqdm import tqdm
import time


class PerceptualHash:
    def __init__(self, image_path: str):
        self.image_path = image_path
        # self.db_path = db_path

    def compute_hash(self):
        """compute the hash of image"""

        # convert the image to grayscale, resize
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        img_small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        img_float = np.float32(img_small)

        # perform dct
        dct = cv2.dct(img_float)
        dct = dct[0:8, 0:8]
        dct = dct.flatten()

        # compute hash
        median = np.median(dct)
        hash_array = (dct > median).astype(int)
       
        hash_str = ''.join(map(str, hash_array))
        return hash_str
  
    def hamming_distance( hash1, hash2):
        """calculates the hamming distance of two hashes"""
        int1 = int(hash1, 2)
        int2 = int(hash2, 2)
        xor_result = int1 ^ int2
        return bin(xor_result).count('1')
    
    

class HashDatabase:
    """build a hash database and search for similar images"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """initialize the database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                phash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # create index on phash
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_phash ON image_hashes(phash)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_image_hash(self, image_path: str, phash: str):
        """add an image hash to the database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO image_hashes (image_path, phash)
            VALUES (?, ?)
        ''', (image_path, phash))
        
        conn.commit()
        conn.close()
    
    def get_all_hashes(self) -> Dict[str, str]:
        """get all image hashes from the database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, phash FROM whole_db')
        results = cursor.fetchall()
        conn.close()
        
        hash_dict = {}
        for id, phash in results:
            hash_dict[id] = phash
        return hash_dict
    
    def find_similar(self, query_hash: str, max_distance: int = 15, 
                     max_results: int = 1000) -> List[Dict]:
        """
        get similar images
        
        Args:
            query_hash: input hash
            max_distance: max hamming distance
            max_results: max number of results
            
        Returns:
            list of similar images
        """

        all_hashes = self.get_all_hashes()
        results = []
        
        for id, stored_hash in all_hashes.items():
            distance = PerceptualHash.hamming_distance(hash1 = query_hash, hash2 = stored_hash)
            
            if distance <= max_distance:
                results.append({'id': id, 'distance': distance})
        
        # sort by distance
        results.sort(key=lambda x: x['distance'])
        
        return results[:max_results]
    


class ImageHashProcessor:
    """process images and add their hashes to the database"""

    def __init__(self, db_path: str):
        self.db = HashDatabase(db_path)

    def image_generator(self, directory_path: str, image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """generate all images in a directory"""

        directory = Path(directory_path)

        for ext in image_extensions:
            yield from directory.rglob(f'*{ext}')
            yield from directory.rglob(f'*{ext.upper()}')

    def process_directory(self, directory_path: str):
        """process all images in a directory and add their hashes to the database"""
        
        processed = 0
        errors = 0

        for image_path in tqdm(self.image_generator(directory_path)):

            try:
                phash = PerceptualHash(image_path).compute_hash()
                if phash:
                    self.db.add_image_hash(str(image_path), phash)
                    processed += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                errors += 1

        print(f"Processed {processed} images. {errors} errors.")
        



def process_images(directory_path: str, db_path: str):
    """process all images in a directory and add their hashes to the database"""

    processor = ImageHashProcessor(db_path)
    processor.process_directory(directory_path)
    

def get_similar_images(image_path: str, db_path: str, max_distance: int = 30, max_results: int = 1000):
    """get similar images to a given image"""

    db = HashDatabase(db_path)
    query_hash = PerceptualHash(image_path).compute_hash()
    results = db.find_similar(query_hash, max_distance, max_results)
    return results




if __name__ == '__main__':
    #process_images(directory_path=, db_path=)
    pass