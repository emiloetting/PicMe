import cv2
import numpy as np
import sqlite3
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    

class WaveletHash:
    def __init__(self, db_path: str, image_path: str):
        self.db_path = db_path
        self.image_path = image_path


    def compute_wavelet_hash(self, image_path: str):
        """Berechnet echten optimierten Wavelet-Hash"""
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            

            
            # 32x32 images
            img_32 = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            img_float = np.float32(img_32)
            
            # wavelet transform
            low_freq = cv2.GaussianBlur(img_float, (5, 5), 1.5)
            
            # comprimise to final 8x8
            final_8x8 = cv2.resize(low_freq, (8, 8), interpolation=cv2.INTER_AREA)
            
            # generate hash
            median = np.median(final_8x8)
            hash_array = (final_8x8 > median).astype(int)
            
            return ''.join(map(str, hash_array.flatten()))
            
        except Exception as e:
            print(f"Fehler beim Hashen von {image_path}: {e}")
            return None

    def add_wavelet_column(self):
        """Fügt wavelet_hash Spalte hinzu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("PRAGMA table_info(whole_db)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'wavelet_hash' not in columns:
                cursor.execute('ALTER TABLE whole_db ADD COLUMN wavelet_hash TEXT')
                print("✓ wavelet_hash Spalte hinzugefügt")
            else:
                print("✓ wavelet_hash Spalte existiert bereits")
                
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_wavelet_hash 
                ON whole_db(wavelet_hash)
            ''')
            
            conn.commit()

        except Exception as e:
            print(f"Fehler beim Hinzufügen der Spalte: {e}")

        conn.close()

    def get_images_without_wavelet_hash(self):
        """Holt alle Bilder ohne Wavelet-Hash"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_path FROM whole_db 
            WHERE wavelet_hash IS NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results

    def update_wavelet_hash(self, image_id: int, wavelet_hash: str):
        """Einzelnes Hash-Update"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE whole_db 
            SET wavelet_hash = ? 
            WHERE id = ?
        ''', (wavelet_hash, image_id))
        
        conn.commit()
        conn.close()

    def migrate_all_images_sequential_optimized(self, bulk_size: int = 1000):
        """all wavelet hashes in the database"""

        self.add_wavelet_column()
        images_to_process = self.get_images_without_wavelet_hash()
        
        if not images_to_process:
            print("no images to process")
            return
        
        total_images = len(images_to_process)
        print(f"{total_images:,} images to process")
        

        processed_count = 0
        error_count = 0
        updates_buffer = []
        
        for i, (img_id, image_path) in enumerate(tqdm(images_to_process)):
            
            hash_result = self.compute_wavelet_hash(image_path)
            
            if hash_result:
                updates_buffer.append((hash_result, img_id))
                processed_count += 1
            else:
                error_count += 1

            if len(updates_buffer) >= bulk_size:
                self.bulk_update_wavelet_hashes_fast(updates_buffer)
                updates_buffer = []
                print(f"saved batch!")

        
        if updates_buffer:
            self.bulk_update_wavelet_hashes_fast(updates_buffer)
            print(f"saved final batch!")
        
 

    def bulk_update_wavelet_hashes_fast(self, hash_updates):
        """Schnelles Bulk-Update nach PRIMARY KEY Fix"""
        if not hash_updates:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA journal_mode=WAL")
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.executemany('''
                UPDATE whole_db 
                SET wavelet_hash = ? 
                WHERE id = ?
            ''', hash_updates)
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            print(f" DB error: {e}")
            cursor.execute("ROLLBACK")
            raise
        finally:
            conn.close()
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
        
        cursor.execute('SELECT id, wavelet_hash FROM whole_db')
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
    query_hash = WaveletHash(db_path, image_path).compute_wavelet_hash(image_path=image_path)
    results = db.find_similar(query_hash, max_distance, max_results)
    return results


def get_wavelet_hashes_optimized(db_path: str, bulk_size: int = 10):
    """get wavelet hashes for all images in the database"""
    migrator = WaveletHash(db_path)
    migrator.migrate_all_images_sequential_optimized(bulk_size)


if __name__ == '__main__':
    get_wavelet_hashes_optimized("500k3.db")
