import cv2
import numpy as np
import sqlite3
from typing import List, Dict
from tqdm import tqdm


def hamming_distance(hash1, hash2):
    """
    calculates the hamming distance of two hashes

    Args:
        hash1: first hash
        hash2: second hash

    Returns:
        hamming distance
    """
    try:
        int1 = int(hash1, 2)
        int2 = int(hash2, 2)
        xor_result = int1 ^ int2
        return bin(xor_result).count('1')
    except Exception as e:
        print(f"error calculating hamming distance: {e}")
        return float('inf')
    

class Hash:
    def __init__(self, db_path: str):
        self.db_path = db_path



    def compute_hash(self, image_path: str):
        """
        calculates the hash of an image
        loads the image as 32x32 gray-scale
        transforms to wavelet and compresses to 8x8
        generates the hash by comparing to median

        Args:
            image_path: path to the image

        Returns:
            hash
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise Exception(f"could not load image {image_path}")
            
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
            
            hash_string =''.join(map(str, hash_array.flatten()))
            return hash_string
            
        except Exception as e:
            print(f"error creating hash for {image_path}: {e}")
            return None

    def add_hash_column(self):
        """
        adds hash column to the database

        Returns:
            None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("PRAGMA table_info (image_hashes)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'hash' not in columns:
                cursor.execute('ALTER TABLE image_hashes ADD COLUMN hash TEXT')
                print("added hash column")
            else:
                print("hash column already exists")
                
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_hash 
                ON image_hashes(hash)
            ''')
            
            conn.commit()

        except Exception as e:
            print(f"error adding hash column: {e}")

        conn.close()

    def get_images_without_hash(self):
        """
        gets all images from the database without hash

        Returns:
            list of tuples (id, image_path)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_path FROM image_hashes 
            WHERE hash IS NULL
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results

    def update_hash(self, image_id: int, hash: str):
        """
        updates the hash of an image
        
        Args:
            image_id: id of the image
            hash: hash of the image

        Returns:
            None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE image_hashes 
            SET hash = ? 
            WHERE id = ?
        ''', (hash, image_id))
        
        conn.commit()
        conn.close()

    def calc_and_add_hashes(self, bulk_size: int = 1000):
        """
        calculates and adds hashes for all images to the database

        Args:
            bulk_size: number of images to process in one batch

        Returns:
            None
        """

        self.add_hash_column()
        images_to_process = self.get_images_without_hash()
        
        if not images_to_process:
            print("no images to process")
            return
        
        total_images = len(images_to_process)
        print(f"{total_images:,} images to process")
        

        processed_count = 0
        error_count = 0
        updates_buffer = []
        
        # calc hash for all images, save in bulk, update db
        for i, (img_id, image_path) in enumerate(tqdm(images_to_process)):
            
            hash_result = self.compute_hash(image_path)
            
            if hash_result:
                updates_buffer.append((hash_result, img_id))
                processed_count += 1
            else:
                error_count += 1

            if len(updates_buffer) >= bulk_size:
                self.update_hashes(updates_buffer)
                updates_buffer = []
                print(f"saved batch!")

        
        if updates_buffer:
            self.update_hashes(updates_buffer)
            print(f"saved final batch!")
        
 

    def update_hashes(self, hash_updates):
        """
        updates hashes in bulk in db

        Args:
            hash_updates: list of tuples (hash, image_id)

        Returns:
            None
        """
        if not hash_updates:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA journal_mode=WAL")
        
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            cursor.executemany('''
                UPDATE image_hashes 
                SET hash = ? 
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
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_all_hashes(self):
        """
        get all image hashes from the database

        Returns:
            dictionary with image ids as keys and hashes as values
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, hash FROM image_hashes')
        results = cursor.fetchall()
        conn.close()
        
        hash_dict = {}
        for id, hash in results:
            hash_dict[id] = hash

        return hash_dict
    
    def find_similar(self, query_hash: str, max_distance: int = 15, 
                     max_results: int = 1000) -> List[Dict]:
        """
        get similar imagesof a given image based on hamming distance to hashes
        
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
            distance = hamming_distance(hash1 = query_hash, hash2 = stored_hash)
            
            if distance <= max_distance:
                results.append({'id': id, 'distance': distance})
        
        # sort by distance
        results.sort(key=lambda x: x['distance'])
        
        return results[:max_results]    
    

def get_similar_images(image_path: str, db_path: str, max_distance: int = 30, max_results: int = 1000):
    """
    compute hash for input image and get similar images to the given image

    Args:
        image_path: path to input image
        db_path: path to database
        max_distance: max hamming distance
        max_results: max number of results

    Returns:
        list of similar images containing id and distance
    """

    db = HashDatabase(db_path)
    query_hash = Hash(db_path).compute_hash(image_path)
    results = db.find_similar(query_hash, max_distance, max_results)
    return results


def insert_hashes (db_path):
    """
    adds hashes to the database

    Args:
        db_path: path to database

    Returns:
        None
    """

    Hash(db_path).calc_and_add_hashes()
    print("hashes added")

