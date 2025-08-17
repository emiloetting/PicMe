import cv2
import sqlite3
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List
import datetime
import os

cwd = os.getcwd()
path_to_database_folder = os.path.join(cwd, "DataBase")


class DatabaseBuilder:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def create_database(self):
        """ Creates new database with image_hashes table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                hash TEXT,
                created_at TEXT,
                image_32x32 BLOB
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_images_from_directory(self, image_directory: str, extensions: List[str] = None):
        """
        Adds all images from directory to database
        
        Args:
            image_directory: path to directory
            extensions: list of image extensions

        Yields:
            str: image path
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        folder = Path(image_directory)
        

        for image_path in tqdm(folder.rglob("*.*")):
            if image_path.suffix.lower() in extensions:
                try:
                    yield str(image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    def add_images(self, image_directory: str, batch_size: int = 1000):
        """
        Adds images with paths and 32x32 versions
        
        Args:
            image_paths: list of image paths
            batch_size: number of images to add at once

        Returns:
            None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA journal_mode=WAL")
        
        processed = 0
        errors = 0
        batch_data = []
        
        try:
            for image_path in tqdm(self.add_images_from_directory(image_directory), desc="Processing images"):
                try:
                    # Create 32x32 version
                    image_32x32 = self.create_32x32_image(str(image_path))
                    if image_32x32 is None:
                        errors += 1
                        continue
                    
                    # Serialize as blob
                    image_blob = pickle.dumps(image_32x32)
                    created_at = datetime.datetime.now().isoformat()
                    
                    batch_data.append((str(image_path), None, created_at, image_blob))
                    processed += 1
                    
                    # Check if batch is full then update db
                    if len(batch_data) >= batch_size:
                        cursor.executemany('''
                            INSERT OR IGNORE INTO image_hashes 
                            (image_path, hash, created_at, image_32x32) 
                            VALUES (?, ?, ?, ?)
                        ''', batch_data)
                        batch_data = []
                        conn.commit()
                        
                except Exception as e:
                    print(f"Error with {image_path}: {e}")
                    errors += 1
                    continue
            
            if batch_data:
                cursor.executemany('''
                    INSERT OR IGNORE INTO image_hashes 
                    (image_path, hash, created_at, image_32x32) 
                    VALUES (?, ?, ?, ?)
                ''', batch_data)
                conn.commit()
            
            print(f"added images success: {processed}, errors: {errors}")
            
        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            conn.close()

    def create_32x32_image(self, image_path: str):
        """Creates 32x32 BGR image for SSIM"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            # Create 32x32 version
            img_32x32 = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            img_32x32 = cv2.cvtColor(img_32x32, cv2.COLOR_GRAY2BGR)
            
            return img_32x32
            
        except Exception:
            print(f"Error creating 32x32 image for {image_path}")
            return None



def build_complete_database(db_path: str, image_directory: str):
    """Builds complete database using existing Hash classes"""
    builder = DatabaseBuilder(db_path)

    builder.create_database()
    
    #Add all images with 32x32 versions
    builder.add_images(image_directory)

if __name__ == "__main__":
    build_complete_database(f"{path_to_database_folder}/new_databse.db", "image_directory_path")

