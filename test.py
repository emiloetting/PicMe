import sqlite3

conn = sqlite3.connect('500k.db')
cursor = conn.cursor()

image_path = r"E:\data\image_data\500k\pixabay_dataset_v1\images_07\3d-model-world-earth-geography-2895712.jpg"
cursor.execute("SELECT phash FROM whole_db WHERE image_path = ?", (image_path,))
spaltennamen = cursor.fetchall()
print(spaltennamen)
image_path2 = r"C:\Users\joche\Documents\BigData\Repo\PicMe\SSIM\Testbilder\3d-model-world-earth-geography-2894348.jpg"
cursor.execute("SELECT phash FROM whole_db WHERE image_path = ?", (image_path2,))
# Dann Spaltennamen aus description holen
spaltennamen = cursor.fetchall()
print(spaltennamen)
conn.commit()
conn.close()

