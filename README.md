# Welcome to PicMe - The **P**ixel-**I**nformed **C**ontent-**M**atching **E**ngine

<p align="center">
<img width="500" height="500" alt="428210645-5ba1cbd0-10fb-4d38-89df-30beb048400d" src="https://github.com/user-attachments/assets/9aed75fc-cd97-4515-ab09-d10ac2b7b79d" />
</p>

**PicMe** is a tool to be used for finding the most similar images on a 500k+ image database to a single or double image input within just a few secs :stopwatch:  
The image recommender allows for three different metrics to be used as basis for the similarity search: 
- :bar_chart: the images' **color distribution**
- :paperclip: their 256-dimensional **embeddings** created by :link: [@OpenAI](https://github.com/OPENAI) 's ViT-B/32 CLIP model
- :triangular_ruler: each image's **structural similarity index measure** (SSIM)

<br>
<br>

# Preview :tv:
Here a taste of what to expect :clinking_glasses:  

<p align="center">
<img width="330" height="437" alt="weighted_color" src="https://github.com/user-attachments/assets/da2731a6-9a4f-4af3-9eb7-0cfc4839fe83" />
<img width="330" height="440" alt="object" src="https://github.com/user-attachments/assets/61a34852-012d-454a-b550-f4348b275c39" />
<img width="330" height="442" alt="ssim" src="https://github.com/user-attachments/assets/d991a8fd-d489-4a17-ba28-3bda35861304" />
</p>

<br>
<br>

# Repo structure :atom: 
```
PicMe/
├── .github/
│   └── workflows/
│       └── CICD_Linting_Testing.yaml
├── .gitignore
├── Analysis/
│   ├── cluster_plot.html
│   ├── clustering.py
│   ├── dim_reduction.py
│   └── plot.py
├── ColorSimilarity/
│   ├── __init__.py
│   ├── colorClusterquantized.py
│   └── main_helper.py
├── DataBase/
│   └── __init__.py
├── GUI_data/
│   ├── loading.gif
│   └── PicMe_logo_cleaned.png
├── GUI.py
├── Initialization/
│   ├── __init__.py
│   └── setup.py
├── LICENSE
├── ObjectSimilarity/
│   ├── __init__.py
│   ├── create_embeddings.py
│   └── similar_image.py
├── Profiling/
│   ├── clip_double_profiling.py
│   ├── clip_single_profiling.py
│   ├── color_double_profiling_unextreme_weightened.py
│   ├── color_single_profiling.py
│   ├── run_profilings.sh
│   ├── snakeviz_static_html.py
│   ├── ssim_double_profiling.py
│   ├── ssim_single_profiling.py
│   ├── test_img_1.jpg
│   └── test_img_2.jpg
├── README.md
├── requirements.txt
├── SSIM/
│   ├── __init__.py
│   ├── create_hash_database.py
│   ├── hash.py
│   └── ssim.py
└── tests/
    ├── test_color_similarity.py
    ├── test_object_similarity.py
    └── test_ssim.py
```
<br>
<br>

# QUICKSTART Guide :rocket:

### :one: Collecting dependancies :package:
In order to set up the workspace, it is highly recommend to install all depencies into your environment.  
Use the `requirements.txt` provided within the repo to collect dependancies quickly. 
```python
pip install -r requirements.txt
```

### 2️⃣⚠️  File access required for out-of-the-box usage :floppy_disk: :warning:
PicMe's image-recommendation system is based upon pre-computed image embeddings stored across several relational databases.    
The software will return the repsective windows-style image paths of the top 12 matches.   
These filepaths are meant to be opened using a specifically formatted hard drive :floppy_disk:  
In order to be able to use the **PicMe** image recommender at its full potential, it is easiest to have access to said hard drive or its folder structure. 
### HOWEVER, YOU CAN BUILD YOUR OWN DATABASE AS WELL. KEEP IN MIND THAT IT MIGHT TAKE SOME TIME BASED UPON IMAGE SIZES AND AMOUNT. SEE END OF READ ME FOR INSTRUCTION ON HOW TO BUILD YOUR OWN DATA BACKEND ⬇️
#### :bangbang: :open_file_folder: If you are in posession of the images & store them in the required directory-tree, the software is likely to work  :palm_tree: :bangbang:
:plunger: **Should you still experience path issues, you can adjust the returned image paths in line 163 `GUI.py` file :memo: within method `run` of class `FindWorker` by *adding* something like this:**
<p align="center">
<img width="1804" height="208" alt="image" src="https://github.com/user-attachments/assets/b6b8279a-cfc6-4f99-8d00-bbaeb93f0809" />
</p>
Of course, adjustments must be made according to your hard drive and machine setup..  

### 3️⃣ Collecting Database and [ANNOY](https://github.com/spotify/annoy)-Index objects:   
Since the image-recommendation is based upon feature-vector calculations, 6 files and databases in total for handling index-IDs, filepaths and embeddings are required in order to make use of the code as is.  
📥 In order to collect these files, the `setup.py` script (to be found within module `Initialization`) is to be executed. It will collect all files from a Google-Drive storage and store them within directory `DataBase`.  
Collected files are:  
- `clip_embeddings_paths.json`
- `clip_embeddings.ann`
- `color_ann_index.ann`
- `color_database.db`
- `emd_cost_full.npy`
- `hash_database.db`

### 4️⃣ Starting the application
The application as depicted in preview above can be started by executing the `GUI.py` skript :scroll:

<br>
<br>

# 🕹️ Handling the GUI
The GUI-design was intended to be understandable quite intuitively.  
However, just in case, here's a quick descprition of the GUI's key features:
<br>
<br>

<div align="center" style="display: flex; gap: 10px;">
<img height="900" alt="image" src="https://github.com/user-attachments/assets/1ad8c470-f14c-41d3-b120-7c7d6b74d8ac" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/88259dd6-c85b-4456-815b-ebb961e5cc27" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/a5986e40-4db4-4d42-8694-1e083fe000d0" />
</div>

<br>
<br>

- **Load an image** by *Drag'n'Dropping* the image file within the dotted rectangle   
- click :heavy_plus_sign: to add another rectangle to drop a second input image into   
- click :heavy_minus_sign: to remove the second loaded image after dropping to get back to single input match-finding  
- **Update** the input image by simply dragging the next image into the rectangle(s)  
- **Select** a metric / mode to be used for finding your input's best matches  
- :level_slider: For **color-based** 2️⃣-input mode: adjust sliders to dial in respective weights color profiles for input images  
- START match-findind code by clicking   

<br>
<br>

# :hammer_and_wrench: Build your own backend (incl. databases, mapping-.json-file & [ANNOY](https://github.com/spotify/annoy)-indices :gear:
### :exclamation: REMEMBER: THIS MIGHT TAKE A LOT OF TIME DEPENDING ON AVAILABLE COMPUTATIONAL POWER, AMOUNT OF IMAGES AND IMAGE SIZES :exclamation:


# :gift: BONUS: Overview of image-distribution in vector space based upon implemented CLIP-embeddings :) :magic_wand:
<p align="center">
<img width="1533" height="1073" alt="image" src="https://github.com/user-attachments/assets/35781ca1-97e7-4812-a0b9-27e32e053a55" />
</p>



