#!/bin/bash
# Create output directory if it doesn't exist
mkdir -p Profiling/profiles_html

# Single img color based profiling
.venv/Scripts/python.exe -m cProfile -o Profiling/profiles_html/color_single.prof Profiling/color_single_profiling.py
.venv/Scripts/python.exe Profiling/snakeviz_static_html.py Profiling/profiles_html/color_single.prof

# 2 img color based profiling
.venv/Scripts/python.exe -m cProfile -o Profiling/profiles_html/color_double_unextreme.prof Profiling/color_double_profiling_unextreme_weightened.py
.venv/Scripts/python.exe Profiling/snakeviz_static_html.py Profiling/profiles_html/color_double_unextreme.prof

# 2 img SSIM based profiling
.venv/Scripts/python.exe -m cProfile -o Profiling/profiles_html/ssim_double.prof Profiling/ssim_double_profiling.py
.venv/Scripts/python.exe Profiling/snakeviz_static_html.py Profiling/profiles_html/ssim_double.prof