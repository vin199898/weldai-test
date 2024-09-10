import streamlit as st
import pandas as pd
import numpy as np



# --- PAGE SETUP ---

home_page = st.Page(
    "views/Homepage.py",
    title="Homepage",
    icon=":material/home:",
    default=True,
    
)
    
project_0_page = st.Page(
    "views/ParametersOptimization.py",
    title="Parameters Optimization Tool",
    icon=":material/manufacturing:",
    
    

)

project_1_page = st.Page(
    "views/WPSLibrary.py",
    title="WPS Library",
    icon=":material/book_2:",
    
    
)

project_2_page = st.Page(
    "views/WeldInspection.py",
    title="Weld Inspection Tool ",
    icon=":material/frame_inspect:",
    
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---


pg = st.navigation(
    
    {
        "Main": [home_page],  
        
        "WELD PARAMETERS OPTIMIZATION ": [project_0_page, project_1_page],
        
        "WELD DEFECT DETECTION": [project_2_page],
    }
    
)


st.logo("assets/logo.png")


st.sidebar.text("Build by WeldAi Team")

pg.run()