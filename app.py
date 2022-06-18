import streamlit as st
from app_helper import get_result, Image, RImgModel, model_splitter
from pathlib import Path
import os

os.environ["AZURE_STORAGE_ACCOUNT"] = st.secrets["AZURE_STORAGE_ACCOUNT"]
os.environ["AZURE_STORAGE_KEY"] = st.secrets["AZURE_STORAGE_KEY"]


DATASET_FOLDER = Path("Dataset")
SAMPLE_IMG = DATASET_FOLDER/'upperwear/jacket/upperwear_jacket6771.png'

if not DATASET_FOLDER.exists():
    os.system('dvc pull Dataset.dvc')

st.set_page_config(page_title="Outfits Search 🛍️👗👠", page_icon="🛒", 
                    menu_items={'About': "**Outfits Search🛍️👗👠** Prediction App"})


st.markdown(f"<h1 style='text-align: Center;'>Outfits Search🛍️👗👠</h1><br>", True)



file = st.file_uploader("Search Image", type=["png", "jpg", "jpeg"])
st.markdown("<br>", True)
_, col2, _  = st.columns(3)

with col2:
    if file:
        img = Image.open(file)
        st.image(img)
    else:
        img = Image.open(SAMPLE_IMG)
        st.image(img, caption="Jacket")


st.write()

with st.spinner("Loading Results..."):
    st.markdown(f"<br><h3 style='text-align: left;'>Similar Products 🛍️</h3><br>", True)
    images = get_result(img)
    st.image(images, width=110)