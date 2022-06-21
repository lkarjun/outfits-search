import streamlit as st
import streamlit.components.v1 as components
from app_helper import *
from pathlib import Path
import os

os.environ["AZURE_STORAGE_ACCOUNT"] = st.secrets["AZURE_STORAGE_ACCOUNT"]
os.environ["AZURE_STORAGE_KEY"] = st.secrets["AZURE_STORAGE_KEY"]

SAMPLE_IMG = Path("SampleImages")
SAMPLE_FILES = range(1, 8)

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
        img = Image.open(SAMPLE_IMG/f'{random.choice(SAMPLE_FILES)}.jpg')
        st.image(img, caption="Sample")


st.write()

with st.spinner("Loading Results..."):
    st.markdown(f"<br><h3 style='text-align: left;'>Buy Similar Products 🛍️</h3><br>", True)
    rslt_idx = get_result(img)
    rslt = result_html(rslt_idx)
    components.html(rslt, height=8000)