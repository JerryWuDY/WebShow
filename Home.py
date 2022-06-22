import streamlit as st

from CommonConfig import barPy
barPy.Websidebar()

# Part:Sidebar. You can add your information here ღ( ´･ᴗ･` )
st.sidebar.title("About")
st.sidebar.info(
    """
    Professor: <https://civil.sysu.edu.cn/teacher/477>
    Student  : Please call me Mr.handsome
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Prof.Zhao: <https://civil.sysu.edu.cn/teacher/477>
    [Email](https://civil.sysu.edu.cn/teacher/477) | [Tel](https://civil.sysu.edu.cn/teacher/477)
    """
)

st.sidebar.title("For More")
st.sidebar.info(
    '''How dare you?'''
)

# Here is the main body of your website.
st.title("Streamlit for Geospatial Applications")

# Copy the "https://streamlit.geemap.org/"
st.markdown(
    """
    This multi-pages web app demonstrates various interactive web apps created using [streamlit](https://streamlit.io) and open-source mapping libraries,
    such as [leafmap](https://leafmap.org), [geemap](https://geemap.org), [pydeck](https://deckgl.readthedocs.io), and [kepler.gl](https://docs.kepler.gl/docs/keplergl-jupyter).
    This is an open-source project and you are very welcome to contribute your comments, questions, resources, and apps as [issues](https://github.com/giswqs/streamlit-geospatial/issues) or
    [pull requests](https://github.com/giswqs/streamlit-geospatial/pulls) to the [GitHub repository](https://github.com/giswqs/streamlit-geospatial).

    """
)

st.info("Click on the left sidebar menu to navigate to the different apps.")

st.subheader("*Timelapse* of Satellite Imagery")
st.markdown(
    """
    The following timelapse animations were created using the **Timelapse web app**. Click `Timelapse` on the left sidebar menu to create your own timelapse for any location around the globe.
"""
)

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/spain.gif")
    st.image("https://github.com/giswqs/data/raw/main/timelapse/las_vegas.gif")

with row1_col2:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/goes.gif")
    st.image("https://github.com/giswqs/data/raw/main/timelapse/fire.gif")
