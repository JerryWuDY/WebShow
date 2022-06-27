from CommonConfig import barPy
barPy.Websidebar()

import streamlit as st
# import ee
# import os
# import datetime
# import folium
# import geemap.colormaps as cm
# import shapely.geometry.Polygon

@st.cache
def upload_file_togdf(data):
    import tempfile
    import os
    import uuid

    _,file_extension = os.path.splittext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(),f"{file_id}{file_extension}")

    with open (file_path,"wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gpf = gpd.read_file(file_path, driver = "KML")
    else:
        gdf = gdp.read_file(file_path)

    return gdf



def app():

    today = date.today()

    st.title("Create Satellite Timelapse")

    st.markdown(
        """
        An interactive web app for creating [Landsat](https://developers.google.com/earth-engine/datasets/catalog/landsat)/[GOES](https://jstnbraaten.medium.com/goes-in-earth-engine-53fbc8783c16) timelapse for any location around the globe. 
        The app was built using [streamlit](https://streamlit.io), [geemap](https://geemap.org), and [Google Earth Engine](https://earthengine.google.com). For more info, check out my streamlit [blog post](https://blog.streamlit.io/creating-satellite-timelapse-with-streamlit-and-earth-engine). 
    """
    )

    row1_col1, row1_col2 = st.columns([2,1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    st.session_state["ee_asset_id"] = None
    st.session_state["bands"] = None
    st.session_state["palette"] = None
    st.session_state["vis_params"] = None

    with row1_col1:
        m = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=True,
            Draw_export=True,
            locate_control=True,
            plugin_LatLngPopup=False,
        )
        m.add_basemap("ROADMAP")

    with row1_col2:

        keyword = st.text_input("Search for a location:", "")
        if keyword:
            locations = geemap.geocode(keyword)
            if locations is not None and len(locations) > 0:
                str_locations = [str(g)[1:-1] for g in locations]
                location = st.selectbox("Select a location:", str_locations)
                loc_index = str_locations.index(location)
                selected_loc = locations[loc_index]
                lat, lng = selected_loc.lat, selected_loc.lng
                folium.Marker(location=[lat, lng], popup=location).add_to(m)
                m.set_center(lng, lat, 12)
                st.session_state["zoom_level"] = 12

        collections = st.selectbox(
            "Select a saterllite image collection: ",
            [
                "Landsat TM-ETM-OLI Surface Reflectance",
                "Sentinel-2 MSI Surface Reflectance",
            ],
            index=1,
        )

        if collection in [
            "Landsat TM-ETM-OLI Surface Reflectance",
            "Sentinel-2 MSI Surface Reflectance",
        ]:
            roi_options = ["Uploaded GeoJSON"] + list(landsat_rois,keys())
        else:
            roi_options = ["Uploaded GeoJSON"]

        if collection =="Any Earth Engine ImageCollection":
            keyword = st.text_input("Enter a keyword to search (e.g., MODIS):","")
            if keyword:
                assets = geemap.search




# Just for meeting
with st.expander("markdown"):
    st.markdown(
        "Here is the *markdown* WEB"
        "<https://markdown.com.cn/>"
    )


st.markdown(
    "## GEE学习汇报（2022/6/24-） "
    "<https://streamlit.geemap.org/>"
)
with st.expander("GEE"):
    imgGEE = "https://github.com/JerryWuDY/WebShow/raw/main/src/GEE1.png"
    st.image(imgGEE)

st.
st.markdown(
    "## Paper deploy section） "
)
with st.expander("Paper"):
    imgUrban = "https://github.com/JerryWuDY/WebShow/raw/main/src/Urban.png"
    st.image(imgUrban)