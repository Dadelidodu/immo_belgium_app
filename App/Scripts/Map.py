import folium
import branca
import geopandas as gpd
from streamlit_folium import folium_static
import streamlit as st
import os


@st.cache_data
def load_geo():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geojson_dataset_path = os.path.join(script_dir, "../data/simplified_output.geojson")
    return gpd.read_file(geojson_dataset_path)


@st.cache_data
def load_map():
    return folium.Map(location=[50.8503, 4.3517], zoom_start=7)


@st.cache_data
def calculate_price_range(_df):
    min_price = _df["mean_price_by_locality"].min()
    max_price = 5000
    return min_price, max_price


def calculate_color_map(min_price, max_price):
    colormap = branca.colormap.LinearColormap(
        ["#202060", "#00ff00"], vmin=min_price, vmax=max_price
    )
    colormap.caption = "Price Range (€)"

    return colormap


def style_function(feature, branca_cmap):
    style_function = {
        "fillColor": branca_cmap(feature["properties"]["mean_price_by_locality"]),
        "weight": 0.1,
        "fillOpacity": 0.8,
    }
    return style_function


@st.cache_data
def tooltip():
    tooltip = folium.GeoJsonTooltip(
        fields=["Zip Code", "mean_price_by_locality", "mun_name_fr"],
        aliases=["Postal Code:", "Price/m² (€):", "Municipality:"],
        localize=True,
    )
    tooltip = tooltip
    return tooltip


def layer(_df, branca_cmap):
    geojson_layer = folium.GeoJson(
        _df,
        style_function=lambda feature: style_function(feature, branca_cmap),
        tooltip=tooltip(),
        name="geojson_layer",
    )
    return geojson_layer


def display_map(_df):
    if "map_initialized" not in st.session_state:
        # Set the map
        map = load_map()
        folium.TileLayer("cartodb dark_matter").add_to(map)

        # Set variables for price range
        min_price, max_price = calculate_price_range(_df)

        # Set the color map for pricing
        branca_cmap = calculate_color_map(min_price, max_price)
        branca_cmap.add_to(map)

        # Set Geojson layer and apply styles
        geojson_layer = layer(_df, branca_cmap)
        geojson_layer.add_child(
            folium.GeoJsonPopup(
                fields=["mun_name_fr", "Zip Code", "mean_price_by_locality"],
                aliases=["Commune", "Post Code", "Median Price/m2 per Commune"]
            )
        )
        geojson_layer.add_to(map)

        # Store the rendered map in the session state
        st.session_state["map"] = map
        st.session_state["map_initialized"] = True

    # Render the cached map
    return folium_static(st.session_state["map"])
