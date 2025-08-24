# -*- coding: utf-8 -*-
"""Food_Planner_App_Using_Agno_Agents_OpenRouter_StreamLit.py"""

import streamlit as st
import os
from serpapi import GoogleSearch
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.openrouter import OpenRouter
from datetime import datetime

# --- Streamlit Page Config ---
st.set_page_config(page_title="🍲 AI Food Planner for Travellers", layout="wide")

st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #d35400;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">🍲 AI Food Planner for Travellers</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get personalized restaurant & meal recommendations while travelling</p>', unsafe_allow_html=True)

# --- User Inputs ---
st.markdown("### 📍 Where are you?")
location = st.text_input("Enter your current city or location:", "Delhi, India")

st.markdown("### 🍴 Food Preferences")
meal_time = st.radio("When do you want to eat?", ["Morning", "Afternoon", "Night"])
diet = st.radio("Diet Preference:", ["Veg", "Non-Veg", "Both"])
liked_food = st.text_input("🍜 A food item you like:", "Biryani")
budget = st.selectbox("💰 Budget Range:", ["Low", "Medium", "High"])

# --- Sidebar ---
st.sidebar.title("⚙️ Food Planner Settings")
radius = st.sidebar.slider("Search Radius (km):", 1, 20, 5)
min_rating = st.sidebar.slider("Minimum Restaurant Rating:", 1.0, 5.0, 4.0, 0.1)

# --- API Keys ---
SERPAPI_KEY = "fd2bb54f91a2fb66f868214f58e3e95eba46d9436fba04d7862292625bfc83a1"  # replace with your valid SerpAPI key
os.environ["OPENROUTER_API_KEY"] = "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"  # replace with your OpenRouter key

# --- SerpAPI Helper Function ---
def fetch_restaurants(location, query, radius_km=5):
    params = {
        "engine": "google_maps",
        "q": query,
        "location": location,
        "type": "restaurant",
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    return search.get_dict()

# --- Agents ---
food_recommender = Agent(
    name="Food Recommender",
    instructions=[
        "Suggest local dishes based on the city, user diet preference (veg/non-veg), and meal time (morning/afternoon/night).",
        "Take into account the liked food item to bias the recommendations.",
        "Make sure suggestions are local and culturally relevant.",
        "Format output with recommended dishes and why they are good choices."
    ],
    model=OpenRouter(id="google/gemini-2.5-flash-lite"),  # ✅ using OpenRouter
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

restaurant_recommender = Agent(
    name="Restaurant Finder",
    instructions=[
        "Search for restaurants near the given location.",
        "Filter by minimum rating, price/budget, and cuisine relevance (veg/non-veg, liked item).",
        "Return results with name, rating, address, price level if available, and Google Maps link.",
    ],
    model=OpenRouter(id="google/gemini-2.5-flash-lite"),  # ✅ using OpenRouter
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# --- Generate Plan ---
if st.button("🍲 Get Food Recommendations"):
    with st.spinner("🔍 Finding local dishes..."):
        food_prompt = (
            f"Suggest dishes in {location} for {meal_time}. "
            f"Preference: {diet}. Liked item: {liked_food}. Budget: {budget}."
        )
        dish_results = food_recommender.run(food_prompt, stream=False)

    with st.spinner("📍 Searching best restaurants..."):
        rest_prompt = (
            f"Find restaurants in {location} serving {liked_food} or similar. "
            f"Diet: {diet}, Meal Time: {meal_time}, Budget: {budget}, Min Rating: {min_rating}. "
            f"Include address, rating, price level, and Google Maps links."
        )
        rest_results = restaurant_recommender.run(rest_prompt, stream=False)

    # --- Display Results ---
    st.subheader("🍽️ Recommended Dishes")
    st.write(dish_results.content)

    st.subheader("🏨 Restaurant Suggestions")
    st.write(rest_results.content)

    st.success("✅ Food plan generated successfully!")
