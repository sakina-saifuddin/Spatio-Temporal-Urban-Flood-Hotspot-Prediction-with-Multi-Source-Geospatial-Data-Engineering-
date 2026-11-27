"""Page registry shared by the router and in-page quick navigation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]

PAGES = [
    {"key": "home", "file": "pages/home.py", "title": "Home", "icon": "🏠",
     "chapter": "Chapter 1 · Welcome to Houston",
     "blurb": "Meet Scout and learn what FloodSense can and cannot tell you."},
    {"key": "explore", "file": "pages/explore.py", "title": "Explore Houston", "icon": "🗺️",
     "chapter": "Chapter 2 · Discover Your Neighborhood",
     "blurb": "Search a public place or coordinates, inspect cells, and compare two areas."},
    {"key": "priority", "file": "pages/priority.py", "title": "Priority Areas", "icon": "📋",
     "chapter": "Chapter 3 · Build a More Resilient Houston",
     "blurb": "Rank and filter sampled areas that may warrant further investigation."},
    {"key": "historical", "file": "pages/historical.py", "title": "Historical Storms", "icon": "🌧️",
     "chapter": "Chapter 4 · Journey Through Houston's Flood History",
     "blurb": "Compare saved predictions with recorded reports for four storms."},
    {"key": "insights", "file": "pages/model_insights.py", "title": "Behind the Predictions", "icon": "🔬",
     "chapter": "Chapter 5 · Discover the Science",
     "blurb": "Methods, validation, SHAP explanations, and limitations."},
]


def page_by_key(key: str) -> dict:
    return next(page for page in PAGES if page["key"] == key)


_PAGE_OBJECTS: dict = {}


def build_navigation():
    pages = []
    for page in PAGES:
        obj = st.Page(
            page["file"],
            title=page["title"],
            icon=page["icon"],
            url_path="" if page["key"] == "home" else page["key"],
            default=page["key"] == "home",
        )
        _PAGE_OBJECTS[page["key"]] = obj
        pages.append(obj)
    return st.navigation(pages, position="sidebar")


def page_link(key: str, label: str | None = None) -> None:
    page = page_by_key(key)
    target = _PAGE_OBJECTS.get(key, page["file"])
    st.page_link(target, label=label or page["title"], icon=page["icon"], width="stretch")


def switch_to(key: str) -> None:
    st.switch_page(_PAGE_OBJECTS.get(key, page_by_key(key)["file"]))


def chapter_header(key: str, title: str, subtitle: str) -> None:
    page = page_by_key(key)
    st.markdown(
        f'<p class="fs-chapter">{page["chapter"]}</p><h1 class="fs-title">{title}</h1>'
        f'<p class="fs-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )
