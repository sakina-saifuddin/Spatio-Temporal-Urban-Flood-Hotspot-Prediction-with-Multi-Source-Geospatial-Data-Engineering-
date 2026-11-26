"""Wrapping stat cards and tables rendered as HTML so long values never truncate."""

from __future__ import annotations

import html

import streamlit as st


def stat_cards(items, compact: bool = False) -> None:
    """Render equal-height cards. ``items``: iterable of (value, label) or (value, label, note)."""
    cards = []
    for item in items:
        value, label, *rest = item
        note = f'<div class="n">{html.escape(str(rest[0]), quote=False)}</div>' if rest and rest[0] else ""
        cards.append(
            f'<div class="fs-stat"><div class="v">{html.escape(str(value), quote=False)}</div>'
            f'<div class="l">{html.escape(str(label), quote=False)}</div>{note}</div>'
        )
    klass = "fs-stats fs-stats-compact" if compact else "fs-stats"
    st.markdown(f'<div class="{klass}">{"".join(cards)}</div>', unsafe_allow_html=True)


def html_table(rows, headers, caption: str | None = None) -> None:
    """Static table whose cells wrap, with no internal scrolling."""
    head = "".join(f"<th scope='col'>{html.escape(str(h), quote=False)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            (f"<th scope='row'>{html.escape(str(c), quote=False)}</th>" if i == 0 else f"<td>{html.escape(str(c), quote=False)}</td>")
            for i, c in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    cap = f"<caption>{html.escape(caption, quote=False)}</caption>" if caption else ""
    st.markdown(f'<div class="fs-table-wrap"><table class="fs-table">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)
