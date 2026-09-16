"""Bao AI - Streamlit Rendering Components.

Small, presentation-only helpers used by streamlit_app.py. Split out so
streamlit_app.py stays about wiring the pipeline to the UI, not markup.

The animated tree here is deliberately scoped to one place: the "thinking"
state while the orchestrator is running. An avatar icon re-renders on every
message and users see it dozens of times per session — animating that
would be distracting, not polished. A loading state is shown once per
turn and exists specifically to communicate "something is happening," which
is exactly what motion is for. That's the difference between an animation
that earns its place and one that's just decoration.
"""

from __future__ import annotations

# Brand palette lifted from the logo (docs/assets/logo.jpg): deep navy line
# work over a lighter blue, on an off-white ground.
_NAVY = "#1b2a4a"
_BLUE = "#4a7ba6"
_LEAF = "#3d6b91"

_THINKING_TREE_HTML = f"""
<div class="bao-thinking">
  <svg viewBox="0 0 120 120" width="56" height="56" xmlns="http://www.w3.org/2000/svg">
    <g class="bao-branches" stroke="{_NAVY}" stroke-width="3" stroke-linecap="round" fill="none">
      <path d="M60 70 L60 30" />
      <path d="M60 55 L40 35" />
      <path d="M60 55 L80 35" />
      <path d="M60 42 L48 25" />
      <path d="M60 42 L72 25" />
    </g>
    <g class="bao-leaves" fill="{_LEAF}">
      <circle cx="40" cy="35" r="4" />
      <circle cx="80" cy="35" r="4" />
      <circle cx="48" cy="25" r="3.5" />
      <circle cx="72" cy="25" r="3.5" />
      <circle cx="60" cy="28" r="3.5" />
    </g>
    <g class="bao-roots" stroke="{_BLUE}" stroke-width="2.5" stroke-linecap="round" fill="none">
      <path d="M60 70 L45 90" />
      <path d="M60 70 L75 90" />
      <path d="M60 70 L60 95" />
    </g>
  </svg>
  <span class="bao-thinking-label">Thinking&hellip;</span>
</div>

<style>
.bao-thinking {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.25rem 0;
}}
.bao-thinking-label {{
    color: {_NAVY};
    opacity: 0.75;
    font-size: 0.9rem;
}}
.bao-branches {{
    transform-origin: 60px 70px;
    animation: bao-sway 2.2s ease-in-out infinite;
}}
.bao-leaves circle {{
    animation: bao-pulse 1.8s ease-in-out infinite;
}}
.bao-leaves circle:nth-child(2) {{ animation-delay: 0.2s; }}
.bao-leaves circle:nth-child(3) {{ animation-delay: 0.4s; }}
.bao-leaves circle:nth-child(4) {{ animation-delay: 0.6s; }}
.bao-leaves circle:nth-child(5) {{ animation-delay: 0.8s; }}
.bao-roots {{
    animation: bao-grow 2.2s ease-in-out infinite;
    transform-origin: 60px 70px;
}}
@keyframes bao-sway {{
    0%, 100% {{ transform: rotate(-1.5deg); }}
    50% {{ transform: rotate(1.5deg); }}
}}
@keyframes bao-pulse {{
    0%, 100% {{ opacity: 0.5; }}
    50% {{ opacity: 1; }}
}}
@keyframes bao-grow {{
    0%, 100% {{ opacity: 0.6; }}
    50% {{ opacity: 1; }}
}}
</style>
"""


def thinking_indicator_html() -> str:
    """Returns the branded loading animation shown while the orchestrator
    is processing a turn — a swaying/pulsing baobab silhouette instead of
    Streamlit's default spinner. Render with
    `st.markdown(thinking_indicator_html(), unsafe_allow_html=True)` inside
    an `st.empty()` placeholder so it can be cleared once a result is ready.
    """
    return _THINKING_TREE_HTML
