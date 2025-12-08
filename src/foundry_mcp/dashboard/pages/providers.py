"""Providers page - AI provider status grid."""

import streamlit as st

from foundry_mcp.dashboard.components.cards import status_badge
from foundry_mcp.dashboard.data.stores import get_providers


def render():
    """Render the Providers page."""
    st.header("AI Providers")

    providers = get_providers()

    if not providers:
        st.info("No AI providers configured or providers module not available.")
        st.caption("Providers are configured in foundry-mcp.toml under [providers]")
        return

    # Summary
    available_count = len([p for p in providers if p.get("available")])
    total_count = len(providers)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Providers", total_count)
    with col2:
        st.metric("Available", available_count)
    with col3:
        st.metric("Unavailable", total_count - available_count)

    st.divider()

    # Provider grid
    st.subheader("Provider Status")

    # Create grid layout
    cols = st.columns(3)

    for i, provider in enumerate(providers):
        with cols[i % 3]:
            with st.container(border=True):
                # Header with status
                provider_id = provider.get("id", "unknown")
                is_available = provider.get("available", False)

                status = "available" if is_available else "unavailable"
                status_badge(status, label=provider_id)

                # Description
                description = provider.get("description", "")
                if description:
                    st.caption(description[:100] + "..." if len(description) > 100 else description)

                # Tags
                tags = provider.get("tags", [])
                if tags:
                    st.markdown(" ".join([f"`{tag}`" for tag in tags[:5]]))

                # Models
                models = provider.get("models", [])
                if models:
                    with st.expander("Models"):
                        for model in models[:10]:
                            if isinstance(model, dict):
                                st.text(f"- {model.get('id', model.get('name', 'unknown'))}")
                            else:
                                st.text(f"- {model}")
                        if len(models) > 10:
                            st.caption(f"...and {len(models) - 10} more")

                # Metadata
                metadata = provider.get("metadata", {})
                if metadata:
                    with st.expander("Details"):
                        for key, value in list(metadata.items())[:5]:
                            st.text(f"{key}: {value}")

    # Refresh button
    st.divider()
    if st.button("Refresh Provider Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
