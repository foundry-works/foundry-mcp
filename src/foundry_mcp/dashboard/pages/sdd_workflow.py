"""SDD Workflow page - spec progress, phase burndown, task tracking."""

import streamlit as st

from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.components.charts import pie_chart, bar_chart, empty_chart

# Try importing pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _get_specs():
    """Get list of specifications."""
    try:
        from foundry_mcp.core.spec import list_specs
        from foundry_mcp.config import get_config

        config = get_config()
        specs_dir = config.specs_dir
        return list_specs(specs_dir)
    except ImportError:
        return []
    except Exception:
        return []


def _get_spec_data(spec_id: str):
    """Get detailed spec data."""
    try:
        from foundry_mcp.core.spec import load_spec
        from foundry_mcp.config import get_config

        config = get_config()
        return load_spec(config.specs_dir, spec_id)
    except Exception:
        return None


def _calculate_progress(spec_data: dict) -> dict:
    """Calculate progress metrics from spec data."""
    if not spec_data:
        return {"total": 0, "completed": 0, "in_progress": 0, "pending": 0, "blocked": 0, "percentage": 0}

    tasks = spec_data.get("tasks", {})
    status_counts = {"completed": 0, "in_progress": 0, "pending": 0, "blocked": 0}

    for task_id, task in tasks.items():
        status = task.get("status", "pending")
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["pending"] += 1

    total = sum(status_counts.values())
    percentage = (status_counts["completed"] / total * 100) if total > 0 else 0

    return {
        "total": total,
        **status_counts,
        "percentage": percentage,
    }


def render():
    """Render the SDD Workflow page."""
    st.header("SDD Workflow")

    # Get specs
    specs = _get_specs()

    if not specs:
        st.info("No specifications found.")
        st.caption("Create specs using `foundry-cli spec create` or the MCP spec-create tool.")
        return

    # Spec selector
    spec_options = {s.get("title", s.get("id", "unknown")): s.get("id") for s in specs if isinstance(s, dict)}

    if not spec_options:
        st.warning("Could not parse specification list")
        return

    selected_title = st.selectbox(
        "Select Specification",
        options=list(spec_options.keys()),
        key="sdd_spec_selector",
    )

    selected_id = spec_options.get(selected_title)
    if not selected_id:
        return

    # Load spec data
    spec_data = _get_spec_data(selected_id)
    if not spec_data:
        st.warning(f"Could not load spec: {selected_id}")
        return

    # Calculate progress
    progress = _calculate_progress(spec_data)

    st.divider()

    # Progress overview
    st.subheader("Progress Overview")

    # Progress bar
    st.progress(progress["percentage"] / 100, text=f"Overall Progress: {progress['percentage']:.1f}%")

    # KPI cards
    kpi_row(
        [
            {"label": "Total Tasks", "value": progress["total"]},
            {"label": "Completed", "value": progress["completed"]},
            {"label": "In Progress", "value": progress["in_progress"]},
            {"label": "Pending", "value": progress["pending"]},
            {"label": "Blocked", "value": progress["blocked"]},
        ],
        columns=5,
    )

    st.divider()

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Status Distribution")
        if PANDAS_AVAILABLE and progress["total"] > 0:
            status_df = pd.DataFrame([
                {"status": "completed", "count": progress["completed"]},
                {"status": "in_progress", "count": progress["in_progress"]},
                {"status": "pending", "count": progress["pending"]},
                {"status": "blocked", "count": progress["blocked"]},
            ])
            status_df = status_df[status_df["count"] > 0]  # Filter zero counts

            pie_chart(
                status_df,
                values="count",
                names="status",
                title=None,
                height=300,
            )
        else:
            empty_chart("No task data")

    with col2:
        st.subheader("Phase Progress")
        phases = spec_data.get("phases", [])
        if PANDAS_AVAILABLE and phases:
            phase_data = []
            for phase in phases:
                if isinstance(phase, dict):
                    phase_tasks = phase.get("tasks", [])
                    completed = len([t for t in phase_tasks if isinstance(t, dict) and t.get("status") == "completed"])
                    total = len(phase_tasks)
                    phase_data.append({
                        "phase": phase.get("title", phase.get("id", "unknown"))[:20],
                        "completed": completed,
                        "total": total,
                        "percentage": (completed / total * 100) if total > 0 else 0,
                    })

            if phase_data:
                phase_df = pd.DataFrame(phase_data)
                bar_chart(
                    phase_df,
                    x="phase",
                    y="percentage",
                    title=None,
                    height=300,
                )
            else:
                empty_chart("No phase data")
        else:
            empty_chart("No phase data")

    st.divider()

    # Task list
    st.subheader("Tasks")
    tasks = spec_data.get("tasks", {})

    if tasks:
        # Status filter
        status_filter = st.multiselect(
            "Filter by Status",
            options=["completed", "in_progress", "pending", "blocked"],
            default=["in_progress", "pending", "blocked"],
            key="task_status_filter",
        )

        # Build task list
        task_list = []
        for task_id, task in tasks.items():
            if isinstance(task, dict):
                status = task.get("status", "pending")
                if status in status_filter:
                    task_list.append({
                        "id": task_id,
                        "title": task.get("title", "Untitled"),
                        "status": status,
                        "estimated_hours": task.get("estimated_hours", 0),
                        "actual_hours": task.get("actual_hours", 0),
                    })

        if task_list and PANDAS_AVAILABLE:
            task_df = pd.DataFrame(task_list)
            st.dataframe(
                task_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "estimated_hours": st.column_config.NumberColumn("Est. Hours", format="%.1f", width="small"),
                    "actual_hours": st.column_config.NumberColumn("Act. Hours", format="%.1f", width="small"),
                },
            )
        elif task_list:
            for t in task_list:
                st.text(f"- [{t['status']}] {t['title']}")
        else:
            st.info("No tasks matching filter")
    else:
        st.info("No tasks defined in this spec")

    # Time tracking summary
    st.divider()
    st.subheader("Time Tracking")

    total_estimated = sum(t.get("estimated_hours", 0) for t in tasks.values() if isinstance(t, dict))
    total_actual = sum(t.get("actual_hours", 0) for t in tasks.values() if isinstance(t, dict))
    variance = total_actual - total_estimated

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Hours", f"{total_estimated:.1f}h")
    with col2:
        st.metric("Actual Hours", f"{total_actual:.1f}h")
    with col3:
        delta_color = "inverse" if variance > 0 else "normal"
        st.metric(
            "Variance",
            f"{variance:+.1f}h",
            delta=f"{(variance / total_estimated * 100):+.1f}%" if total_estimated > 0 else "N/A",
            delta_color=delta_color,
        )
