"""
Display Module — Terminal Output & Charts
==========================================
Handles all user-facing output for the Travel Weather Advisor.

Components:
  1. Terminal Display (Rich) — Colorful formatted tables and panels
     - Trip header with destination, dates, purpose
     - Per-day weather + clothing table
     - Master packing list (deduplicated across all days)
     - Day-by-day weather alerts
     - Daily clothing narratives (natural language suggestions)
     - Optimization results (3 stages: GA, Knapsack, Summary)

  2. Matplotlib Charts — 3-panel forecast visualization
     - Temperature band (min/max fill + average line)
     - Precipitation bar chart
     - UV index line with risk-level shading

  3. Plain Text Fallback — When Rich is not installed
     - Same information in plain text format

Dependencies:
  - rich (optional) — for colored terminal output
  - matplotlib (optional) — for forecast charts
"""

from datetime import date as _date
from pathlib import Path
from models import DayRecommendation, TripContext
from collections import Counter  # For counting item quantities in optimization display
import os
import subprocess
import sys

# Try to import Rich for pretty terminal output; fall back to plain text
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_date(d: str) -> str:
    """
    Format an ISO date string (YYYY-MM-DD) into a readable format.
    Example: "2026-05-01" → "Fri 01 May"
    """
    try:
        return _date.fromisoformat(d).strftime("%a %d %b")
    except Exception:
        return d


def _join_items(items: list) -> str:
    """
    Join a list of items into a natural English phrase.
    Example: ["t-shirt", "shorts", "jacket"] → "t-shirt, shorts and jacket"
    """
    if len(items) == 1:
        return items[0].lower()
    return ", ".join(i.lower() for i in items[:-1]) + f" and {items[-1].lower()}"


def _alert_color(alerts: list) -> str:
    """
    Determine the color for alert display based on severity.
    - Red: Heavy/strong/violent weather or thunderstorms
    - Yellow: Other alerts (moderate rain, UV, wind, snow)
    - Green: No alerts
    """
    if not alerts:
        return "green"
    text = " ".join(alerts).lower()
    if any(w in text for w in ("heavy", "violent", "strong", "thunderstorm")):
        return "red"
    return "yellow"


def _fitness_bar(score: float, width: int = 8) -> str:
    """
    Render a fitness score (0–1) as a colored text progress bar.
    - Green: ≥ 0.7 (good)
    - Yellow: ≥ 0.4 (moderate)
    - Red: < 0.4 (poor)
    
    Example: score=0.75 → "██████░░ 0.75" (green)
    """
    filled = round(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
    return f"[{color}]{bar}[/] {score:.2f}"


# ═══════════════════════════════════════════════════════════════════════════
# CLOTHING NARRATIVE — Natural language daily suggestions
# ═══════════════════════════════════════════════════════════════════════════

def _clothing_narrative(rec) -> str:
    """
    Turn a day's clothing list into a readable paragraph.
    
    Groups items into:
      - Base layers (t-shirts, shirts, trousers, shorts, casual wear)
      - Outer layers (jackets, coats, waterproofs)
      - Footwear (shoes, boots)
      - Accessories (gloves, scarves, socks)
    
    Then weaves them into flowing sentences.
    
    Example output:
      "On Fri 01 May, expect Moderate rain, 26–32°C. 
       For your base layer, go with t-shirt, shorts and casual wear. 
       Bring waterproof jacket for the outer layer. 
       Comfortable walking shoes will be the right footwear choice."
    """
    # Define item categories for grouping
    BASE_LAYER   = {"T-shirt or short sleeves", "Lightweight breathable clothing",
                    "Long-sleeve shirt", "Thermal underlayer", "Warm sweater or fleece",
                    "Shorts or light trousers", "Jeans or trousers",
                    "Casual wear", "Smart casual outfit", "Business attire"}
    OUTER_LAYER  = {"Heavy winter coat", "Light jacket or fleece", "Waterproof jacket",
                    "Windproof jacket", "Smart jacket"}
    FOOTWEAR     = {"Insulated boots", "Waterproof snow boots", "Comfortable walking shoes",
                    "Formal shoes"}
    ACCESSORIES  = {"Gloves and scarf", "Thermal socks"}

    clothing = rec.clothing
    
    # If no clothing changes needed for the day
    if not clothing:
        return f"On {_fmt_date(rec.date)}, the weather is {rec.summary}. No specific clothing changes needed for the day."

    # Group items into categories
    base    = [c for c in clothing if c in BASE_LAYER]
    outer   = [c for c in clothing if c in OUTER_LAYER]
    shoes   = [c for c in clothing if c in FOOTWEAR]
    acc     = [c for c in clothing if c in ACCESSORIES]
    other   = [c for c in clothing if c not in BASE_LAYER | OUTER_LAYER | FOOTWEAR | ACCESSORIES]

    # Build sentence parts for each category
    parts = []
    if base:
        parts.append("For your base layer, go with " + _join_items(base))
    if outer:
        parts.append("bring " + _join_items(outer) + " for the outer layer")
    if shoes:
        parts.append(_join_items(shoes) + " will be the right footwear choice")
    if acc:
        parts.append("add " + _join_items(acc) + " to stay comfortable")
    if other:
        parts.append("also consider " + _join_items(other))

    # Combine parts into flowing sentences
    sentence = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
    intro = f"On {_fmt_date(rec.date)}, expect {rec.summary}."
    
    return f"{intro} {sentence}"


# ═══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB FORECAST CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_forecast(forecasts: list, context: TripContext,
                  start_date: str, end_date: str) -> Path:
    """
    Generate a 3-panel forecast chart:
      1. Temperature band (min/max fill + average line)
      2. Precipitation bar chart with heavy rain threshold
      3. UV index line with risk-level shading (green/yellow/orange/red)
    
    Saves as PNG and returns the file path.
    
    Args:
        forecasts: List of DayForecast objects
        context: TripContext (city, country, purpose)
        start_date: Trip start date (YYYY-MM-DD)
        end_date: Trip end date (YYYY-MM-DD)
    
    Returns:
        Path to saved PNG file
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # Prepare data arrays
    dates    = [_date.fromisoformat(f.date) for f in forecasts]
    temp_min = np.array([f.temp_min for f in forecasts])
    temp_max = np.array([f.temp_max for f in forecasts])
    temp_avg = (temp_min + temp_max) / 2
    precip   = np.array([f.precipitation_mm for f in forecasts])
    uv       = np.array([f.uv_index_max for f in forecasts])

    method_label = "Historical Prediction (past climate avg + trend)"
    title = (f"{context.city}, {context.country}  |  "
             f"{start_date} → {end_date}  |  {method_label}")

    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: Temperature ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(dates, temp_min, temp_max,
                     alpha=0.25, color="tomato", label="Min–Max range")
    ax1.plot(dates, temp_avg, color="crimson",
             linewidth=2, marker="o", markersize=4, label="Avg temp")
    ax1.plot(dates, temp_min, color="steelblue",
             linewidth=1, linestyle="--", alpha=0.6, label="Min")
    ax1.plot(dates, temp_max, color="darkorange",
             linewidth=1, linestyle="--", alpha=0.6, label="Max")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Precipitation ────────────────────────────────────────────────
    ax2 = axes[1]
    # Color bars by intensity: red (>20mm heavy), blue (>1mm rain), light blue (dry)
    colors = ["#d32f2f" if p > 20 else "#1976d2" if p > 1 else "#90caf9"
              for p in precip]
    ax2.bar(dates, precip, color=colors, alpha=0.8, width=0.6, label="Precipitation")
    ax2.axhline(1,  color="gray",  linewidth=0.8, linestyle="--", alpha=0.5, label="1mm threshold")
    ax2.axhline(20, color="red",   linewidth=0.8, linestyle="--", alpha=0.5, label="Heavy rain (20mm)")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: UV Index ─────────────────────────────────────────────────────
    ax3 = axes[2]
    # Risk-level background bands (WHO UV index scale)
    ax3.axhspan(0,  3,  alpha=0.07, color="green")   # Low
    ax3.axhspan(3,  6,  alpha=0.07, color="yellow")  # Moderate
    ax3.axhspan(6,  8,  alpha=0.07, color="orange")  # High
    ax3.axhspan(8,  16, alpha=0.07, color="red")     # Very High/Extreme
    for level, label, color in [(3, "Moderate (3)", "gold"),
                                 (6, "High (6)", "orange"),
                                 (8, "Very High (8)", "red")]:
        ax3.axhline(level, color=color, linewidth=0.8, linestyle="--", alpha=0.6, label=label)
    ax3.plot(dates, uv, color="darkorange",
             linewidth=2, marker="s", markersize=4, label="UV Index")
    ax3.set_ylabel("UV Index")
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── X-axis formatting ─────────────────────────────────────────────────────
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and return path
    city_slug = context.city.lower().replace(" ", "_").replace(",", "")
    out_path = Path(__file__).parent / f"{city_slug}_{start_date}_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# RICH TERMINAL DISPLAY — Main formatted output
# ═══════════════════════════════════════════════════════════════════════════

def display_rich(context, start_date, end_date, recommendations, trip_packing, 
                 n_years=None, optimization_result=None):
    """
    Main Rich terminal display function.
    
    Shows in order:
      1. Trip header (destination, dates, purpose, method)
      2. Per-day weather + clothing table
      3. Master packing list (deduplicated across all days)
      4. Day-by-day weather alerts
      5. Optimization results (if --optimize was used):
         - Stage 1: GA Pareto Front (different packing options)
         - Stage 2: Knapsack Final Selection (with quantities)
         - Stage 3: Optimization Summary (removed items with reasons)
    """
    console = Console()

    # ── Trip Header ───────────────────────────────────────────────────────────
    method_tag = f"[yellow]Historical Prediction ({n_years}yr avg + trend)[/]"
    header = (
        f"[bold]Trip:[/] {context.city}, {context.country}  |  "
        f"[bold]Dates:[/] {_fmt_date(start_date)} → {_fmt_date(end_date)}  |  "
        f"[bold]Purpose:[/] {context.purpose.title()}  |  "
        f"[bold]Method:[/] {method_tag}"
    )
    console.print(Panel(header, title="[bold cyan]Travel Weather Advisor[/]", border_style="cyan"))

    # ── Per-Day Weather + Clothing Table ──────────────────────────────────────
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Date",     style="cyan",  no_wrap=True)
    table.add_column("Weather",  style="white")
    table.add_column("Clothing", style="white")
    table.add_column("Alerts",   style="white")

    for rec in recommendations:
        alert_color  = _alert_color(rec.alerts)
        alerts_text  = "\n".join(rec.alerts) if rec.alerts else "[green]All good[/]"
        clothing_text = "\n".join(f"• {c}" for c in rec.clothing[:4])
        if len(rec.clothing) > 4:
            clothing_text += f"\n  [dim]+ {len(rec.clothing) - 4} more[/]"
        table.add_row(
            _fmt_date(rec.date), rec.summary, clothing_text,
            f"[{alert_color}]{alerts_text}[/]",
        )
    console.print(table)

    # ── Master Packing List (deduplicated across all days) ─────────────────────
    clothing_lines = "\n".join(f"  • {c}" for c in trip_packing["clothing"])
    packing_lines  = "\n".join(f"  • {p}" for p in trip_packing["packing"])
    console.print(Panel(
        f"[bold]Clothing:[/]\n{clothing_lines}\n\n[bold]Gear & Essentials:[/]\n{packing_lines}",
        title="[bold green]Master Packing List[/]", border_style="green",
    ))

    # ── Day-by-Day Weather Alerts ─────────────────────────────────────────────
    tips = []
    for rec in recommendations:
        if rec.alerts:
            tips.append(f"[cyan]{_fmt_date(rec.date)}[/]: " + "  " + "\n  ".join(rec.alerts))
    if tips:
        console.print(Panel("\n".join(tips), title="[bold yellow]Day-by-Day Alerts[/]", border_style="yellow"))

    # ── Daily Clothing Narratives ─────────────────────────────────────────────
    narratives = "\n\n".join(_clothing_narrative(rec) for rec in recommendations)

    # ── Optimization Results (only when --optimize was used) ──────────────────
    if optimization_result is not None:
        _display_rich_optimizer(console, optimization_result)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION DISPLAY — Three-stage results panel
# ═══════════════════════════════════════════════════════════════════════════

def _display_rich_optimizer(console, opt) -> None:
    """
    Render the three optimizer stages as separate rich panels.
    
    Stage 1 — Genetic Algorithm Pareto Front:
      Shows 6-8 unique packing solutions found by the GA, each with
      different trade-offs between comfort (weather suitability) and
      weight (lightness). Displayed as colored progress bars.
    
    Stage 2 — Knapsack Final Selection:
      Shows the single best plan selected, with PLAN ID, quantities, and weight.
      Example: "5x T-shirt or short sleeves", "2x Shorts or light trousers"
      Also shows total weight vs weight limit.
    
    Stage 3 — Optimization Summary:
      Shows what was removed and WHY (with reasons).
      Shows final fitness scores (comfort + weight).
      Shows volume and weight utilization percentages.
      Includes the basic explanation of the optimization result.
    """
    
    from rich.table import Table as _Table

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: Genetic Algorithm Pareto Front
    # Shows different trade-off solutions found by the GA with QUANTITIES
    # ═══════════════════════════════════════════════════════════════════════
    ga_table = _Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    ga_table.add_column("ID",      style="bold white", no_wrap=True, width=4)
    ga_table.add_column("Items",   style="white")
    ga_table.add_column("Comfort", style="white", no_wrap=True, width=9)
    ga_table.add_column("Weight",  style="white", no_wrap=True, width=9)

    # Store Plan A items for comparison
    plan_a_items = set()

    for s in opt.ga_solutions:
        comfort_bar = _fitness_bar(s.fitness_comfort)
        weight_bar  = _fitness_bar(s.fitness_weight)
        
        # Filter out non-clothing and placeholder items
        clean_items = [
            it for it in s.items 
            if "person" not in it.lower() 
            and "unknown" not in it.lower()
            and "no detection" not in it.lower()
            and "NEEDS_PURCHASE" not in it
        ]
        
        # Group by item name to show quantities per type
        item_counts = Counter(clean_items)
        unique_types = len(item_counts)
        total_copies = len(clean_items)
        
        # Sort: most numerous items first, then alphabetically
        sorted_items = sorted(item_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Show top 3 item types with counts
        items_list = []
        for item_name, count in sorted_items[:3]:
            if count > 1:
                items_list.append(f"{count}x {item_name}")
            else:
                items_list.append(item_name)
        
        items_str = ", ".join(items_list)
        
        # Show total types/items count
        remaining_types = unique_types - 3
        if remaining_types > 0:
            items_str += f"\n     [dim]↳ +{remaining_types} more types ({unique_types} types, {total_copies} items)[/]"
        else:
            items_str += f"\n     [dim]↳ {unique_types} types, {total_copies} items[/]"
        
        # Show how this plan differs from Plan A
        if s.solution_id == "A":
            plan_a_items = set(clean_items)
        elif plan_a_items:
            this_plan = set(clean_items)
            removed_from_a = plan_a_items - this_plan
            added_to_a = this_plan - plan_a_items
            
            if removed_from_a or added_to_a:
                diff = []
                if removed_from_a:
                    diff.append(f"-{', '.join(list(removed_from_a)[:2])}")
                if added_to_a:
                    diff.append(f"+{', '.join(list(added_to_a)[:2])}")
                items_str += f"\n     [dim]vs A: {'; '.join(diff)}[/]"
            else:
                items_str += f"\n     [dim]vs A: same items, different qty[/]"
        
        ga_table.add_row(s.solution_id, items_str, comfort_bar, weight_bar)

    console.print(Panel(
        ga_table,
        title="[bold magenta]Stage 1 — Genetic Algorithm  Pareto Front[/]",
        border_style="magenta",
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: Knapsack Final Selection — Best plan from GA
    # ═══════════════════════════════════════════════════════════════════════
    weight_color = "green" if opt.total_weight <= opt.weight_limit else "red"
    weight_line  = (f"[{weight_color}]{opt.total_weight:.1f}kg[/] "
                    f"/ {opt.weight_limit:.1f}kg limit")
    
    # Filter out non-clothing and placeholder items from final list
    clean_final = [
        it for it in opt.final_items 
        if "person" not in it.lower() 
        and "unknown" not in it.lower()
        and "NEEDS_PURCHASE" not in it
    ]
    
    # Count how many of each item type
    item_counts = Counter(clean_final)
    
    # [ADDED] Identify which GA plan was selected by matching item sets
    selected_plan_id = None
    final_set = set(clean_final)
    for s in opt.ga_solutions:
        ga_set = set(s.items)
        if ga_set == final_set:
            selected_plan_id = s.solution_id
            break
    
    # If exact match not found, find the most similar GA solution
    if selected_plan_id is None and opt.ga_solutions:
        best_overlap = 0
        for s in opt.ga_solutions:
            overlap = len(set(s.items) & final_set)
            if overlap > best_overlap:
                best_overlap = overlap
                selected_plan_id = s.solution_id
    
    # Build the packing list display with quantities, sorted alphabetically
    items_lines = ""
    for item_name, count in sorted(item_counts.items()):
        if count > 1:
            items_lines += f"  • {count}x {item_name}\n"
        else:
            items_lines += f"  • {item_name}\n"
    
    # [ADDED] Build title with selected plan ID
    title_text = (f"Stage 2 — Knapsack  Final Selection  (Plan {selected_plan_id} selected)" 
                  if selected_plan_id else "Stage 2 — Knapsack  Final Selection")
    
    console.print(Panel(
        f"[bold]Plan selected:[/] [cyan]{selected_plan_id if selected_plan_id else 'Best'}[/]\n"
        f"[bold]Total weight:[/] {weight_line}\n\n"
        f"[bold]Optimized packing list ({len(item_counts)} unique items, "
        f"{len(clean_final)} total):[/]\n"
        f"{items_lines}",
        title=f"[bold green]{title_text}[/]",
        border_style="green",
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: Optimization Summary
    # Shows what was removed, why, and final metrics
    # ═══════════════════════════════════════════════════════════════════════
    
    # Filter removed items to exclude non-clothing detections
    clean_removed = [
        it for it in opt.removed_items 
        if "person" not in it.lower() 
        and "unknown" not in it.lower()
        and "NEEDS_PURCHASE" not in it
    ]
    
    # Count removed items by type and show reasons
    if clean_removed:
        removed_counts = Counter(clean_removed)
        removed_lines = ""
        for item_name, count in sorted(removed_counts.items()):
            # Get the reason this item was removed (from GA/optimizer)
            reason = opt.removal_reasons.get(item_name, ["No specific reason"])
            reason_str = reason[0] if reason else "Unknown"
            
            if count > 1:
                removed_lines += f"  [red]✗[/] {count}x {item_name} — {reason_str}\n"
            else:
                removed_lines += f"  [red]✗[/] {item_name} — {reason_str}\n"
    else:
        removed_lines = "  [green]Nothing removed — all items fit within constraints.[/]\n"
    
    # Show weight and volume utilization metrics
    metrics_lines = (
        f"Comfort [bold]{opt.final_fitness_comfort:.2f}[/]  |  "
        f"Weight  [bold]{opt.final_fitness_weight:.2f}[/]\n"
        f"Volume used: {opt.volume_utilization:.1f}%  |  "
        f"Weight used: {opt.weight_utilization:.1f}%"
    )
    
    console.print(Panel(
        f"[bold]Removed items:[/]\n{removed_lines}\n"
        f"[bold]Final fitness:[/]  {metrics_lines}\n\n"
        f"[bold]Explanation:[/]  {opt.basic_explanation}",
        title="[bold yellow]Stage 3 — Optimization Summary[/]",
        border_style="yellow",
    ))

# ═══════════════════════════════════════════════════════════════════════════
# PLAIN TEXT FALLBACK — When Rich is not installed
# ═══════════════════════════════════════════════════════════════════════════

def display_plain(context, start_date, end_date, recommendations, trip_packing, n_years=None):
    """
    Plain text fallback display when Rich is not installed.
    Shows the same information without colors or formatting.
    """
    sep = "-" * 60
    print(sep)
    print("Travel Weather Advisor")
    print(f"Trip   : {context.city}, {context.country}")
    print(f"Dates  : {start_date} to {end_date}")
    print(f"Purpose: {context.purpose.title()}")
    print(sep)
    
    # Per-day recommendations
    for rec in recommendations:
        print(f"\n{_fmt_date(rec.date)} — {rec.summary}")
        for c in rec.clothing:
            print(f"    • {c}")
        for p in rec.packing:
            print(f"    * {p}")
        for a in rec.alerts:
            print(f"    ! {a}")
    
    # Master packing list
    print(f"\n{sep}\nMASTER PACKING LIST\n{sep}")
    for c in trip_packing["clothing"]:
        print(f"  • {c}")
    for p in trip_packing["packing"]:
        print(f"  * {p}")
    print(sep)
    
    # Daily clothing narratives
    print(f"\n{sep}\nDAILY CLOTHING SUGGESTIONS\n{sep}")
    for rec in recommendations:
        print()
        print(_clothing_narrative(rec))
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def display(context, start_date, end_date, recommendations, trip_packing, 
            n_years=None, optimization_result=None):
    """
    Main display entry point.
    Automatically uses Rich if available, falls back to plain text.
    """
    if RICH_AVAILABLE:
        display_rich(context, start_date, end_date, recommendations, trip_packing, n_years, optimization_result)
    else:
        display_plain(context, start_date, end_date, recommendations, trip_packing, n_years)

# ═══════════════════════════════════════════════════════════════════════════
# JSON EXPORT — For GUI/Streamlit integration
# ═══════════════════════════════════════════════════════════════════════════

def export_for_gui(optimization_result, trip_packing, photo_recommendations=None) -> dict:
    """
    Export final packing list and optimization results in a GUI-friendly JSON format.
    
    Returns a clean dictionary ready for JSON serialization:
    {
        "packing_list": [
            {"item": "T-shirt or short sleeves", "quantity": 3, "category": "clothing"},
            {"item": "Shorts or light trousers", "quantity": 2, "category": "clothing"},
            {"item": "Compact umbrella", "quantity": 1, "category": "packing"},
            ...
        ],
        "summary": {
            "unique_items": 10,
            "total_items": 14,
            "total_weight_kg": 3.9,
            "weight_limit_kg": 20.0,
            "volume_utilization_pct": 56.8,
            "weight_utilization_pct": 19.5,
            "comfort_score": 0.75,
            "space_left_for_souvenirs_l": 30.5
        },
        "optimization_plan": "Plan A",
        "removed_items": [...],
        "photo_suggestions": [...]  (if photos provided)
    }
    """
    from collections import Counter
    
    # Define item categories for classification
    CLOTHING_ITEMS = {
        "Heavy winter coat", "Thermal underlayer", "Warm sweater or fleece",
        "Gloves and scarf", "Insulated boots", "Light jacket or fleece",
        "Long-sleeve shirt", "Jeans or trousers", "T-shirt or short sleeves",
        "Lightweight breathable clothing", "Shorts or light trousers",
        "Waterproof jacket", "Windproof jacket", "Waterproof snow boots",
        "Thermal socks", "Business attire", "Formal shoes",
        "Comfortable walking shoes", "Casual wear", "Smart casual outfit",
        "Smart jacket",
    }
    PACKING_ITEMS = {
        "Compact umbrella", "Waterproof bag cover", "Sunscreen SPF 50+",
        "Sunscreen SPF 30+", "Sunglasses", "Wide-brim hat",
        "Hand warmers", "Laptop bag", "Power adapter", "Business cards",
        "Day backpack", "Phone charger / power bank", "City map or offline maps",
        "Reusable water bottle", "Small gift (optional)", "Phone charger",
    }
    
    output = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. PACKING LIST — With quantities and categories
    # ═══════════════════════════════════════════════════════════════════
    if optimization_result and optimization_result.final_items:
        # Filter out non-clothing detections
        clean_items = [
            it for it in optimization_result.final_items
            if "person" not in it.lower()
            and "unknown" not in it.lower()
            and "NEEDS_PURCHASE" not in it
        ]
        
        # Count quantities
        item_counts = Counter(clean_items)
        
        packing_list = []
        for item_name, count in sorted(item_counts.items()):
            # Determine category
            if item_name in CLOTHING_ITEMS:
                category = "clothing"
            elif item_name in PACKING_ITEMS:
                category = "packing"
            else:
                category = "other"
            
            packing_list.append({
                "item": item_name,
                "quantity": count,
                "category": category,
            })
        
        output["packing_list"] = packing_list
        
        # ═══════════════════════════════════════════════════════════════
        # 2. SUMMARY — Key metrics for display
        # ═══════════════════════════════════════════════════════════════
        output["summary"] = {
            "unique_items": len(item_counts),
            "total_items": len(clean_items),
            "total_weight_kg": optimization_result.total_weight,
            "weight_limit_kg": optimization_result.weight_limit,
            "volume_utilization_pct": optimization_result.volume_utilization,
            "weight_utilization_pct": optimization_result.weight_utilization,
            "comfort_score": round(optimization_result.comfort_score, 3),
            "space_left_for_souvenirs_l": optimization_result.space_left_for_souvenirs,
            "optimization_mode": optimization_result.basic_explanation.split(":")[0].replace("Optimized (", "").replace(")", "") if optimization_result.basic_explanation else "balanced",
        }
        
        # ═══════════════════════════════════════════════════════════════
        # 3. OPTIMIZATION PLAN — Which GA plan was selected
        # ═══════════════════════════════════════════════════════════════
        # Find which GA plan matches the final items
        final_set = set(clean_items)
        selected_plan = None
        for s in (optimization_result.ga_solutions or []):
            if set(s.items) == final_set:
                selected_plan = s.solution_id
                break
        
        output["optimization_plan"] = selected_plan or "Best"
        output["comfort_fitness"] = optimization_result.final_fitness_comfort
        output["weight_fitness"] = optimization_result.final_fitness_weight
        
        # ═══════════════════════════════════════════════════════════════
        # 4. REMOVED ITEMS — What was excluded and why
        # ═══════════════════════════════════════════════════════════════
        clean_removed = [
            it for it in optimization_result.removed_items
            if "person" not in it.lower()
            and "unknown" not in it.lower()
            and "NEEDS_PURCHASE" not in it
        ]
        
        removed_counts = Counter(clean_removed)
        removed_list = []
        for item_name, count in sorted(removed_counts.items()):
            reasons = optimization_result.removal_reasons.get(item_name, ["Optimization trade-off"])
            removed_list.append({
                "item": item_name,
                "quantity_removed": count,
                "reason": reasons[0] if reasons else "Unknown",
            })
        
        output["removed_items"] = removed_list
        
        # ═══════════════════════════════════════════════════════════════
        # 5. INSIGHTS — Human-readable optimization insights
        # ═══════════════════════════════════════════════════════════════
        output["insights"] = optimization_result.optimization_insights or []
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. PHOTO SUGGESTIONS — If wardrobe photos were provided
    # ═══════════════════════════════════════════════════════════════════
    if photo_recommendations:
        photo_suggestions = []
        
        # Items found in wardrobe with specific photos
        for item in photo_recommendations.get("recommended_items", []):
            photo_suggestions.append({
                "item": item["item_name"],
                "quantity_needed": item["quantity_needed"],
                "pack_count": item["pack_count"],
                "missing_count": item["missing_count"],
                "suitability_score": item["suitability_score"],
                "photos_to_pack": item.get("photos_to_pack", []),
                "status": "found_in_wardrobe",
            })
        
        # Items not in wardrobe — need to purchase
        for item in photo_recommendations.get("items_without_photos", []):
            photo_suggestions.append({
                "item": item["item_name"],
                "quantity_needed": item["quantity_needed"],
                "pack_count": 0,
                "missing_count": item["quantity_needed"],
                "suitability_score": item["suitability_score"],
                "photos_to_pack": [],
                "status": "needs_purchase",
            })
        
        output["photo_suggestions"] = photo_suggestions
    
    # ═══════════════════════════════════════════════════════════════════
    # 7. 3D BIN PACKING LAYOUT — If available
    # ═══════════════════════════════════════════════════════════════════
    if optimization_result and optimization_result.bin_pack_layout:
        output["bin_pack_3d"] = optimization_result.bin_pack_layout
    
    return output


def save_gui_json(optimization_result, trip_packing, output_path=None, photo_recommendations=None) -> str:
    """
    Save the GUI-friendly JSON to a file.
    
    Args:
        optimization_result: From packing_optimizer
        trip_packing: From recommender build_trip_packing_list()
        output_path: Where to save (default: gui_packing_output.json)
        photo_recommendations: From main.py build_photo_recommendations()
    
    Returns:
        Path to the saved JSON file
    """
    import json
    from pathlib import Path
    
    if output_path is None:
        output_path = Path(__file__).parent / "summary_optimized_packing_list.json"
    else:
        output_path = Path(output_path)
    
    # Build the export data
    export_data = export_for_gui(optimization_result, trip_packing, photo_recommendations)
    
    # Add trip context if available from trip_packing
    # (trip_packing doesn't have context, so this is optional)
    
    # Save to file
    output_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
    print(f"GUI JSON saved → {output_path}")
    
    return str(output_path)