"""Shared finishing touches for every chart: no grid, one legend outside the right-most panel, proper sub- and superscripts."""
import re

import matplotlib.pyplot as plt
import matplotlib.text

# chemical and score names -> matplotlib mathtext (rendered as real subscripts)
NAMES = [(r"PM2\.5", r"PM$_{2.5}$"), (r"PM10", r"PM$_{10}$"), (r"\bNO2\b", r"NO$_{2}$"), (r"\bNOX\b", r"NO$_{x}$"),
         (r"\bSO2\b", r"SO$_{2}$"), (r"\bNH3\b", r"NH$_{3}$"), (r"\bOZONE\b", r"O$_{3}$"),
         (r"\bs_embed\b", r"$s_{embed}$"), (r"\bs_raw\b", r"$s_{raw}$")]
UNDERSCORE = re.compile(r"\b([A-Za-z]+)_([A-Za-z0-9.]+)\b")     # any other a_b -> a with subscript b


def pretty(text):
    """Chart text with proper subscripts instead of plain digits and underscores.

    What: "PM2.5" -> PM with subscript 2.5, "NO2" -> NO with subscript 2, "s_embed" -> s with
          subscript embed, any other word_word -> word with a subscript; µg/m³ keeps its superscript.
    How:  named patterns first; then every remaining a_b outside $...$ becomes a$_{b}$.
    """
    if not text or "$" in text:
        return text
    for pattern, new in NAMES:
        text = re.sub(pattern, new, text)
    parts = text.split("$")                    # even parts are plain text, odd parts are already maths
    for i in range(0, len(parts), 2):
        parts[i] = UNDERSCORE.sub(r"\1$_{\2}$", parts[i])
    return "$".join(parts)


def one_legend_outside(fig):
    """Merge every legend of the figure into one, placed outside the right-most panel.

    What: removes each panel and figure legend and draws one legend (each label once) to the
          right of the top right-most visible panel. A legend that sits in a switched-off panel
          (e.g. the empty 8th panel of a station grid) is already outside the data and is kept.
    How:  collects the handles and texts of the existing legends, removes them, adds a new
          legend with bbox_to_anchor just right of that panel.
    """
    visible = [ax for ax in fig.axes if ax.axison]
    found = [ax.get_legend() for ax in visible if ax.get_legend()] + list(fig.legends)
    if not found or not visible:
        return
    handles, labels = [], []
    for legend in found:
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            if text.get_text() not in labels:
                handles.append(handle)
                labels.append(text.get_text())
        legend.remove()
    right = max(visible, key=lambda ax: (round(ax.get_position().x1, 2), ax.get_position().y1))
    right.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False,
                 fontsize=8)


def table_figure(table, title, path, pct_cols=()):
    """A table as a picture: row labels = the table index, one column per table column.

    What: saves a plain PNG table (no colour coding); `pct_cols` are written as
          percentages, counts get thousands separators.
    How:  matplotlib's table inside an empty axes, saved with save_figure.
    """
    def cell(col, v):
        if col in pct_cols:
            return f"{v:.1f}%"
        return f"{int(v):,}" if float(v).is_integer() else f"{v:.1f}"

    text = [[cell(c, v) for c, v in row.items()] for _, row in table.iterrows()]
    labels = [" · ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index]
    fig, ax = plt.subplots(figsize=(2.2 * len(table.columns) + 3.5, 0.34 * len(table) + 1.2), facecolor="#fcfcfb")
    ax.axis("off")
    t = ax.table(cellText=text, rowLabels=labels, colLabels=list(table.columns),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.3)
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    save_figure(fig, path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def save_figure(fig, path, **kwargs):
    """Save a chart after the shared finishing touches (used by every chart instead of fig.savefig).

    What: grid off in every panel, one legend outside the right-most panel, sub- and
          superscripts in every text; saved with a tight box so the outside legend is kept.
    How:  matplotlib Text objects of the figure are rewritten with pretty() just before saving.
    """
    for ax in fig.axes:
        ax.grid(False)
    one_legend_outside(fig)
    fig.canvas.draw()                             # creates the tick labels that matplotlib makes at draw time
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            old = [t.get_text() for t in axis.get_ticklabels()]
            new = [pretty(t) for t in old]
            if new != old:                        # fix only text ticks (categories), leave number ticks alone
                axis.set_ticks(axis.get_ticklocs(), new)
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(pretty(text.get_text()))
    fig.savefig(path, bbox_inches="tight", **kwargs)
