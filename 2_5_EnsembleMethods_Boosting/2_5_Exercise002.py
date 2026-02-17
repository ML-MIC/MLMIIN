
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Markdown

# Define the cells (Markdown and Code)
external_cells = [
    nbformat.v4.new_markdown_cell("$$P(\\text{not included}) = \\left( \\dfrac{n - 1}{n}\\right)^n\\longrightarrow \\frac{1}{e}\\approx 0.37 \\approx 1/3$$"),
]

# Inject the Markdown cell properly
for cell in external_cells:
    if cell.cell_type == "markdown":
        display(Markdown(cell.source))  # Render Markdown properly
    else:
        InteractiveShell.instance().set_next_input(cell.source, replace=False)
