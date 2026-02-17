
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Markdown

# Define the cells (Markdown and Code)
external_cells = [
    nbformat.v4.new_markdown_cell("$$P(\text{not included}) = \left(1 - \frac{1}{n}\right)^n\tendsto \frac{1}{3}\approx 1/3$$"),
]

# Inject the Markdown cell properly
for cell in external_cells:
    if cell.cell_type == "markdown":
        display(Markdown(cell.source))  # Render Markdown properly
    else:
        InteractiveShell.instance().set_next_input(cell.source, replace=False)
