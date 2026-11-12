"""Visual check of the gallery shapes -> results/figures/shapes_grid.png"""

from tinycast.data.shapes import gallery
from tinycast.eval.plots import ROOT, shapes_grid

if __name__ == "__main__":
    out = ROOT / "results" / "figures" / "shapes_grid.png"
    shapes_grid(gallery(), out)
    print(out)
