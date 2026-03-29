import pyqtgraph as pg
from coorx.ui import ProjectedView

app = pg.mkQApp()
view = ProjectedView()
view.show()
app.exec()