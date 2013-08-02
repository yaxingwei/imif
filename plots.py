from core.modules.vistrails_module import Module, NotCacheable
from core.modules.basic_modules import String, Boolean
from packages.spreadsheet.basic_widgets import SpreadsheetCell
from packages.spreadsheet.spreadsheet_cell import QCellWidget
from matrix import Matrix
from matplotlib.transforms import Bbox
from matplotlib.widgets import  RectangleSelector
import matplotlib.cm as cm
import taylor_diagram
import numpy as np
import pylab
#from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
#import vtk
from PyQt4 import QtGui
#import vtk.util.numpy_support as VN
import matplotlib.pyplot as plt
from numpy import arange
from numpy.random import rand



################################################################################
class Coordinator(NotCacheable, Module):
    """
    Coordinator is intended to receive selected element in a view, and
    update all the registered views
    """

    def __init__(self):
        Module.__init__(self)
        
    def compute(self):
        self.modules = []
        
    def notifyModules(self, selectedIds):
        for mod in self.modules:
            mod.updateSelection(selectedIds);
        
    def register(self, module):
        if not module in self.modules:
            self.modules.append(module)
    
    def unregister(self, module):
        self.modules.remove(module)

################################################################################
class MplWidget(QCellWidget):
    """
    """
    
    def __init__(self, parent=None):
        """ MplWidget(parent: QWidget) -> MplWidget
        Initialize the widget with its central layout
        """
        
        QCellWidget.__init__(self, parent)
        centralLayout = QtGui.QVBoxLayout()
        self.setLayout(centralLayout)
        centralLayout.setMargin(0)
        centralLayout.setSpacing(0)
        
        # Create a new Figure Manager and configure it
        pylab.figure(str(self))
        self.figManager = pylab.get_current_fig_manager()
        self.figManager.toolbar.hide()
        self.layout().addWidget(self.figManager.window)
        
        self.inputPorts = None;
        self.selectedIds = []

    def deleteLater(self):
        """ deleteLater() -> None        
        Overriding PyQt deleteLater to free up resources
        
        """
        # Destroy the old one if possible
        if self.figManager:
            try:                    
                pylab.close(self.figManager.canvas.figure)
                # There is a bug in Matplotlib backend_qt4. It is a
                # wrong command for Qt4. Just ignore it and continue
                # to destroy the widget
            except:
                pass
            
            self.figManager.window.deleteLater()
        QCellWidget.deleteLater(self)
        
    def updateContents(self, inputPorts=None):
        """ updateContents(inputPorts: tuple) -> None
        Update the widget contents based on the input data
        """
        if inputPorts is not None: 
            self.inputPorts = inputPorts
            self.coord = self.inputPorts[0]
            if self.coord is not None: self.coord.register(self)
        
        # select our figure
        fig = pylab.figure(str(self))
        pylab.setp(fig, facecolor='w')

        
        # matplotlib plot
        self.draw()
        
        # Capture window into history for playback
        # Call this at the end to capture the image after rendering
        QCellWidget.updateContents(self, inputPorts)

    def draw(self, fig):
        raise NotImplementedError("Please Implement this method") 
    
    def onselect(self, eclick, erelease):
        raise NotImplementedError("Please Implement this method") 

################################################################################
class SeriesWidget(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)
        
    def draw(self):
        (self.coord, series, title, xlabel, ylabel, showLegend) = self.inputPorts
        colors = pylab.cm.jet(np.linspace(0,1, series.values.shape[0]))
        
        pylab.clf()
        pylab.title(title)
        
        ll = pylab.plot(series.values.T, linewidth=1)
        for pos, _ids in enumerate(series.ids):
            ll[pos].set_color(colors[pos])
            if _ids in self.selectedIds:
                ll[pos].set_linewidth(3)
        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)

        if showLegend:
            pylab.legend(pylab.gca().get_lines(),
                         series.labels,
                         numpoints=1, prop=dict(size='small'), loc='upper right')
                
        self.figManager.canvas.draw()
        self.rectSelector = RectangleSelector(pylab.gca(), self.onselect, drawtype='box', 
                                              rectprops=dict(alpha=0.4, facecolor='yellow'))
        self.rectSelector.set_active(True)

    def updateSelection(self, selectedIds):
        self.selectedIds = selectedIds
        self.updateContents();
    
    def onselect(self, eclick, erelease):
        pass
    
class SeriesPlot(SpreadsheetCell):
    """
    """
    my_namespace = 'views'
    name         = 'Series Plot'
    
    _input_ports = [('coord',      Coordinator,  False),
                    ('series',     Matrix,  False),
                    ('title',      String,  False),
                    ('xlabel',     String,  False),
                    ('ylabel',     String,  False),
                    ('showLegend', Boolean, False),
                    ]
    
    def compute(self):
        """ compute() -> None        
        """
        coord       = self.forceGetInputFromPort('coord', None)
        series      = self.getInputFromPort('series')
        title       = self.forceGetInputFromPort('title', '')
        xlabel      = self.forceGetInputFromPort('xlabel', '')
        ylabel      = self.forceGetInputFromPort('ylabel', '')
        showLegend  = self.forceGetInputFromPort('showLegend', True)
        self.displayAndWait(SeriesWidget, (coord, series, title, xlabel, ylabel, showLegend))

################################################################################
class DendrogramWidget(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)

    def draw(self):
        (_, matrix, title, xlabel, ylabel, method) = self.inputPorts
        
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        pylab.clf()
#        fig.set_frameon(False)
        
        pylab.title(title)
        Y = pdist(matrix.values)
        Z = linkage(Y, method=method)
        dendrogram(Z, 
                   labels=matrix.labels,
                   leaf_rotation=45)
        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)

        pylab.tight_layout(pad=0.4)

        self.figManager.canvas.draw()

    def updateSelection(self, selectedIds):
        pass

    def onselect(self, eclick, erelease):
        pass

class Dendrogram(SpreadsheetCell):
    """
    """
    my_namespace = 'views'
    name         = 'Dendrogram'
    
    _input_ports = [('coord',      Coordinator, False),
                    ('matrix',     Matrix,      False),
                    ('title',      String,      False),
                    ('xlabel',     String,      False),
                    ('ylabel',     String,      False),
                    ('method',     String,      False), # 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
                    ]
    
    def compute(self):
        """ compute() -> None        
        """
        coord       = self.forceGetInputFromPort('coord', None)
        matrix      = self.getInputFromPort('matrix')
        title       = self.forceGetInputFromPort('title', '')
        xlabel      = self.forceGetInputFromPort('xlabel', '')
        ylabel      = self.forceGetInputFromPort('ylabel', '')
        method      = self.forceGetInputFromPort('method', 'complete')
        self.displayAndWait(DendrogramWidget, (coord, matrix, title, xlabel, ylabel, method))

################################################################################
class TaylorDiagramWidget(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)
        
        self.markers = ['o','x','*',',','+','.','s','v','<','>','^','D','h','H','_','8',
                        'd',3,0,1,2,7,4,5,6,'1','3','4','2','|','x']

    def draw(self):
        (self.coord, self.stats, title, showLegend) = self.inputPorts
                
        stds, corrs = self.stats.values[:,0], self.stats.values[:,1]
        self.Xs = stds*corrs
        self.Ys = stds*np.sin(np.arccos(corrs))
        
        colors = pylab.cm.jet(np.linspace(0,1,len(self.stats.ids)))

        pylab.clf()
        fig = pylab.figure(str(self))
        dia = taylor_diagram.TaylorDiagram(stds[0], corrs[0], fig=fig, label=self.stats.labels[0])
        dia.samplePoints[0].set_color(colors[0])  # Mark reference point as a red star
        if self.stats.ids[0] in self.selectedIds: dia.samplePoints[0].set_markeredgewidth(3)
        
        # add models to Taylor diagram
        for i, (_id, stddev,corrcoef) in enumerate(zip(self.stats.ids[1:], stds[1:], corrs[1:])):
            label = self.stats.labels[i+1]
            size = 3 if _id in self.selectedIds else 1
            dia.add_sample(stddev, corrcoef,
                           marker='o', #self.markers[i],
                           ls='',
                           mfc=colors[i+1],
                           mew = size,
                           label=label
                           )

        # Add grid
        dia.add_grid()

        # Add RMS contours, and label them
        contours = dia.add_contours(levels=5, colors='0.5') # 5 levels in grey
        pylab.clabel(contours, inline=1, fontsize=10, fmt='%.1f')

        # Add a figure legend and title
        if showLegend:
            fig.legend(dia.samplePoints,
                       [ p.get_label() for p in dia.samplePoints ],
                       numpoints=1, prop=dict(size='small'), loc='upper right')
        fig.suptitle(title, size='x-large') # Figure title
        self.figManager.canvas.draw()
        
        self.rectSelector = RectangleSelector(pylab.gca(), self.onselect, drawtype='box', 
                                              rectprops=dict(alpha=0.4, facecolor='yellow'))
        self.rectSelector.set_active(True)

    def updateSelection(self, selectedIds):
        self.selectedIds = selectedIds
        self.updateContents();
    
    def onselect(self, eclick, erelease):
        if (self.coord is None): return

        left, bottom = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
        right, top = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)
        region = Bbox.from_extents(left, bottom, right, top)
        
        selectedIds = []
        for (x, y, idd) in zip(self.Xs, self.Ys, self.stats.ids):
            if region.contains(x, y):
                selectedIds.append(idd)
        self.coord.notifyModules(selectedIds)


class TaylorDiagram(SpreadsheetCell):
    """
    """
    my_namespace = 'views'
    name         = 'Taylor Diagram'
    
    _input_ports = [('coord',      Coordinator, False),
                    ('stats',      Matrix, False),
                    ('title',      String,  False),
                    ('showLegend', Boolean, False),
                    ]
    
    def compute(self):
        """ compute() -> None        
        """
        coord      = self.forceGetInputFromPort('coord', None)
        stats      = self.getInputFromPort('stats')
        title      = self.forceGetInputFromPort('title', '')
        showLegend = self.forceGetInputFromPort('showLegend', True)
        self.displayAndWait(TaylorDiagramWidget, (coord, stats, title, showLegend))

################################################################################
#class ParallelCoordinatesWidget(QCellWidget):
#    def __init__(self, parent=None):
#        QCellWidget.__init__(self, parent)
#        
#        centralLayout = QtGui.QVBoxLayout()
#        self.setLayout(centralLayout)
#        centralLayout.setMargin(0)
#        centralLayout.setSpacing(0)
#        self.columnRemoved = None   
#        self.view = vtk.vtkContextView()
#        self.widget = QVTKRenderWindowInteractor(self, 
#                                                 rw=self.view.GetRenderWindow(),
#                                                 iren=self.view.GetInteractor()
#                                                )
#        self.widget.AddObserver("RightButtonPressEvent", self.selfDeleteCallback)
#        

#        self.chart = vtk.vtkChartParallelCoordinates()
#        self.view.GetScene().AddItem(self.chart)

#        self.layout().addWidget(self.widget)

#        # Create a annotation link to access selection in parallel coordinates view
#        self.annotationLink = vtk.vtkAnnotationLink()
#        # If you don't set the FieldType explicitly it ends up as UNKNOWN (as of 21 Feb 2010)
#        # See vtkSelectionNode doc for field and content type enum values
#        self.annotationLink.GetCurrentSelection().GetNode(0).SetFieldType(1)     # Point
#        self.annotationLink.GetCurrentSelection().GetNode(0).SetContentType(4)   # Indices
#        # Connect the annotation link to the parallel coordinates representation
#        self.chart.SetAnnotationLink(self.annotationLink)
#        self.annotationLink.AddObserver("AnnotationChangedEvent", self.selectionCallback)
#        self.widget.AddObserver("LeftButtonPressEvent", self.selfAddCallback)
#        
#    def selfAddCallback(self, iren, strObj): 
#      
#        
#        for self.columnRemoved, attr in zip(self.matrix.values.T, self.matrix.attributes):
#            column = VN.numpy_to_vtk(self.columnRemoved.copy(), deep=True)
#            column.SetName(attr)
#            self.table.AddColumn(column)
#            self.table.Modified()
#            print 'Left Clicked', column 
#            self.chart.GetPlot(0).SetInput(self.table)
#        self.widget.Render()   
#        
#    def selfDeleteCallback(self, iren, strObj):
#      
##          
##           print "Event class name ", event.GetClassName(), "finished"

#      #!  print self.view.GetScene().GetTransform().TransformPoints()
##       for id in range(self.chart.GetNumberOfAxes()):
##         print iren.GetEventPosition()
##         print "Point0", self.chart.GetAxis(id).GetPoint1()[0], self.chart.GetAxis(id).GetPoint1()[1]
##         print "Point1", self.chart.GetAxis(id).GetPoint2()[0], self.chart.GetAxis(id).GetPoint2()[1]

##           evt= vtk.vtkContextMouseEvent() 
##           evt.SetInteractor(event)
##           evt.SetPos(event.GetEventPosition())
##           print event.GetEventPosition()
##           print evt.GetScenePos()
##         axisClicked = None  
#        
#        for id in range(self.chart.GetNumberOfAxes()):
#           if iren.GetEventPosition()[0] > self.chart.GetAxis(id).GetPoint1()[0]-10 and \
#              iren.GetEventPosition()[0] <= self.chart.GetAxis(id).GetPoint1()[0]+10:
#                self.columnRemoved =id
#                break
##                 axisClicked = self.chart.GetAxis(id)
##                 column = VN.numpy_to_vtk(id.copy(), deep=True)
#        if self.columnRemoved is not None:
#        
#                self.table.RemoveColumn(self.columnRemoved)
#                
#                print "Number of columns", self.table.GetNumberOfColumns()
##                 print "Removed axis id", id\
#                self.table.Modified()
#                self.chart.GetPlot(0).SetInput(self.table)
#                self.widget.Render()
##                   
##                 print y.GetEventPosition()
##                 print self.chart.GetAxis(id).GetPoint1()[1]
##                 print self.chart.GetAxis(id)
##                 print "clicked"
##             else:
##                print "not correct"
#               

#    def updateContents(self, inputPorts):
#        (self.coord, self.matrix) = inputPorts 
#        if self.coord is not None: self.coord.register(self)
#    
#        self.createTable(self.matrix)
#        self.widget.Initialize()
#        
#        # Capture window into history for playback
#        # Call this at the end to capture the image after rendering
#        QCellWidget.updateContents(self, inputPorts)
#        
#    def updateSelection(self, selectedIds):
#        if len(selectedIds)==0: return

#        Ids = VN.numpy_to_vtkIdTypeArray(np.array(selectedIds), deep=True)

#        node = vtk.vtkSelectionNode()
#        node.SetContentType(vtk.vtkSelectionNode.INDICES)
#        node.SetFieldType(vtk.vtkSelectionNode.POINT)
#        node.SetSelectionList(Ids)
#        
#        selection = vtk.vtkSelection()
#        selection.AddNode(node)
#        
#        self.annotationLink.SetCurrentSelection(selection)
#        self.widget.Render()
#        
#    def createTable(self, matrix):
#        self.table = vtk.vtkTable()
#        for col, attr in zip(matrix.values.T, matrix.attributes):
#            column = VN.numpy_to_vtk(col.copy(), deep=True)
#            column.SetName(attr)
#            self.table.AddColumn(column)
#        self.chart.GetPlot(0).SetInput(self.table)

#        min_ = matrix.values.min()-0.01
#        max_ = matrix.values.max()+0.01
#        for i in range(self.chart.GetNumberOfAxes()):
#            self.chart.GetAxis(i).SetRange(min_, max_)
#            self.chart.GetAxis(i).SetBehavior(vtk.vtkAxis.FIXED);
##            self.chart.GetAxis(i).SetPosition(vtk.vtkAxis.LEFT)
##            self.chart.GetAxis(i).GetTitleProperties().SetOrientation(30)

#    def selectionCallback(self, caller, event):
#        if self.coord is None: return
#        
#        annSel = self.annotationLink.GetCurrentSelection()
#        if annSel.GetNumberOfNodes() > 0:
#            idxArr = annSel.GetNode(0).GetSelectionList()
#            if idxArr.GetNumberOfTuples() > 0:
#                self.coord.unregister(self)
#                self.coord.notifyModules(VN.vtk_to_numpy(idxArr))
#                self.coord.register(self)

class ParallelCoordinates(SpreadsheetCell):
    """
    """
    my_namespace = 'views'
    name         = 'Parallel Coordinates'
    
    _input_ports = [('coord',       Coordinator, False),
                    ('matrix',      Matrix, False)
                    ]
    
    
    def compute(self):
        """ compute() -> None        
        """
        coord       = self.forceGetInputFromPort('coord', None)
        matrix      = self.getInputFromPort('matrix')
        #self.displayAndWait(ParallelCoordinatesWidget, (coord, matrix,))
        


class BarChartWidget(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)
          
    def draw(self):
        (self.coord, matrix, title, xlabel, ylabel, showLegend) = self.inputPorts
        #colors = pylab.cm.jet(np.linspace(0,1, series.values.shape[0]))
          
        pylab.clf()
        pylab.title(title)
          
        
 
        val = matrix.values

        print val

        print val.shape
 
        num = val.shape[0]

        
        pos = arange(num)+.5   
        width = 0.4
        colors = ['r', 'g', 'b', 'c']
        cs = [colors] * num
        cs[0] = 'c'

        #fig = plt.figure()
        #ax = fig.add_subplot(111)

        ax = pylab.gca()
        ax.bar(pos,val,width, 0.1)


        xTickMarks = ['Group'+str(i) for i in range(0,num)]
        ax.set_xticks(pos+width)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=45, fontsize=10)


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(title)

        #plt.show()

        self.figManager.canvas.draw()
        
 
          
         
class BarChart(SpreadsheetCell):



    """
BarChart
====================

This module is used to generate bar char from a 1D matrix


Input
----------

matrix: the 1D matrix to be visualized
title: title of the plot
xlable: label for horizontal axis
ylable: label for vertical axis
showLegend: whether to show legend

Output
-----------

The output is a barchart shown in the spreadsheet 

    """

      
    _input_ports = [('coord',      Coordinator,  False),
                    ('matrix',     Matrix,  False),
                    ('title',      String,  False),
                    ('xlabel',     String,  False),
                    ('ylabel',     String,  False),
                    ('showLegend', Boolean, False),
                    ]
      
    def compute(self):
        """ compute() -> None        
        """
          
        coord       = self.forceGetInputFromPort('coord', None)
        matrix      = self.getInputFromPort('matrix')
        title       = self.forceGetInputFromPort('title', '')
        xlabel      = self.forceGetInputFromPort('xlabel', '')
        ylabel      = self.forceGetInputFromPort('ylabel', '')
        showLegend  = self.forceGetInputFromPort('showLegend', True)
        self.displayAndWait(BarChartWidget, (coord, matrix, title, xlabel, ylabel, showLegend))
      
         
class HeatMapWidget(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)
          
    def draw(self):
        (self.coord, matrix, title, xlabel, ylabel, showLegend) = self.inputPorts
        #colors = pylab.cm.jet(np.linspace(0,1, series.values.shape[0]))
          
        pylab.clf()
        pylab.title(title)
          

        data = matrix.values

        column_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        row_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        #fig, ax = plt.subplots()


        ax = pylab.gca()

        rs = np.tril(data, 0)

        heatmap = ax.pcolor(rs, cmap=plt.cm.Reds)
 
        # heatmap = ax.pcolor(data, cmap=plt.cm.Reds)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        # plt.show()
        self.figManager.canvas.draw()
        

          
         
class HeatMap(SpreadsheetCell):


    """
HeatMap
====================

This module is used to generate heatmap from a pair-wise comparison matrix.


Input
----------

matrix: pair-wise comparison matrix to be visualized
title: title of the plot
xlable: label for horizontal axis
ylable: label for vertical axis
showLegend: whether to show legend

Output
-----------

The output is a heatmap shown in the spreadsheet 

    """


    my_namespace = 'views'
    name         = 'Heat Map'
      
    _input_ports = [('coord',      Coordinator,  False),
                    ('matrix',     Matrix,  False),
                    ('title',      String,  False),
                    ('xlabel',     String,  False),
                    ('ylabel',     String,  False),
                    ('showLegend', Boolean, False),
                    ]
      
    def compute(self):
        """ compute() -> None        
        """
          
        coord       = self.forceGetInputFromPort('coord', None)
        matrix      = self.getInputFromPort('matrix')
        title       = self.forceGetInputFromPort('title', '')
        xlabel      = self.forceGetInputFromPort('xlabel', '')
        ylabel      = self.forceGetInputFromPort('ylabel', '')
        showLegend  = self.forceGetInputFromPort('showLegend', True)
        self.displayAndWait(HeatMapWidget, (coord, matrix, title, xlabel, ylabel, showLegend))

