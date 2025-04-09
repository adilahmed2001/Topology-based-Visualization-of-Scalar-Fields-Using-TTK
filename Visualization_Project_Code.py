import math
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets.vtk import VtkRemoteView

from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor, vtkRenderer
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

import vtk as vtk_lib
import topologytoolkit as ttk


from vtkmodules.vtkIOLegacy import vtkDataSetReader

is_3d = False

def set_is_data_set_3d(reader):
    
    global is_3d
    try:
        data = reader.GetOutput()
        extent = data.GetExtent() 
        dims = [extent[i+1] - extent[i] + 1 for i in range(0, 6, 2)]

        if dims[2] == 1: 
            is_3d = False
        else:
            is_3d = True
    except:
        is_3d = False

# Trame setup
server = get_server()
state, ctrl = server.state, server.controller

# VTK pipeline setup
reader = None
geometry_filter = vtkGeometryFilter()
triangle_filter = vtkDataSetTriangleFilter()
mapper = vtk_lib.vtkDataSetMapper()
actor = vtk_lib.vtkActor()
renderer = vtkRenderer()
renderer.SetBackground(0.7, 0.7, 0.7)
render_window = vtk_lib.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetOffScreenRendering(1)

interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor_style = vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interactor_style)
interactor.Initialize()

surface_actor = None
# Global variables
loaded_data = None
outlineData = vtk_lib.vtkOutlineFilter()
mapOutline = vtk_lib.vtkDataSetMapper()
outlineActor = vtk_lib.vtkActor()
geometry_actor = None

# Global actor lists
contour_tree_actors = []
morse_smale_actors = []
outline_actors = []

# State variables
state.visualization_types = ["Contour Tree", "Morse-Smale Complex"]
state.color_options = [
    "SolidColor",
    "SeparatrixType",
    "SeparatrixFunctionDifference",
    "SeparatrixFunctionMaximum",
    "SeparatrixFunctionMinimum",
    "SourceId",
    "SeparatrixId",
]
state.show_vtk_input = True
state.file_type = "vtk"
state.file_types = ["vtk"]
state.opacity = 0.7
state.opacity = 0.5
state.vtk_file = None
state.show_contour_tree = False
state.show_morse_smale = False
state.persistence = 0.01
state.color_array_name = "SolidColor"
state.error_message = ""
state.show_separatrices1 = True
state.show_separatrices2 = False
state.solid_color = "#FF0000"
state.point_size = 0.05

lut = vtk_lib.vtkLookupTable()

# Helper functions
def is_data_loaded():
    return loaded_data is not None and loaded_data.GetNumberOfPoints() > 0

def reset_camera():
    if is_data_loaded():
        renderer.ResetCamera()
        ctrl.view_reset_camera()
        ctrl.view_update()
    else:
        state.error_message = "No data loaded to reset camera."

def hsv_to_rgb(hsv):
    """Convert HSV to RGB."""
    h, s, v = hsv
    h = h / 60.0
    i = math.floor(h) % 6
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if i == 0:
        return [v, t, p]
    elif i == 1:
        return [q, v, p]
    elif i == 2:
        return [p, v, t]
    elif i == 3:
        return [p, q, v]
    elif i == 4:
        return [t, p, v]
    else:
        return [v, p, q]

def make_lut(color_scheme, lut):
    
    if color_scheme == "critical_type":
        lut.SetNumberOfTableValues(4)  
        lut.Build()
        # Assign colors to critical types
        lut.SetTableValue(0, 0, 0, 1, 1)# 0: Minimum (blue)
        lut.SetTableValue(1, 1, 0.85, 0.2, 0.7) # 1: 1-Saddle (yellow)
        lut.SetTableValue(2, 0.5, 1, 0.5, 0.8)# 2: 2-Saddle (orange)
        lut.SetTableValue(3, 1, 0, 0, 1)# 3: Maximum (red)
    else:
        nc = 256
        lut.SetNumberOfTableValues(nc)
        lut.Build()

        if color_scheme == "rainbow":
            sMin = 0.
            sMax = 1.
            
            hsv = [0.0,1.0,1.0]
            
            for i in range(0, nc):
                s = float(i) / nc 
                hsv[0] = 240. - 240. *(s-sMin)/(sMax-sMin)
                rgb = hsv_to_rgb(hsv)
                rgb.append(1.0)  
                lut.SetTableValue(i, *rgb)

        elif color_scheme == "bwr":
            sMin = 0.
            sMax = 1.

            hsv = [0.0, 1.0, 1.0]

            for i in range(0, nc):
                s = float(i) / nc 
                if s <= 1/2 :
                    hsv[0] = 240.
                    hsv[1] = 1. - 2 * s
                else :
                    hsv[0] = 0.
                    hsv[1] = 2 * s - 1.
                rgb = hsv_to_rgb(hsv)
                rgb.append(1.0)  
                lut.SetTableValue(i, *rgb)     

        elif color_scheme == "heatmap":
            sMin = 0.
            sMax = 1.

            for i in range(0, nc):
                rgb = [0., 0. , 0.]
                s = float(i) / nc 
                if s <= 1/3 :
                    rgb[0] = 3 * s
                elif s > 1/3 and s <= 2/3 :
                    rgb[0] = 1.
                    rgb[1] = 3 * s - 1.
                else:
                    rgb[0], rgb[1] =  1., 1.
                    rgb[2] = 3 * s - 2.
                    
                rgb.append(1.0)  
                lut.SetTableValue(i, *rgb)     

        elif color_scheme == "grayscale":
            sMin = 0.
            sMax = 1.

            for i in range(0, nc):
                rgb = [0., 0. , 0.]
                s = float(i) / nc 
                rgb[0], rgb[1], rgb[2] = s, s, s
                rgb.append(1.0)  
                lut.SetTableValue(i, *rgb)     

        elif color_scheme == "saturation":
            for i in range(nc):
                s = float(i) / nc  

                hue = 120 
                saturation = s  
                value = 0.5  

                rgb = hsv_to_rgb([hue, saturation, value])

                rgb.append(1.0) 
                lut.SetTableValue(i, *rgb)
        
        elif color_scheme == "intensity":
            for i in range(nc):
                s = float(i) / nc 

                intensity = s 
                rgb = [0, intensity, 0] 
                
                rgb.append(1.0) 
                lut.SetTableValue(i, *rgb)

    return lut


###############################
### DATASET LOADING SECTION ###
###############################
def remove_geometry_actor():
    global geometry_actor
    if geometry_actor:
        renderer.RemoveActor(geometry_actor)
        geometry_actor = None

@state.change("file_type")
def update_file_inputs(file_type, **kwargs):
    if file_type == "vtk":
        state.show_vtk_input = True

@state.change("vtk_file")
def update_vtk_file(vtk_file, **kwargs):
    if state.file_type == "vtk":
        update_file(vtk_file)

def update_file(file_data, **kwargs):
    global loaded_data, reader, outlineData, mapOutline, outlineActor, is_3d

    if not file_data:
        state.error_message = "Please select a file."
        return

    try:
        if state.file_type == "vtk":
            reader = vtkDataSetReader()
            if isinstance(file_data, dict) and 'content' in file_data:
                reader.ReadFromInputStringOn()
                reader.SetInputString(file_data['content'])
            else:
                raise ValueError("Invalid file_data format for VTK")

        reader.Update()
        set_is_data_set_3d(reader)
        print(is_3d)
        loaded_data = reader.GetOutput()

        if not loaded_data or loaded_data.GetNumberOfPoints() == 0:
            state.error_message = "Failed to load the file. Please check if it's a valid file."
            return

        if state.file_type == "vtk":
            loaded_data.GetPointData().SetActiveScalars('s')

        remove_geometry_actor()

        for actor in contour_tree_actors:
            renderer.RemoveActor(actor)

        for actor in morse_smale_actors:
            renderer.RemoveActor(actor)

        state.show_contour_tree = False
        state.show_morse_smale = False


        update_iso_slider_range()

        mapper.SetInputData(loaded_data)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray('s')  
        scalar_range = loaded_data.GetPointData().GetArray('s').GetRange()
        mapper.SetScalarRange(scalar_range)
        mapper.SetLookupTable(lut)
        geometry_actor = vtk_lib.vtkActor()
        
        if is_3d : 
            geometry_actor.GetProperty().SetOpacity(0)
        else:
            make_lut(state.color_scheme, lut)
            geometry_actor.GetProperty().SetOpacity(1)

        geometry_actor.SetMapper(mapper)
        renderer.AddActor(geometry_actor)

        setup_outline()
        reset_camera()
        state.error_message = ""

        ctrl.view_update()

    except Exception as e:
        state.error_message = f"Error loading file: {str(e)}"

@state.change("color_scheme")
def update_color_scheme(color_scheme, **kwargs):
    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        return
    
    make_lut(color_scheme, lut)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(loaded_data.GetScalarRange())
    ctrl.view_update()

def setup_outline():
    global outlineData, mapOutline, outlineActor, outline_actors

    if reader is None or reader.GetOutput() is None:
        print("Error: Reader is not initialized or no data loaded.")
        state.error_message = "Error: Reader is not initialized or no data loaded."
        return
    
    renderer.RemoveActor(outlineActor)
    outlineData.SetInputConnection(reader.GetOutputPort())
    outlineData.Update()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())
    mapOutline.Update()
    outlineActor.SetMapper(mapOutline)
    colors = vtk_lib.vtkNamedColors()
    outlineActor.GetProperty().SetColor(colors.GetColor3d("Black"))
    outlineActor.GetProperty().SetLineWidth(2.0)
    renderer.AddActor(outlineActor)
    outline_actors.append(outlineActor)

def setup_isosurface():
    global surface_actor

    remove_surface_actor()

    extract_isosurface(state.iso_value, state.iso_alpha)

def extract_isosurface(iso_value, opacity):
    global surface_actor, is_3d
    if (not is_data_loaded()) and (not is_3d):
        return
    
    if not loaded_data.IsA("vtkImageData"):
        return

    iso_surface_filter = vtk_lib.vtkMarchingCubes()
    iso_surface_filter.SetInputData(loaded_data)
    iso_surface_filter.Update()

    iso_surface_filter.SetValue(0, iso_value)
 
    iso_surface_stripper = vtk_lib.vtkStripper()
 
    iso_surface_stripper.SetInputConnection(iso_surface_filter.GetOutputPort())
    iso_surface_stripper.Update()
 
    iso_surface_mapper = vtk_lib.vtkPolyDataMapper()
    iso_surface_mapper.SetInputConnection(iso_surface_stripper.GetOutputPort())
    iso_surface_mapper.SetScalarRange(loaded_data.GetScalarRange())
    iso_surface_mapper.ScalarVisibilityOff()
 
    surface_actor = vtk_lib.vtkActor()
    surface_actor.SetMapper(iso_surface_mapper)
 
    surface_actor.GetProperty().SetDiffuseColor(0.83, 0.66, 0.47)
    surface_actor.GetProperty().SetOpacity(opacity)
    surface_actor.GetProperty().SetSpecular(.3)
    surface_actor.GetProperty().SetSpecularPower(20)
 
    renderer.AddActor(surface_actor)
   
    update_surface_visibility()
 
def remove_surface_actor():
    global surface_actor
   
    renderer.RemoveActor(surface_actor)
 
def update_surface_visibility():
    if surface_actor:
        surface_actor.SetVisibility(state.isosurface_visible)
 
@state.change("isosurface_visible")
def update_isosurface_visibility(isosurface_visible, **kwargs):
    update_surface_visibility()
    ctrl.view_update()
 
@state.change("iso_value")
def update_iso_contour(iso_value, **kwargs):
    if not is_data_loaded():
        state.error_message = "Please load a data file first."
        return
   
    remove_surface_actor()
    extract_isosurface(iso_value, state.iso_alpha)
    ctrl.view_update()
 
@state.change("iso_alpha")
def update_transparency(iso_alpha, **kwargs):
    if not is_data_loaded() or not surface_actor:
        return
    #TODO: Task #3 update the transparency of the iso surface
 
    remove_surface_actor()
    extract_isosurface(state.iso_value, iso_alpha)
 
    ctrl.view_update()
 
def update_iso_slider_range():
    if is_data_loaded():
        range = loaded_data.GetScalarRange()
 
        state.iso_slider_min = range[0]
        state.iso_slider_max = range[1]
        state.iso_value = range[0]
 
    else:
        state.iso_slider_min = 0
        state.iso_slider_max = 100
        state.iso_value = 0

###################################
### TTK VISUALIZATION FUNCTIONS ###
###################################
def visualize_contour_tree():
    global contour_tree_actors
    contour_tree_actors = []

    if not is_data_loaded():
        state.error_message = "No data loaded."
        return

    loaded_data.GetPointData().SetActiveScalars('s')

    if not loaded_data.IsA("vtkUnstructuredGrid"):
        triangle_filter = vtk_lib.vtkDataSetTriangleFilter()
        triangle_filter.SetInputData(loaded_data)
        triangle_filter.Update()
        unstructured_data = triangle_filter.GetOutput()
    else:
        unstructured_data = loaded_data

    persistence_diagram = ttk.ttkPersistenceDiagram()
    persistence_diagram.SetInputData(unstructured_data)
    persistence_diagram.SetInputArrayToProcess(0, 0, 0, 0, 's')
    persistence_diagram.Update()
    persistence_output = persistence_diagram.GetOutput()

    threshold = vtk_lib.vtkThreshold()
    threshold.SetInputData(persistence_output)
    threshold.SetInputArrayToProcess(0, 0, 0, 1, 'Persistence') 
    threshold.SetLowerThreshold(state.persistence) 
    threshold.Update()
    filtered_pairs = threshold.GetOutput()

    pairs_to_grid = vtk_lib.vtkDataSetTriangleFilter()
    pairs_to_grid.SetInputData(filtered_pairs)
    pairs_to_grid.Update()
    persistent_critical_points = pairs_to_grid.GetOutput()

    simplification = ttk.ttkTopologicalSimplification()
    simplification.SetInputDataObject(0, unstructured_data)  
    simplification.SetInputDataObject(1, persistent_critical_points) 
    simplification.SetInputArrayToProcess(0, 0, 0, 0, 's')
    simplification.Update()
    simplified_data = simplification.GetOutput()

    contour_tree = ttk.ttkContourTree()
    contour_tree.SetInputDataObject(simplified_data)
    contour_tree.SetInputArrayToProcess(0, 0, 0, 0, 's')
    contour_tree.Update()

    contour_tree_arcs = contour_tree.GetOutput(1)  
    contour_tree_nodes = contour_tree.GetOutput(0) 

    arc_geometry_filter = vtk_lib.vtkGeometryFilter()
    arc_geometry_filter.SetInputData(contour_tree_arcs)
    arc_geometry_filter.Update()
    arc_polydata = arc_geometry_filter.GetOutput()

    arc_mapper = vtk_lib.vtkPolyDataMapper()
    arc_mapper.SetInputData(arc_polydata)
    arc_mapper.ScalarVisibilityOn()
    arc_mapper.SetScalarModeToUseCellFieldData()

    lut = make_lut(state.color_scheme, vtk_lib.vtkLookupTable())
    arc_mapper.SetLookupTable(lut)

    if contour_tree_arcs.GetCellData().HasArray("SegmentationId"):
        arc_mapper.SelectColorArray("SegmentationId")
        arc_mapper.SetScalarRange(contour_tree_arcs.GetCellData().GetArray("SegmentationId").GetRange())

    arc_actor = vtk_lib.vtkActor()
    arc_actor.SetMapper(arc_mapper)
    arc_actor.GetProperty().SetLineWidth(3)
    arc_actor.GetProperty().SetOpacity(1.0)
    renderer.AddActor(arc_actor)
    contour_tree_actors.append(arc_actor)

    sphere_source = vtk_lib.vtkSphereSource()
    sphere_source.SetRadius(state.point_size)
    sphere_source.SetThetaResolution(16)
    sphere_source.SetPhiResolution(16)

    glyph = vtk_lib.vtkGlyph3D()
    glyph.SetSourceConnection(sphere_source.GetOutputPort())
    glyph.SetInputData(contour_tree_nodes)
    glyph.SetScaleModeToDataScalingOff()
    glyph.SetScaleFactor(1.0)  
    glyph.Update()

    glyph_mapper = vtk_lib.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())
    glyph_mapper.ScalarVisibilityOn()
    glyph_mapper.SetColorModeToMapScalars()
    glyph_mapper.SetScalarModeToUsePointFieldData()

    lut = make_lut(state.color_scheme, vtk_lib.vtkLookupTable())
    glyph_mapper.SetLookupTable(lut)

    if contour_tree_nodes.GetPointData().HasArray("CriticalType"):
        glyph_mapper.SelectColorArray('CriticalType')
        glyph_mapper.SetScalarRange(contour_tree_nodes.GetPointData().GetArray('CriticalType').GetRange())

    glyph_actor = vtk_lib.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetOpacity(1.0)
    renderer.AddActor(glyph_actor)
    contour_tree_actors.append(glyph_actor)

    ctrl.view_update()


def visualize_morse_smale_complex():
    global morse_smale_actors
    morse_smale_actors = []

    if not is_data_loaded():
        state.error_message = "No data loaded."
        return

    loaded_data.GetPointData().SetActiveScalars('s')

    if not loaded_data.IsA("vtkUnstructuredGrid"):
        triangle_filter = vtk_lib.vtkDataSetTriangleFilter()
        triangle_filter.SetInputData(loaded_data)
        triangle_filter.Update()
        unstructured_data = triangle_filter.GetOutput()
    else:
        unstructured_data = loaded_data

    persistence_diagram = ttk.ttkPersistenceDiagram()
    persistence_diagram.SetInputData(unstructured_data)
    persistence_diagram.SetInputArrayToProcess(0, 0, 0, 0, 's')
    persistence_diagram.Update()
    persistence_output = persistence_diagram.GetOutput()

    if persistence_output.GetNumberOfCells() == 0 or not persistence_output.GetCellData().HasArray("Persistence"):
        state.error_message = "No valid persistence data found. Please adjust the input data."
        return
    
    threshold = vtk_lib.vtkThreshold()
    threshold.SetInputData(persistence_output)
    threshold.SetInputArrayToProcess(0, 0, 0, 1, 'Persistence')  
    threshold.SetLowerThreshold(state.persistence)  
    threshold.Update()
    filtered_pairs = threshold.GetOutput()

    pairs_to_grid = vtk_lib.vtkDataSetTriangleFilter()
    pairs_to_grid.SetInputData(filtered_pairs)
    pairs_to_grid.Update()
    persistent_critical_points = pairs_to_grid.GetOutput()

    simplification = ttk.ttkTopologicalSimplification()
    simplification.SetInputDataObject(0, unstructured_data)  
    simplification.SetInputDataObject(1, persistent_critical_points)  
    simplification.SetInputArrayToProcess(0, 0, 0, 0, 's')
    simplification.Update()
    simplified_data = simplification.GetOutput()

    msc = ttk.ttkMorseSmaleComplex()
    msc.SetInputDataObject(simplified_data)
    msc.SetInputArrayToProcess(0, 0, 0, 0, 's')
    msc.SetComputeCriticalPoints(True)
    msc.SetComputeAscendingSeparatrices1(True)
    msc.SetComputeDescendingSeparatrices1(True)
    msc.SetComputeAscendingSeparatrices2(True)
    msc.SetComputeDescendingSeparatrices2(True)
    msc.SetComputeSaddleConnectors(True)
    msc.Update()

    critical_points = msc.GetOutput(0)  
    separatrices_1 = msc.GetOutput(1) 
    separatrices_2 = msc.GetOutput(2)  

    color_lut = make_lut(state.color_scheme, vtk_lib.vtkLookupTable())

    critical_points_mapper = vtk_lib.vtkDataSetMapper()
    critical_points_mapper.SetInputData(critical_points)
    critical_points_mapper.ScalarVisibilityOn()
    critical_points_mapper.SetScalarModeToUsePointFieldData()

    if critical_points.GetPointData().GetArray("CellDimension"):
        scalar_range = critical_points.GetPointData().GetArray("CellDimension").GetRange()
        critical_points_mapper.SetScalarRange(scalar_range)
        critical_points_mapper.SelectColorArray("CellDimension")
    critical_points_mapper.SetLookupTable(color_lut)
    critical_points_actor = vtk_lib.vtkActor()
    critical_points_actor.SetMapper(critical_points_mapper)
    critical_points_actor.GetProperty().SetPointSize(2)

    renderer.AddActor(critical_points_actor)
    morse_smale_actors.append(critical_points_actor)

    color_lut = make_lut(state.color_scheme, vtk_lib.vtkLookupTable())

    if state.show_separatrices1 and separatrices_1.GetCellData().GetArray("SeparatrixType"):
        separatrices1_mapper = vtk_lib.vtkDataSetMapper()
        separatrices1_mapper.SetInputData(separatrices_1)
        separatrices1_mapper.ScalarVisibilityOn()
        separatrices1_mapper.SetScalarModeToUseCellFieldData()
        scalar_range = separatrices_1.GetCellData().GetArray("SeparatrixType").GetRange()
        separatrices1_mapper.SelectColorArray("SeparatrixType") 
        separatrices1_mapper.SetScalarRange(scalar_range)
        separatrices1_mapper.SetLookupTable(color_lut)
        separatrices1_actor = vtk_lib.vtkActor()
        separatrices1_actor.SetMapper(separatrices1_mapper)
        separatrices1_actor.GetProperty().SetLineWidth(0.5)

        renderer.AddActor(separatrices1_actor)
        morse_smale_actors.append(separatrices1_actor)

    if state.show_separatrices2:
        separatrices2_mapper = vtk_lib.vtkDataSetMapper()
        separatrices2_mapper.SetInputData(separatrices_2)

        if state.color_array_name == "SolidColor":
            separatrices2_mapper.ScalarVisibilityOff()
            separatrices2_actor = vtk_lib.vtkActor()
            separatrices2_actor.SetMapper(separatrices2_mapper)
            
            r, g, b = [int(state.solid_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
            separatrices2_actor.GetProperty().SetColor(r, g, b)

            separatrices2_actor.GetProperty().SetOpacity(state.opacity)
        else:
            if separatrices_2.GetCellData().HasArray(state.color_array_name):
                separatrices2_mapper.ScalarVisibilityOn()
                separatrices2_mapper.SetScalarModeToUseCellFieldData()
                separatrices2_mapper.SelectColorArray(state.color_array_name)

                scalar_range = separatrices_2.GetCellData().GetArray(state.color_array_name).GetRange()
                separatrices2_mapper.SetScalarRange(scalar_range)

                color_lut = make_lut(state.color_scheme, vtk_lib.vtkLookupTable())
                separatrices2_mapper.SetLookupTable(color_lut)

            separatrices2_actor = vtk_lib.vtkActor()
            separatrices2_actor.SetMapper(separatrices2_mapper)
            separatrices2_actor.GetProperty().SetOpacity(state.opacity)

        renderer.AddActor(separatrices2_actor)
        morse_smale_actors.append(separatrices2_actor)

    ctrl.view_update()

def reset_contour_tree_state():
    state.opacity = 0.7
    state.persistence = 0.01 
    state.point_size = 0.05

@state.change("show_contour_tree")
def handle_morse_smale_toggle(show_contour_tree, **kwargs):
    if not show_contour_tree:
        reset_contour_tree_state()  

def reset_morse_smale_state():
    state.show_separatrices1 = True
    state.show_separatrices2 = False
    state.color_array_name = "SolidColor" 
    state.opacity = 0.7
    state.solid_color = "#FF0000"  
    state.persistence = 0.01

@state.change("show_morse_smale")
def handle_morse_smale_toggle(show_morse_smale, **kwargs):
    if not show_morse_smale:
        reset_morse_smale_state()  

@state.change("show_contour_tree", "persistence", "opacity", "point_size", "color_scheme")
def update_contour_tree_visualization(show_contour_tree, **kwargs):
    global contour_tree_actors
    for actor in contour_tree_actors:
        renderer.RemoveActor(actor)
    contour_tree_actors = []
    if show_contour_tree:
        visualize_contour_tree()
        renderer.Render()
        ctrl.view_update()
    setup_outline()
    reset_camera()

@state.change("show_morse_smale", "show_separatrices1", "show_separatrices2", "persistence", "color_array_name", "opacity", "solid_color", "color_scheme")
def update_morse_smale_visualization(show_morse_smale, **kwargs):
    global morse_smale_actors
    for actor in morse_smale_actors:
        renderer.RemoveActor(actor)
    morse_smale_actors = []
    if show_morse_smale:
        visualize_morse_smale_complex()
        renderer.Render()
        ctrl.view_update()
    setup_outline()
    reset_camera()

########################
### UI SETUP SECTION ###
########################

def setup_ui():
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("COSC 6344 Visualization – Project")

        with layout.toolbar:
            vuetify.VToolbarTitle("Topology-based Visualization (TTK)")
            vuetify.VSpacer()
            vuetify.VBtn(
                "Reset Camera",
                prepend_icon="mdi-crop-free",
                click=reset_camera,
            )

        with layout.drawer:
            with vuetify.VForm():
                with vuetify.VContainer(fluid=True):
                    vuetify.VSelect(
                        v_model="file_type",
                        items=("file_types",),
                        label="File Type",
                        variant="outlined",
                        density="compact",
                    )
                    vuetify.VFileInput(
                        v_model="vtk_file",
                        label="Select VTK File",
                        variant="outlined",
                        density="compact",
                        accept=".vtk",
                        v_show="show_vtk_input",
                    )
                    vuetify.VSelect(
                        v_model=("color_scheme", "bwr"),
                        items=("color_schemes", ["critical_type", "bwr", "rainbow", "heatmap", "grayscale", "intensity", "saturation", ]),
                        label="Color Scheme",
                        variant="outlined",
                        density="compact",
                    )
                    vuetify.VDivider()

                    vuetify.VCheckbox(
                        v_model="show_contour_tree",
                        label="Contour Tree",
                    )
                    vuetify.VSlider(
                        v_model=("point_size", 0.05),
                        min=0.01,
                        max=5,
                        step=0.01,
                        label="Set Point Size",
                        thumb_label="always",
                    )

                    vuetify.VDivider()
                    vuetify.VCheckbox(
                        v_model="show_morse_smale",
                        label="Morse-Smale Complex",
                    )
                    vuetify.VSwitch(
                        v_model="show_separatrices1",
                        label="Show Separatrices 1",
                        density="compact",
                    )
                    vuetify.VSwitch(
                        v_model="show_separatrices2",
                        label="Show Separatrices 2",
                        density="compact",
                    )


                    vuetify.VSelect(
                        v_model="color_array_name",
                        items = ("color_options", ), 
                        label="Coloring Array",
                        variant="outlined",
                        density="compact",
                    )

                    vuetify.VColorPicker(
                        v_model=("solid_color", "#FF0000"),
                        mode="hex",
                        hide_mode_switch=True,
                        hide_inputs=True,
                        label="MS Solid Color",
                        density="compact",
                        style="margin-bottom: 16px;"
                    )       
                
                    vuetify.VDivider()

                    vuetify.VSlider(
                        v_model=("opacity", 0.5),
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        label="Set Opacity",
                        thumb_label="always",
                    )


                    vuetify.VSlider(
                        v_model="persistence",
                        min=0.,
                        max=5,
                        step=0.001,
                        label="Persistence Threshold",
                        thumb_label="always",
                    )

                    vuetify.VRangeSlider(
                    v_model=("scalar_range", [0, 100]),
                    min=("iso_slider_min", 0),
                    max=("iso_slider_max", 100),
                    label="Scalar Range",
                    thumb_label="always",
                    density="compact",
                )
 
               
 
                    with vuetify.VRow(align="center", dense=True):
                        with vuetify.VCol(cols=1): 
                            vuetify.VCheckbox(v_model=("isosurface_visible", False))
                        with vuetify.VCol(cols=11):
                            vuetify.VSlider(
                                v_model=("iso_value", 0),
                                min=("iso_slider_min", 0),
                                max=("iso_slider_max", 100),
                                label="Iso Value",
                                thumb_label="always",
                                density="compact",
                            )
                    vuetify.VSlider(
                        v_model=("iso_alpha", 1),
                        min=("iso_alpha_min", 0),
                        max=("iso_alpha_max", 1),
                        label="Opacity",
                        thumb_label="always",
                        density="compact",
                    )                
 
        with layout.content:
            with vuetify.VContainer(
                fluid=True,
                classes="pa-0 fill-height",
            ):
               
                vuetify.VAlert(
                    v_if="error_message",
                    type="error",
                    prominent=True,
                    text=("error_message",),
                )
              
                view = VtkRemoteView(render_window)
                ctrl.on_server_ready.add(view.update)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera



if __name__ == "__main__":
    setup_ui()
    server.start(port=7654)

