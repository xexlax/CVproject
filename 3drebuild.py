import vtk

import  os

label_path = './masks1/'
labelList = os.listdir(label_path)
pic_num = len(labelList)
# 定义渲染窗口、交互模式
aRender = vtk.vtkRenderer()
Renwin = vtk.vtkRenderWindow()
Renwin.AddRenderer(aRender)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(Renwin)

# 定义个图片读取接口
PNG_Reader = vtk.vtkPNGReader()

PNG_Reader.SetNumberOfScalarComponents(1)
PNG_Reader.SetFileDimensionality(3)  # 说明图像是三维的
 # 定义图像大小，本行表示图像大小为（512*512*240）


PNG_Reader.SetDataExtent(0, 200, 0,200, 0, pic_num-1)
 # 设置图像的存放位置
PNG_Reader.SetFilePrefix(label_path)
 # 设置图像前缀名字
 #表示图像前缀为数字（如：0.jpg）
PNG_Reader.SetFilePattern("%s%d.png")
PNG_Reader.Update()
PNG_Reader.SetDataByteOrderToLittleEndian()

# 计算轮廓的方法
contour = vtk.vtkMarchingCubes()
contour.SetInputConnection(PNG_Reader.GetOutputPort())
contour.ComputeNormalsOn()
contour.SetValue(0, 255)


mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contour.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.SetBackground([0.1, 0.1, 0.5])
renderer.AddActor(actor)

window = vtk.vtkRenderWindow()
window.SetSize(512, 512)
window.AddRenderer(renderer)


interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

# 开始显示
window.Render()
interactor.Initialize()
interactor.Start()
