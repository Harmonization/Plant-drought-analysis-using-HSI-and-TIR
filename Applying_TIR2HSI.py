from PyQt5 import QtWidgets as QW, QtGui, QtCore
import sys, re, numpy as np, pandas as pd, skimage.exposure as exposure, matplotlib.pyplot as plt, pysptools.util as util, cv2
from matplotlib.colors import LinearSegmentedColormap as LSCM
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NTool
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Window(QW.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("HSI + TIR")
        
        self.fig_l, self.ax_l = plt.subplots(); self.ax_l.axis('off')
        self.fig_r, self.ax_r = plt.subplots(); self.ax_r.axis('off')
        self.fig_l.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.fig_r.subplots_adjust(left=.02, right=.98, bottom=0, top=1)
        self.canvas_l = FigureCanvasQTAgg(self.fig_l); self.canvas_r = FigureCanvasQTAgg(self.fig_r)
        toolbar_l = NTool(self.canvas_l, self); toolbar_r = NTool(self.canvas_r, self)
        
        self.cbar = 0; self.patch = []; self.rect_click = False; self.scatter_lst = []
        self.point_dct = {'hsi': [], 'tir': [], 'hsi_img': [], 'tir_img': []}; self.homo_flag = False
        
        def Clear():
            if self.homo_flag: self.ax_l.clear(); self.ax_r.clear(); self.ax_l.axis('off'); self.ax_r.axis('off')
                
            self.rect = [0, 0, self.rgb.shape[0], self.rgb.shape[1]] # x1, y1, x2, y2
            if not self.homo_flag: [el.remove() for el in self.point_dct['hsi_img']]; [el.remove() for el in self.point_dct['tir_img']]
            self.point_dct = {'hsi': [], 'tir': [], 'hsi_img': [], 'tir_img': []}
            if self.patch: self.patch.pop().remove()
            
            if self.homo_flag: Plot_RGB(self.rgb); self.tir = self.full_tir; Plot_TIR(); self.homo_flag = False
            else: self.canvas_l.draw(); self.canvas_r.draw()
        
        def Open_HSI():
            path = QW.QFileDialog.getOpenFileName(self, "Open File", '', '*.hdr')[0]
            if path:
                data, _ = util.load_ENVI_file(path)
                self.hsi = np.rot90(data, 3)
                self.rgb = exposure.rescale_intensity(self.hsi[:, :, (70, 52, 19)], out_range=(0,5))
                Clear(); Plot_RGB(self.rgb)
        
        def Open_TIR():
            path = QW.QFileDialog.getOpenFileName(self, "Open File", '', '*.xlsx')[0]
            if path:
                self.tir = self.full_tir = pd.read_excel(path, header = None).to_numpy()
                tir_min.setText(str(np.around(np.percentile(self.tir, 1), 2))); tir_max.setText(str(np.around(np.percentile(self.tir, 99), 2)))
                Clear(); Plot_TIR()
        
        def Plot_RGB(img):
            ax_img_l = self.ax_l.imshow(img)
            self.canvas_l.draw()
    
        def Plot_TIR():
            ax_img_r = self.ax_r.imshow(self.tir, cmap=LSCM.from_list("map", ["lawngreen", "yellow", "red"]), 
                                        vmin=float(tir_min.text()), vmax=tir_max.text())
            if self.cbar: self.cbar.remove()
            
            divider = make_axes_locatable(self.ax_r)
            cax = divider.append_axes("bottom", size="7%", pad='5%')
            
            self.cbar = plt.colorbar(ax_img_r, cax=cax, orientation="horizontal")
            self.canvas_r.draw()
            
        
        open_hsi = QW.QPushButton('Open HSI'); open_hsi.clicked.connect(Open_HSI)
        open_tir = QW.QPushButton('Open TIR'); open_tir.clicked.connect(Open_TIR)
        
        tir_min = QW.QLineEdit(''); tir_min.returnPressed.connect(Plot_TIR); tir_min.setFixedSize(60, 35)
        tir_max = QW.QLineEdit(''); tir_max.returnPressed.connect(Plot_TIR); tir_max.setFixedSize(60, 35)
        
        roi = QW.QRadioButton('Выбрать область'); point = QW.QRadioButton('Совмещение точек'); roi.setChecked(True)
        
        def Back(key='hsi'):
            if self.point_dct[key]: 
                self.point_dct[key].pop(); self.point_dct[f'{key}_img'].pop().remove()
                if key=='hsi': self.canvas_l.draw()
                else: self.canvas_r.draw()
            
        def Homography():
            x1, y1, x2, y2 = map(round, self.rect)
            self.pot = self.hsi[y1: y2, x1: x2]
            p_tir = np.array(self.point_dct['tir']); p_hsi = np.array(self.point_dct['hsi'])
            H = cv2.findHomography(p_tir, p_hsi)[0]
            self.tir = cv2.warpPerspective(self.full_tir, H, self.hsi.shape[:2])[y1: y2, x1: x2]
            
            self.ax_l.clear(); self.ax_r.clear(); self.ax_l.axis('off'); self.ax_r.axis('off')
            [el.remove() for el in self.point_dct['hsi_img']]; [el.remove() for el in self.point_dct['tir_img']]
            if self.patch: self.patch.pop().remove()
            Plot_RGB(self.pot[:, :, (70, 52, 19)]); Plot_TIR()
            self.homo_flag = True
            
        def Save():
            save_path = ''
            save_path = QW.QFileDialog.getSaveFileName(self, "Save File", '', '.npy')[0]
            if save_path:
                cube = np.dstack([self.pot, self.tir[:, :, np.newaxis]])
                np.save(save_path, cube)
        
        b_back_l = QW.QPushButton('Назад'); b_back_l.clicked.connect(lambda x: Back('hsi'))
        b_back_r = QW.QPushButton('Назад'); b_back_r.clicked.connect(lambda x: Back('tir'))
        b_clear = QW.QPushButton('Очистить'); b_clear.clicked.connect(Clear)
        b_homo = QW.QPushButton('Совместить'); b_homo.clicked.connect(Homography)
        b_save = QW.QPushButton('Сохранить'); b_save.clicked.connect(Save)
        
        hbox = QW.QHBoxLayout(); vbox_1, vbox_2 = QW.QVBoxLayout(), QW.QVBoxLayout()
        vbox_1.addWidget(self.canvas_l); 
        hbox_l = QW.QHBoxLayout(); hbox_l.addWidget(toolbar_l); hbox_l.addWidget(b_back_l); vbox_1.addLayout(hbox_l)
        
        mini_hbox = QW.QHBoxLayout(); mini_hbox.addWidget(open_hsi); mini_hbox.addWidget(open_tir)
        mini_hbox.addWidget(tir_min); mini_hbox.addWidget(tir_max); mini_hbox.addWidget(roi); mini_hbox.addWidget(point)
        
        mini_hbox_2 = QW.QHBoxLayout(); mini_hbox_2.addWidget(b_clear)
        mini_hbox_2.addWidget(b_homo); mini_hbox_2.addWidget(b_save)
        
        vbox_2.addLayout(mini_hbox); vbox_2.addLayout(mini_hbox_2); vbox_2.addWidget(self.canvas_r)
        hbox_r = QW.QHBoxLayout(); hbox_r.addWidget(toolbar_r); hbox_r.addWidget(b_back_r); vbox_2.addLayout(hbox_r)
        hbox.addLayout(vbox_1); hbox.addLayout(vbox_2)
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QW.QWidget()
        widget.setLayout(hbox)
        self.setCentralWidget(widget)
        self.show()
        
        def coord(event):
            x, y = map(float, re.findall("\d+\.\d+", str(event))[:2])
            return x, y
    
        # Interact
        def onclick(event):
            if str(event.button) == 'MouseButton.RIGHT':
                x, y = coord(event)
                if roi.isChecked():
                    if not self.rect_click:
                        self.rect[0] = x; self.rect[1] = y
                        self.rect_click = True
                    else:
                        self.rect[2] = x; self.rect[3] = y
                        x1, y1, x2, y2 = self.rect
                        if x1 > x2: x1, x2 = x2, x1
                        if y1 > y2: y1, y2 = y2, y1
                        self.rect = [x1, y1, x2, y2]
                        
                        if self.patch: self.patch.pop().remove()
                        self.patch += [self.ax_l.add_patch(Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1, 
                                                         fill=False, color='lime', linewidth=2.5))]
                        self.canvas_l.draw()
                        
                        self.rect_click = False; self.clear = False
                else:
                    self.point_dct['hsi'].append([x, y])
                    self.point_dct['hsi_img'].append(self.ax_l.scatter(x, y, linewidth=2.5))
                    self.canvas_l.draw()
                
        self.fig_l.canvas.mpl_connect('button_press_event', onclick)
        
        def onclick_2(event):
            if str(event.button) == 'MouseButton.RIGHT' and point.isChecked():
                x, y = coord(event)
                self.point_dct['tir'].append([x, y])
                self.point_dct['tir_img'].append(self.ax_r.scatter(x, y, linewidth=2.5))
                self.canvas_r.draw()
        
        self.fig_r.canvas.mpl_connect('button_press_event', onclick_2)
        
        self.showMaximized()
        
if __name__ == '__main__':
    app = QW.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())