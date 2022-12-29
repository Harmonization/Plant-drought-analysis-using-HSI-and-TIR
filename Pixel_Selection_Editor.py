from PyQt5 import QtWidgets as QW, QtGui, QtCore
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NTool
from mpl_toolkits.axes_grid1 import make_axes_locatable
wave = np.load('./hypercube/wave.npy')

from hypercube.box import Box_Plant

class Window(QW.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Подбор порогов")
        
        dct_train = {1: 'C5', 3:'C11', 6:'C9', 8:'C5', 12:'C10', 19:'C5', 25:'C8'} # (для 1 и 3 не C а A)
        dct_test = {1: 'C8', 3: 'C8', 6: 'C8', 8: 'C8', 12: 'C8', 19: 'C4', 25: 'C10'}
        test_dct = {6: 'C8'}
        n_bands=[18, 28, 46, 59, 80, 92, 136, 143]
        
        self.fig_ul, self.ax_ul = plt.subplots(); self.fig_ur, self.ax_ur = plt.subplots()
        self.ax_l = self.fig_ur.add_axes([.08, .05, .4, .2]); self.ax_r = self.fig_ur.add_axes([.57, .05, .4, .2])
        
        self.fig_ul.subplots_adjust(left=.03, right=0.97, bottom=.035, top=1)
        self.fig_ur.subplots_adjust(left=0, right=1, bottom=.28, top=.97)
        self.canvas_ul = FigureCanvasQTAgg(self.fig_ul); self.canvas_ur = FigureCanvasQTAgg(self.fig_ur)
        axis = [self.ax_ul, self.ax_ur, self.ax_l, self.ax_r]
        self.cbar = 0
        
        # Sliders
        open_pot = QW.QComboBox()
        for i in (1, 3, 6, 8, 12, 19, 25):
            open_pot.addItem(f'{i} (Train)', f'{i}'); open_pot.addItem(f'{i} (Test)', f'{-i}')
        open_pot.addItem(f'Change', f'{0}')
        
        thr_l = QW.QLineEdit('54'); thr_l.setFixedSize(35, 35)
        thr_r = QW.QLineEdit('18'); thr_r.setFixedSize(35, 35)
        
        label_thr = QW.QLabel('')
        thr = QW.QSlider(minimum=-51, maximum=101, value=30, orientation=QtCore.Qt.Horizontal)
        a = QW.QSlider(minimum=-50, maximum=151, value=100, orientation=QtCore.Qt.Horizontal)
        b = QW.QSlider(minimum=-50, maximum=151, value=100, orientation=QtCore.Qt.Horizontal)
        n_band = QW.QSlider(minimum=0, maximum=203, value=143, orientation=QtCore.Qt.Horizontal)
        alpha = QW.QSlider(minimum=0, maximum=100, value=0, orientation=QtCore.Qt.Horizontal)
        
        def Load():
            self.day = int(open_pot.currentData())
            if self.day > 0: dct = dct_train; path = f'Data/1_90/{self.day}/{dct[self.day]}'
            elif self.day < 0: dct = dct_test; path = f'Data/1_90/{-self.day}/{dct[-self.day]}'
            else:
                path = QW.QFileDialog.getOpenFileName(self, "Open File", '', '*.npy')[0]
            self.pot = Box_Plant(path=path)
            Slider_Change_Value()
        
        def Alpha_Change():
            self.ax_ul.clear(); self.ax_ul.axis('off')
            self.ax_ul.imshow(self.rgb, alpha=abs(V(alpha) - 1))
            img_tir = self.ax_ul.imshow(self.tir, alpha=V(alpha), cmap='nipy_spectral', interpolation='nearest')
            label_thr.setText(f'Thr: {V(thr)}. Pixels: {self.rgb_pix_len}. TIR:')
            
            if self.cbar: self.cbar.remove()
            divider = make_axes_locatable(self.ax_ul)
            cax = divider.append_axes("bottom", size="7%", pad='5%')
            self.cbar = plt.colorbar(img_tir, cax=cax, orientation="horizontal")
            
            self.canvas_ul.draw()
        
        def V(slider):
            return np.around(slider.value() * .01, 2)
        
        def Slider_Change_Value():
            
            def Clear():
                for ax in axis:
                    ax.clear()
            
            self.pot.Clear_Mask()
            
            b1, b2 = int(thr_l.text()), int(thr_r.text()); R1, R2 = np.around(wave[b1]).astype(int), np.around(wave[b2]).astype(int) # b1, b2 = 142, 54
            s = self.pot.Channel([b1], [b2], func = lambda b1, b2: b1 - b2); s_max = s.max()
            self.pot.Wheat(V(thr), channel=s)
            self.rgb = self.pot.HCube[:, :, (70, 52, 19)].copy(); self.rgb[self.pot.Mask() == 0] = 0
            self.rgb_pix_len = sum(self.pot.Mask().ravel())
            
            channel = self.pot.Channel([n_band.value()])
            line, k, c = self.pot.LR_Deviation(channel, V(a), V(b))
            
            mask = self.pot.Mask(); self.tir = self.pot.TIR(); pix_len = sum(mask.ravel())
            Clear()
            
            try:
                Alpha_Change()
                self.pot.Plot(s, min_t=0.001, max_t=s_max, fig=self.fig_ur, ax=self.ax_ur, cbar=False, color='nipy_spectral'); self.ax_ur.set_title(f'Bands: {b1} - {b2} | {R1} - {R2} Nm. Pixels: {pix_len}')
        
                corr = np.around(np.corrcoef(channel[mask], self.tir[mask])[0, 1], 3)
                self.ax_l.scatter(channel[mask], self.tir[mask], alpha=.4, color='lime')
                self.ax_l.plot(channel[mask], line[mask], alpha=.4, c='red', linewidth=2.5, linestyle='--')
                self.ax_l.set_title(f'a: {V(a)}, b: {V(b)}, Correlation: {corr}')
                self.ax_l.set_xlabel(f'Band: {n_band.value()} | {np.around(wave[n_band.value()]).astype(int)} Nm'); self.ax_l.set_ylabel('TIR')
                
                lst_corr = self.pot.Correlation_Indx(indx_func=0); corr_max = int(lst_corr[0][0])
                lst_corr = np.array(sorted(lst_corr, key=lambda x: int(x[0])), dtype=float)
                self.ax_r.plot(wave, lst_corr[:, 1], linewidth=2.5, color='blue')
                self.ax_r.scatter([wave[n_band.value()]], [lst_corr[n_band.value(), 1]], linewidth=2.5, color='red')
                for special_band in n_bands:
                    y_low = (-.5 if self.day <= 12 else -1.05) - 0.01
                    self.ax_r.vlines(wave[special_band], y_low, lst_corr[special_band, 1], linewidth=1.5, color='black', linestyle='--')
                
                self.ax_r.set_title(f'Best channel: {corr_max} | {np.around(wave[corr_max]).astype(int)} Nm')
                if self.day <= 12: self.ax_r.set_ylim([-.5, 1.05])
                else: self.ax_r.set_ylim([-1.05, .6])
                self.ax_r.set_xlabel('Nm'); self.ax_r.set_ylabel('Pearson Correlation (TIR)')
                
            except:
                pass
            self.canvas_ul.draw(); self.canvas_ur.draw()
        
        open_pot.activated.connect(Load)
        thr.valueChanged.connect(Slider_Change_Value)
        a.valueChanged.connect(Slider_Change_Value)
        b.valueChanged.connect(Slider_Change_Value)
        n_band.valueChanged.connect(Slider_Change_Value)
        alpha.valueChanged.connect(Alpha_Change)
        
        # Hbox
        hbox = QW.QHBoxLayout(); 
        vbox_l = QW.QVBoxLayout(); vbox_l.addWidget(self.canvas_ul); 
        mini_hbox = QW.QHBoxLayout();
        mini_hbox.addWidget(thr); mini_hbox.addWidget(label_thr); mini_hbox.addWidget(alpha); mini_hbox.addWidget(QW.QLabel('Mask channels:')); mini_hbox.addWidget(thr_l); mini_hbox.addWidget(thr_r); vbox_l.addLayout(mini_hbox); hbox.addLayout(vbox_l)
        vbox_r = QW.QVBoxLayout(); vbox_r.addWidget(self.canvas_ur)
        panel_u = QW.QHBoxLayout(); panel_u.addWidget(QW.QLabel('Day:')); panel_u.addWidget(open_pot); vbox_r.addLayout(panel_u)
        hbox_d = QW.QHBoxLayout(); vbox_r.addLayout(hbox_d)
        panel_u.addWidget(QW.QLabel('a:')); panel_u.addWidget(a); panel_u.addWidget(QW.QLabel('b:')); panel_u.addWidget(b); panel_u.addWidget(QW.QLabel('Band:')); panel_u.addWidget(n_band)
        hbox.addLayout(vbox_r)
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QW.QWidget()
        widget.setLayout(hbox)
        self.setCentralWidget(widget)

        open_pot.setCurrentIndex(6); Load()
        self.showMaximized()
        
if __name__ == '__main__':
    app = QW.QApplication(sys.argv)
    app.setStyleSheet('QLabel { font: bold italic }')
    window = Window()
    window.show()
    sys.exit(app.exec())