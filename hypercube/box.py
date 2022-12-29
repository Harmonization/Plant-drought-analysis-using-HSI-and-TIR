import numpy as np, matplotlib.pyplot as plt, cv2
from matplotlib.colors import LinearSegmentedColormap as LSCMap
from sklearn.linear_model import LinearRegression
from functools import reduce

class Box_Plant:
    
    def __init__(self, path='Data/1_90/12/Cube_1', plant=None):
        if plant is None: # Куб HSI+TIR
            if path[-4:] == '.npy': path = path[:-4]
            self.HCube = np.load(f'{path}.npy')
            self.shape = self.HCube.shape[:2]
            self.Clear_Mask()
        
        else: # Создание копии экземпляра класса
            self.HCube = plant.HCube.copy()
            self.shape = plant.HCube.shape
            self.mask = plant.mask
        
    def Channels_Info(self):
        # Получить информацию о соответствии номеров каналов HSI и длин волн
        print('Номер канала: число нанометров')
        return {i: b for i, b in enumerate(np.load('wave.npy'))}
    
    def Channel(self, *bands, func = lambda b1: b1):
        # Вернуть канал или ВИ. Математика каналов задается в func, каналы в bands. Пример:
        # TIR: Channel([204]); NDVI: Channel([136, 137], [96, 97], func = lambda b1, b2: b1 - b2)
        return func(*map(lambda group: self.HCube[:, :, group].mean(axis=2), bands))
    
    def Mask(self):
        # Вернуть копию маски
        return self.mask.copy()
    
    def Set_Mask(self, condition):
        # condition - маска значений, которые нужно обозначить фоном и не учитывать в вычислениях
        #self.HCube[condition, -1] = 0
        self.mask[condition] = 0
        
    def Clear_Mask(self):
        # Очистить маску
        #self.HCube[:, :, -1] = np.ones(self.shape, dtype=bool)
        self.mask = np.ones(self.shape, dtype=bool)
    
    def Wheat(self, thr=.35, channel=0):
        # Создание маски растения, исключающей фон
        if not np.all(channel): channel = self.Channel([54], [18], func = lambda b1, b2: b1 - b2)
        w = (channel - channel.min()) / (channel.max() - channel.min())
        self.Set_Mask(w < thr)
        
    def TIR(self):
        # Возвращает TIR
        return self.HCube[:, :, -1]
        
    def Half(self, half=0):
        # Извлечение половины изображения, half=0 для правой, иначе левая
        border_ = self.HCube.shape[1] // 2 + 10; _border = self.HCube.shape[1] // 2 - 10
        self.HCube = self.HCube[:, :_border, :] if half else self.HCube[:, border_:, :]
        self.mask = self.mask[:, :_border] if half else self.mask[:, border_:]
        self.shape = self.TIR().shape
    
    def Denoise(self, p1=5, p2=95, channel=None, internal=True): 
        # internal=True - меняет маску внутри класса, иначе лишь возвращает измененный вариант
        if channel is None: channel = self.TIR()
        mask = self.Mask()
        v1, v2 = np.percentile(channel[mask], [p1, p2])
        if internal: self.Set_Mask((channel < v1) | (v2 < channel))
        else: mask[(channel < v1) | (v2 < channel)] = 0; return mask
        
    def LR_Deviation(self, channel, t1=1, t2=1):
        # Линейная регрессия между channel и TIR, удаление отклонений от нее
        # channel-одноканальное изображение (индекс или канал), t1-верхний порог, t2-нижний порог
        mask, tir = self.Mask(), self.TIR()
        LR = LinearRegression().fit(channel[mask, np.newaxis], tir[mask])
        line = LR.coef_[0] * channel + LR.intercept_
        
        diff = line - tir
        self.Set_Mask((diff < -t1) | (diff > t2))
        return line, LR.coef_[0], LR.intercept_
    
    def MMNorm_Channel(self, channel):
        # max/min нормировка канала
        MMNorm = lambda data: (data - data.min()) / (data.max() - data.min())
        norm_channel = channel.copy()
        mask = self.Mask()
        norm_channel[mask] = MMNorm(norm_channel[mask])
        return norm_channel
    
    def Scale_Channel(self, channel, s1, s2):
        # Масштабирование канала
        Scale = lambda data, min_d, max_d: data * (max_d - min_d) + min_d
        scale_channel = channel.copy()
        mask = self.Mask()
        scale_channel[mask] = Scale(scale_channel[mask], s1, s2)
        return scale_channel
    
    def Entropy_Channel(self, channel):
        # Энтропия канала
        img = self.Scale_Channel(self.MMNorm_Channel(channel), 0, 256)
        hist = cv2.calcHist([img.astype(np.uint8)[self.Mask()]], [0], None, [256], [0, 256])
        
        entropy = bins = 0
        S = img.shape[0] * img.shape[1]
        lst = [hi[0] / S * np.log(hi[0] / S) for hi in hist if hi[0]]
        entropy = reduce(lambda prev, cur: prev - cur, lst, 0)
        return entropy / np.log(len(lst))
    
    def MSNorm_Channel(self, channel):
        # mean/std нормализация канала
        MSNorm = lambda data: (data - data.mean()) / data.std()
        norm_channel = channel.copy()
        mask = self.Mask()
        norm_channel[mask] = MSNorm(norm_channel[mask])
        return norm_channel
    
    def Add_Box(self, *plant, axis=1):
        # Объединение с другими Box по оси axis
        min_h = min(self.shape[0], *map(lambda p: p.shape[0], plant))
        min_w = min(self.shape[1], *map(lambda p: p.shape[1], plant))
        
        lst = [self] + list(plant)
        
        arr = list(map(lambda p: p.HCube[:min_h, :min_w], lst))
        arr_mask = list(map(lambda p: p.mask[:min_h, :min_w], lst))
            
        self.HCube = np.concatenate(arr, axis=axis)
        self.mask = np.concatenate(arr_mask, axis=axis)
        self.shape = self.TIR().shape
        
    def Split_Box(self, h=3, w=2):
        # Разбиение Box на h частей по вертикали и на w по горизонтали, возвращает список частей
        h_step, w_step = np.array(self.shape) // [h, w]
        
        return [Box_Plant(HCube=self.HCube[i * h_step : i * h_step + h_step, j * w_step : j * w_step + w_step]) 
                for i in range(h) for j in range(w)]
    
    def Correlation_Indx(self, indx_func=lambda x1, x2: x1 - x2, func_name='-', func=lambda x, y: np.corrcoef(x, y)[0, 1], sort=True):
        mask = self.Mask()
        tir = self.TIR()[mask]
        hsi = self.HCube[mask]
        if not indx_func:
            res = {i: np.corrcoef(hsi[:, i], tir)[0, 1] if np.any(hsi[:, i]) else 0 for i in range(204)}
            return sorted(res.items(), key=lambda x: abs(x[1]), reverse=True) if sort else res
        else:
            res = {f'{j}{func_name}{i}': func(indx_func(hsi[:, j], hsi[:, i]), tir) for i in range(204) for j in range(i + 1, 204) 
                           if np.any(hsi[:, i]) and np.any(hsi[:, j])}
            return sorted(res.items(), key=lambda item: abs(item[1]), reverse=True) if sort else res
    
    def Smooth(self, N, mode=0):
        # Усреднение значений пикселей растений в окошке NxN
        # mode: (0 - только HSI, 1 - только TIR, 2 - и HSI и TIR)
        mask = self.Mask()
        if mode == 0:
            bands = range(self.HCube.shape[2] - 1)
        elif mode == 1:
            bands = (204,)
        else:
            bands = range(self.HCube.shape[2])
        cube = self.HCube[:, :, bands]
        sh = self.shape
        n = N // 2
        for b in range(cube.shape[-1]):
            for i in range(n, sh[0] - n + 1, N):
                for j in range(n, sh[1] - n + 1, N):
                    m = mask[i-n:i+n+1, j-n:j+n+1]
                    if m.any(): cube[i-n:i+n+1, j-n:j+n+1, b][m] = cube[i-n:i+n+1, j-n:j+n+1, b][m].mean()
        
    def Plot(self, channel, fig=None, ax=None, min_t=-1, max_t=-1, color=None, mask=None, cbar=True):
        # Визуализация канала
        if color is None: 
            color = LSCMap.from_list("LSCMap", ["lawngreen", "yellow", "red"])
            color.set_under('black')
        if ax is None: fig, ax = plt.subplots()
        ax.axis('off')
        
        if mask is None: mask = self.Mask()
        not_zero = channel[mask]
        if min_t == -1: min_t = not_zero.min()
        if max_t == -1: max_t = not_zero.max()
        
        vis = channel.copy()
        vis[~mask] = 0
        
        img = ax.imshow(vis, cmap=color, vmin=min_t, vmax=max_t)
        if cbar: fig.colorbar(img, ax=ax)
        return img
    
    def Hist(self, channel, ax, min_i=-1, max_i=-1):
        mask = self.Mask()
        not_zero = channel[mask]
        if min_i == -1: min_i = not_zero.min()
        if max_i == -1: max_i = not_zero.max()
        ax.hist(not_zero, 500, [min_i, max_i])
        
    def Plot_TIR(self, min_t=-1, max_t=-1):
        # Визуализация TIR
        print(f'Пикселей растения: {sum(self.Mask().ravel())}')
        fig, ax = plt.subplots(figsize=(9, 5))
        img = self.Plot(self.TIR(), fig, ax, min_t=min_t, max_t=max_t)
        plt.show()