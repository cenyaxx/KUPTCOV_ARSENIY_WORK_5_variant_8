#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практическая работа №5, Вариант 8
HSL ↔ Lab преобразование с промежуточным RGB
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# ============================================================
# 1. РУЧНАЯ РЕАЛИЗАЦИЯ RGB ↔ HSL (по формулам Wikipedia/NIWA)
# ============================================================

def rgb_to_hsl_vectorized(arr):
    """
    Векторизованное преобразование RGB → HSL.
    Вход: numpy array uint8 (H, W, 3)
    Выход: H, S, L каждый (H, W) float
    Диапазоны: H [0,360), S [0,100], L [0,100]
    """
    arr = arr.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    d = mx - mn

    # Lightness
    l = (mx + mn) / 2.0

    # Saturation
    s = np.zeros_like(l)
    mask = (mx != mn)
    s[mask & (l <= 0.5)] = d[mask & (l <= 0.5)] / (mx[mask & (l <= 0.5)] + mn[mask & (l <= 0.5)])
    s[mask & (l > 0.5)] = d[mask & (l > 0.5)] / (2.0 - mx[mask & (l > 0.5)] - mn[mask & (l > 0.5)])

    # Hue
    h = np.zeros_like(l)

    # max == r
    mask_r = (mx == r) & (mx != mn)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / d[mask_r])
    h[mask_r] = (h[mask_r] % 6) * 60.0

    # max == g
    mask_g = (mx == g) & (mx != mn)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / d[mask_g] + 2.0) * 60.0

    # max == b
    mask_b = (mx == b) & (mx != mn)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / d[mask_b] + 4.0) * 60.0

    h = h % 360.0

    return h, s * 100.0, l * 100.0


def hsl_to_rgb_vectorized(h, s, l):
    """
    Векторизованное преобразование HSL → RGB.
    Вход: H [0,360), S [0,100], L [0,100]
    Выход: numpy array uint8 (H, W, 3)
    Алгоритм: через промежуточные p, q (StackOverflow/Wikipedia alternative)
    """
    h = h % 360.0
    s = s / 100.0
    l = l / 100.0

    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0

    rgb = np.zeros((*h.shape, 3), dtype=np.float32)

    seg = (h / 60.0).astype(np.int32)
    seg = np.clip(seg, 0, 5)

    # Сегмент 0: h<60
    mask = seg == 0
    rgb[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=-1)
    # Сегмент 1: 60<=h<120
    mask = seg == 1
    rgb[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=-1)
    # Сегмент 2: 120<=h<180
    mask = seg == 2
    rgb[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=-1)
    # Сегмент 3: 180<=h<240
    mask = seg == 3
    rgb[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=-1)
    # Сегмент 4: 240<=h<300
    mask = seg == 4
    rgb[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=-1)
    # Сегмент 5: 300<=h<360
    mask = seg == 5
    rgb[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=-1)

    rgb = (rgb + m[..., None]) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================
# 2. РЕАЛИЗАЦИЯ RGB ↔ Lab (через XYZ, промежуточное преобразование)
# ============================================================

def rgb_to_xyz_vectorized(arr):
    """RGB → XYZ (D65, sRGB gamma)"""
    arr = arr.astype(np.float32) / 255.0
    mask = arr <= 0.04045
    arr[mask] = arr[mask] / 12.92
    arr[~mask] = np.power((arr[~mask] + 0.055) / 1.055, 2.4)

    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = arr @ m.T
    return xyz


def xyz_to_lab_vectorized(xyz):
    """XYZ → Lab (D65)"""
    xr, yr, zr = 0.95047, 1.0, 1.08883
    t = xyz / np.array([xr, yr, zr], dtype=np.float32)

    mask = t > 0.008856
    t[mask] = np.power(t[mask], 1.0 / 3.0)
    t[~mask] = 7.787 * t[~mask] + 16.0 / 116.0

    l = 116.0 * t[..., 1] - 16.0
    a = 500.0 * (t[..., 0] - t[..., 1])
    b = 200.0 * (t[..., 1] - t[..., 2])
    return l, a, b


def lab_to_xyz_vectorized(l, a, b):
    """Lab → XYZ"""
    xr, yr, zr = 0.95047, 1.0, 1.08883
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    t = np.stack([fx, fy, fz], axis=-1)
    mask = t > 0.2068966
    t[mask] = np.power(t[mask], 3.0)
    t[~mask] = (t[~mask] - 16.0 / 116.0) / 7.787

    xyz = t * np.array([xr, yr, zr], dtype=np.float32)
    return xyz


def xyz_to_rgb_vectorized(xyz):
    """XYZ → RGB (D65, sRGB gamma)"""
    m_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    rgb_lin = xyz @ m_inv.T
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)

    mask = rgb_lin <= 0.0031308
    rgb_lin[mask] = rgb_lin[mask] * 12.92
    rgb_lin[~mask] = 1.055 * np.power(rgb_lin[~mask], 1.0 / 2.4) - 0.055

    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    return (rgb_lin * 255.0).astype(np.uint8)


def rgb_to_lab_vectorized(arr):
    """RGB → Lab"""
    return xyz_to_lab_vectorized(rgb_to_xyz_vectorized(arr))


def lab_to_rgb_vectorized(l, a, b):
    """Lab → RGB"""
    xyz = lab_to_xyz_vectorized(l, a, b)
    return xyz_to_rgb_vectorized(xyz)


# ============================================================
# 3. ГЛАВНЫЙ КЛАСС ПРИЛОЖЕНИЯ
# ============================================================

class HSLLabEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Практическая работа №5 — Вариант 8 (HSL ↔ Lab)")
        self.root.geometry("1200x800")

        # Данные изображения
        self.original_image = None      # PIL Image RGB
        self.original_array = None      # numpy uint8 (H, W, 3)
        self.current_array = None       # numpy uint8 (H, W, 3)
        self.display_image = None       # ImageTk для Canvas

        # HSL оригинала (для каждого пикселя)
        self.h_orig = None
        self.s_orig = None
        self.l_orig = None

        # Параметры ползунков
        self.h_shift = tk.DoubleVar(value=0.0)      # смещение оттенка [-180, 180]
        self.s_scale = tk.DoubleVar(value=100.0)     # масштаб насыщенности [0, 200]
        self.l_shift = tk.DoubleVar(value=0.0)      # смещение яркости [-100, 100]

        # Флаги
        self.use_lab_gateway = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        # --- Верхняя панель (кнопки) ---
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Сохранить результат", command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Сбросить ползунки", command=self.reset_sliders).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(top_frame, text="Lab-шлюз (RGB→Lab→RGB при сохранении)", 
                        variable=self.use_lab_gateway).pack(side=tk.LEFT, padx=20)

        # --- Основная область: изображение слева, панель управления справа ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas с изображением
        self.canvas = tk.Canvas(main_frame, bg="#333333", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Правая панель
        right_frame = ttk.Frame(main_frame, padding=10, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # --- Ползунки HSL ---
        ttk.Label(right_frame, text="Коррекция HSL", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Hue Shift
        ttk.Label(right_frame, text="Смещение Hue (°): 0").pack(anchor=tk.W)
        self.lbl_h = ttk.Label(right_frame, text="0°")
        self.lbl_h.pack(anchor=tk.W)
        scale_h = ttk.Scale(right_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
                            variable=self.h_shift, command=lambda v: self.on_slider_changed(v, self.lbl_h, "°", self.h_shift))
        scale_h.pack(fill=tk.X, pady=(0, 10))

        # Saturation Scale
        ttk.Label(right_frame, text="Масштаб Saturation (%): 100").pack(anchor=tk.W)
        self.lbl_s = ttk.Label(right_frame, text="100%")
        self.lbl_s.pack(anchor=tk.W)
        scale_s = ttk.Scale(right_frame, from_=0, to=200, orient=tk.HORIZONTAL,
                            variable=self.s_scale, command=lambda v: self.on_slider_changed(v, self.lbl_s, "%", self.s_scale))
        scale_s.pack(fill=tk.X, pady=(0, 10))

        # Lightness Shift
        ttk.Label(right_frame, text="Смещение Lightness (%): 0").pack(anchor=tk.W)
        self.lbl_l = ttk.Label(right_frame, text="0%")
        self.lbl_l.pack(anchor=tk.W)
        scale_l = ttk.Scale(right_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                            variable=self.l_shift, command=lambda v: self.on_slider_changed(v, self.lbl_l, "%", self.l_shift))
        scale_l.pack(fill=tk.X, pady=(0, 20))

        # --- Информация о Lab ---
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(right_frame, text="Lab-информация (средние)", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.lab_info = ttk.Label(right_frame, text="L: --  a: --  b: --", font=("Courier", 10))
        self.lab_info.pack(anchor=tk.W, pady=5)

        ttk.Label(right_frame, text="Пиксель под курсором:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.pixel_info = ttk.Label(right_frame, text="RGB: --  HSL: --  Lab: --", font=("Courier", 9))
        self.pixel_info.pack(anchor=tk.W, pady=5)

        # --- Пояснение варианта 8 ---
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        info_text = (
            "Вариант 8: HSL ↔ Lab\n"
            "Промежуточное преобразование через RGB/XYZ.\n\n"
            "При включении 'Lab-шлюз' изображение после\n"
            "HSL-коррекции проходит RGB→Lab→RGB."
        )
        ttk.Label(right_frame, text=info_text, wraplength=280, justify=tk.LEFT, foreground="#555").pack(anchor=tk.W)

        # --- Статус ---
        self.status = ttk.Label(self.root, text="Готово. Загрузите изображение.", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def on_slider_changed(self, value, label, suffix, var):
        label.config(text=f"{var.get():.1f}{suffix}")
        self.apply_changes()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.original_array = np.array(self.original_image)

            # Вычисляем HSL оригинала один раз
            self.status.config(text="Вычисление HSL...")
            self.root.update()
            self.h_orig, self.s_orig, self.l_orig = rgb_to_hsl_vectorized(self.original_array)

            self.apply_changes()
            self.status.config(text=f"Загружено: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def apply_changes(self):
        if self.original_array is None:
            return

        h_shift = self.h_shift.get()
        s_scale = self.s_scale.get() / 100.0
        l_shift = self.l_shift.get()

        # Применяем коррекцию к HSL
        h_new = (self.h_orig + h_shift) % 360.0
        s_new = np.clip(self.s_orig * s_scale, 0.0, 100.0)
        l_new = np.clip(self.l_orig + l_shift, 0.0, 100.0)

        # HSL → RGB
        rgb_new = hsl_to_rgb_vectorized(h_new, s_new, l_new)

        # Вариант 8: промежуточное преобразование Lab (опционально для визуализации)
        # Для демонстрации всегда считаем Lab для отображения
        l_lab, a_lab, b_lab = rgb_to_lab_vectorized(rgb_new)

        # Если включен Lab-шлюз — пропускаем через Lab→RGB (может слегка изменить цвета из-за gamut)
        if self.use_lab_gateway.get():
            rgb_new = lab_to_rgb_vectorized(l_lab, a_lab, b_lab)

        self.current_array = rgb_new

        # Обновляем Lab-информацию (средние)
        avg_l = np.mean(l_lab)
        avg_a = np.mean(a_lab)
        avg_b = np.mean(b_lab)
        self.lab_info.config(text=f"L: {avg_l:6.2f}  a: {avg_a:6.2f}  b: {avg_b:6.2f}")

        # Отображение
        self._update_canvas()

    def _update_canvas(self):
        if self.current_array is None:
            return

        h, w = self.current_array.shape[:2]
        # Масштабируем для отображения, если изображение слишком большое
        max_size = 900
        img = Image.fromarray(self.current_array)
        if max(h, w) > max_size:
            ratio = max_size / max(h, w)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

        self.display_image = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.display_image.width(), height=self.display_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def on_mouse_move(self, event):
        if self.current_array is None or self.display_image is None:
            return

        # Координаты на оригинальном изображении
        canvas_w = self.display_image.width()
        canvas_h = self.display_image.height()
        orig_h, orig_w = self.current_array.shape[:2]

        if canvas_w == 0 or canvas_h == 0:
            return

        scale_x = orig_w / canvas_w
        scale_y = orig_h / canvas_h

        x = int(event.x * scale_x)
        y = int(event.y * scale_y)

        if 0 <= x < orig_w and 0 <= y < orig_h:
            r, g, b = self.current_array[y, x]
            h_val, s_val, l_val = self.h_orig[y, x], self.s_orig[y, x], self.l_orig[y, x]

            # Применяем текущие смещения для отображения актуальных HSL
            h_val = (h_val + self.h_shift.get()) % 360.0
            s_val = np.clip(s_val * (self.s_scale.get() / 100.0), 0, 100.0)
            l_val = np.clip(l_val + self.l_shift.get(), 0, 100.0)

            # Lab текущего пикселя
            l_lab, a_lab, b_lab = rgb_to_lab_vectorized(self.current_array[y:y+1, x:x+1])

            self.pixel_info.config(
                text=f"RGB:({r},{g},{b}) HSL:({h_val:.1f}°, {s_val:.1f}%, {l_val:.1f}%) Lab:({l_lab[0,0]:.1f}, {a_lab[0,0]:.1f}, {b_lab[0,0]:.1f})"
            )

    def reset_sliders(self):
        self.h_shift.set(0.0)
        self.s_scale.set(100.0)
        self.l_shift.set(0.0)
        self.lbl_h.config(text="0°")
        self.lbl_s.config(text="100%")
        self.lbl_l.config(text="0%")
        self.apply_changes()

    def save_image(self):
        if self.current_array is None:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if not path:
            return

        try:
            img = Image.fromarray(self.current_array)
            img.save(path)
            self.status.config(text=f"Сохранено: {path}")
            messagebox.showinfo("Успех", f"Изображение сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")


def main():
    root = tk.Tk()
    app = HSLLabEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
