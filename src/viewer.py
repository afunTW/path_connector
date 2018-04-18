import logging
import os
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.utils import tri

LOGGER = logging.getLogger(__name__)

class PathConnectorViewer(object):
    def __init__(self):
        self.root = None
        self.menu = None

    def init_windows(self):
        self.root = tk.Tk()
        if os.name == 'nt':
            self.root.iconbitmap('beetle.ico')
        self.root.focus_force()
        self.set_all_grid_rowconfigure(self.root, 0)
        self.set_all_grid_columnconfigure(self.root, 0)

        self._init_menu()
        self._init_frame()
        self._init_widgets()

    def _init_menu(self):
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # file menu
        self.menu_file = tk.Menu(self.menu)
        self.menu.add_cascade(label='File', menu=self.menu_file)

        # help menu
        self.menu_help = tk.Menu(self.menu)
        self.menu.add_cascade(label='Help', menu=self.menu_help)

    def _init_frame(self):
        # root > left_panel
        # frame for displaying image label
        self.frame_image = ttk.Frame(self.root)
        self.frame_image.grid(row=0, column=0)
        self.set_all_grid_rowconfigure(self.frame_image, 0)
        self.set_all_grid_columnconfigure(self.frame_image, 0, 1)

        # root > left_panel > display
        self.frame_display = tk.Frame(self.frame_image)
        self.frame_display.grid(row=0, column=0)

        # root > left_panel > display_meta
        self.labelframe_image = ttk.LabelFrame(self.frame_image)
        self.labelframe_image.grid(row=1, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.labelframe_image, 0, 1)
        self.set_all_grid_columnconfigure(self.labelframe_image, 0)

        # root > right_panel
        # operation frame that will display all application relevent information
        self.frame_op = ttk.Frame(self.root)
        self.frame_op.grid(row=0, column=1, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.frame_op, 0, 1, 2)
        self.set_all_grid_columnconfigure(self.frame_op, 0)

        # root > right_panel > state
        # frame for displaying current statement
        self.frame_state = ttk.Frame(self.frame_op)
        self.frame_state.grid(row=0, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.frame_state, 0, 1, 2)
        self.set_all_grid_columnconfigure(self.frame_state, 0)

        # root > right_panel > state > info
        # subframe for displaying frame information
        self.frame_info = ttk.LabelFrame(self.frame_state, text=u'影像訊息')
        self.frame_info.grid(row=0, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.frame_info, 0, 1)
        self.set_all_grid_columnconfigure(self.frame_info, 0)

        # root > right_panel > state > object
        # subframe for displaying object information
        self.frame_object = ttk.LabelFrame(self.frame_state, text=u'目標資訊')
        self.frame_object.grid(row=1, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.frame_object, 0)
        self.set_all_grid_columnconfigure(self.frame_object, 0, 1)

        # root > right_panel > state > legend
        # frame for legend
        self.labelframe_legend = ttk.LabelFrame(self.frame_state, text=u'圖例說明')
        self.labelframe_legend.grid(row=2, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.labelframe_legend, 0, 1)
        self.set_all_grid_columnconfigure(self.labelframe_legend, 0, 1, 2, 3)

        # root > right_panel > display_btn
        self.labelframe_target = ttk.LabelFrame(self.frame_op, text=u'需被標註的 bbox 應該是哪一個目標呢?')
        self.labelframe_target.grid(row=1, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.labelframe_target, 0, 1, 2, 3, 4, 5, 6, 7)
        self.set_all_grid_columnconfigure(self.labelframe_target, 0)

        # root > right_panel > op_btn
        self.labelframe_op = ttk.LabelFrame(self.frame_op, text=u'操作')
        self.labelframe_op.grid(row=2, column=0, sticky='news', padx=5, pady=5)
        self.set_all_grid_rowconfigure(self.labelframe_op, 0, 1, 2, 3, 4, 5, 6, 7)
        self.set_all_grid_columnconfigure(self.labelframe_op, 0, 1)

    def _init_widgets(self):
        # display
        self.label_display = ttk.Label(self.frame_display)
        self.label_display.grid(row=0, column=0)

        # display_meta
        self.label_nframe_v = ttk.Label(self.labelframe_image)
        self.label_nframe_v.grid(row=0, column=0)
        self.scale_nframe_v = ttk.Scale(self.labelframe_image, cursor='hand2')
        self.scale_nframe_v.grid(row=1, column=0, sticky='news', padx=10)

        # info
        self.label_video_name = ttk.Label(self.frame_info)
        self.label_video_name.grid(row=0, column=0, sticky='w')
        self.label_video_time = ttk.Label(self.frame_info)
        self.label_video_time.grid(row=1, column=0, sticky='w')

        # object
        self.treeview_object = ttk.Treeview(self.frame_object, height=3)
        self.treeview_object['columns'] = ('color', 'lastpoint', 'lastdetectedframe')
        self.treeview_object.heading('#0', text=u'名稱', anchor='center')
        self.treeview_object.heading('color', text=u'顏色')
        self.treeview_object.heading('lastpoint', text=u'在本幀是否有被偵測到')
        self.treeview_object.heading('lastdetectedframe', text=u'最後被偵測到的幀數')
        self.treeview_object.column('#0', anchor='w', width=50)
        self.treeview_object.column('color', anchor='center', width=60)
        self.treeview_object.column('lastpoint', anchor='center', width=130)
        self.treeview_object.column('lastdetectedframe', anchor='center', width=130)
        self.treeview_object.grid(row=0, column=0, sticky='news', pady=3)
        self.scrollbar_object = ttk.Scrollbar(self.frame_object, orient='vertical')
        self.scrollbar_object.grid(row=0, column=1, sticky='news', pady=3)

        # legend
        titles = [u'需被標註的\nbbox', u'目標路徑起點', u'目標在本幀的位置', u'目標路徑終點']
        self.label_legend_images = []
        self.label_legend_titles = []
        for i in range(4):
            label_legend_image = ttk.Label(self.labelframe_legend, width=10)
            label_legend_image.grid(row=0, column=i, padx=3, pady=3)
            label_legend_title = ttk.Label(self.labelframe_legend, text=titles[i], width=10, anchor='center')
            label_legend_title.grid(row=1, column=i, padx=3, pady=3)
            self.label_legend_images.append(label_legend_image)
            self.label_legend_titles.append(label_legend_title)

        # fixed op button
        self.label_suggest = ttk.Label(self.labelframe_target)

        # op_btn
        self.btn_return = ttk.Button(self.labelframe_op, text=u'回到需被標註的幀數 (Enter)', cursor='hand2')
        self.btn_return.grid(row=0, columnspan=2, sticky='news', padx=10, pady=2)
        self.btn_manual = ttk.Button(self.labelframe_op, text=u'進入 / 離開 Manual Label (q)', cursor='hand2')
        self.btn_manual.grid(row=1, columnspan=2, sticky='news', padx=10, pady=2)
        self.btn_reset = ttk.Button(self.labelframe_op, text=u'重置 (r)', cursor='hand2')
        self.btn_reset.grid(row=2, columnspan=2, sticky='news', padx=10, pady=2)
        self.btn_remove = ttk.Button(self.labelframe_op, text=u'刪除目標', cursor='hand2')
        self.btn_remove.grid(row=3, columnspan=2, sticky='news', padx=10, pady=2)
        self.btn_replay = ttk.Button(self.labelframe_op, text=u'回放已追踪路徑', cursor='hand2')
        self.btn_replay.grid(row=4, columnspan=2, sticky='news', padx=10, pady=2)

        self.label_max_path = ttk.Label(self.labelframe_op, text=u'顯示路徑的長度:')
        self.label_max_path.grid(row=5, column=0, sticky='ew', padx=10, pady=2)

        self.var_max_path = tk.IntVar()
        # self.label_max_path_value = ttk.Label(self.labelframe_op, textvariable=self.var_max_path)
        # self.label_max_path_value.grid(row=5, column=1)
        self.scale_max_path = ttk.Scale(self.labelframe_op, from_=2, to_=3000, length=200, cursor='hand2')
        self.scale_max_path.grid(row=5, column=1)

        # checkboxes
        self.var_show_bbox = tk.IntVar()
        self.var_show_bbox.set(1)
        self.checkbtn_show_bbox = ttk.Checkbutton(
            self.labelframe_op,
            variable=self.var_show_bbox,
            onvalue=1,
            offvalue=0,
            text=u'顯示埋葬蟲 bounding box')
        self.checkbtn_show_bbox.grid(row=6, column=0, sticky='news', padx=10, pady=5)

        self.var_show_rat = tk.IntVar()
        self.var_show_rat.set(0)
        self.checkbtn_show_rat = ttk.Checkbutton(
            self.labelframe_op,
            variable=self.var_show_rat,
            onvalue=1,
            offvalue=0,
            text=u'顯示老鼠輪廓')
        self.checkbtn_show_rat.grid(row=6, column=1, sticky='news', padx=10, pady=5)

        self.var_clear = tk.IntVar()
        self.var_clear.set(1)
        self.checkbtn_clear = ttk.Checkbutton(
            self.labelframe_op,
            variable=self.var_clear,
            onvalue=1,
            offvalue=0,
            text=u'透鏡')
        self.checkbtn_clear.grid(row=7, column=0, sticky='news', padx=10, pady=5)

        self.var_show_tracked = tk.IntVar()
        self.var_show_tracked.set(1)
        self.checkbtn_show_tracked = ttk.Checkbutton(
            self.labelframe_op,
            variable=self.var_show_tracked,
            onvalue=1,
            offvalue=0,
            text=u'顯示已追踪路徑')
        self.checkbtn_show_tracked.grid(row=7, column=1, sticky='news', padx=10, pady=5)

    def _get_fixed_legends(self):
        shape = (40, 40)
        bg = cv2.merge([np.ones(shape, dtype='uint8') * i for i in [237, 240, 240]])
        c = (20, 20)
        fg = (0, 0, 0)
        color = (50, 50, 255)

        # to be decided
        legend_1 = bg.copy()
        cv2.circle(legend_1, c, 15, color, 1)
        cv2.putText(legend_1, '?', (20 - 8, 20 + 9),cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1)
        legend_1 = cv2.cvtColor(legend_1, cv2.COLOR_BGR2RGB)

        # origin
        legend_2 = bg.copy()
        cv2.circle(legend_2, c, 10, fg, 1)
        cv2.circle(legend_2, c, 13, fg, 1)
        legend_2 = cv2.cvtColor(legend_2, cv2.COLOR_BGR2RGB)

        # current location
        legend_3 = bg.copy()
        tri_pts = tri(c)
        cv2.polylines(legend_3, tri_pts, True, fg, 3)

        # last detected location
        legend_4 = bg.copy()
        tri_pts = tri(c)
        cv2.polylines(legend_4, tri_pts, True, fg, 1)

        return (legend_1, legend_2, legend_3, legend_4)

    def _get_photo_from_path(self, path):
        image = Image.open(path)
        return ImageTk.PhotoImage(image)

    # set grid all column configure
    def set_all_grid_columnconfigure(self, widget, *cols):
        for col in cols:
            widget.grid_columnconfigure(col, weight=1)

    # set grid all row comfigure
    def set_all_grid_rowconfigure(self, widget, *rows):
        for row in rows:
            widget.grid_rowconfigure(row, weight=1)
