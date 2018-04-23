import copy
import logging
import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.keyhandler import KeyHandler
from src.utils import Utils, catchtime
from src.yoloreader import YOLOReader

LOGGER = logging.getLogger(__name__)

# basic setup variables
WIN_NAME = 'Path Connector'
COLOR_NAME = [
    ('green', 'limegreen'),
    ('blue', 'deepskyblue'),
    ('yellow', 'gold'),
    ('purple', 'blueviolet'),
    ('orange', 'orange'),
    ('pink', 'pink'),
    ('cyan', 'cyan'),
    ('black', 'black'),
    ('red', 'red'),
    ('white', 'white')
]
COLOR = [
    (50, 205, 50),
    (255, 191, 0),
    (0, 215, 255),
    (211, 85, 186),
    (0, 165, 255),
    (255, 102, 255),
    (255, 255, 0),
    (0, 0, 0),
    (100, 10, 255),
    (255, 255, 255)
]

# UI required variables
LARGE_FONT= ("Verdana", 12)

class PathConnector(YOLOReader, KeyHandler, Utils):

    def __init__(self, maximum, tol):
        """
        ----------------------------------------------------------------------
        This is a GUI for connecting paths results that were detected by YOLO.
        ----------------------------------------------------------------------
        maximum: temp varible; maximum number of frames to display connected.
        tol: temp variable; maximum tolerance distance between context path coordinate.

        """
        # basic setup
        self.win_name = WIN_NAME
        self.history_file = None
        self.video_path = None
        self.video = None
        self.width = None
        self.height = None
        self.fps = None
        self.resolution = None
        self.total_frame = None
        self.__yolo_results__ = None
        self.__total_n_frame__ = None
        self.is_finish = False

        # basic variables
        self.color = COLOR
        self.color_name = COLOR_NAME
        self.maximum = maximum
        self.tol = tol

        # variables for displaying in canvas
        self._frame = None
        self._orig_frame = None
        self.n_frame = None
        self.last_n_frame = None
        self.stop_n_frame = None
        self.mv_x = 0
        self.mv_y = 0
        self.last_x = 0
        self.last_y = 0
        self.image = None
        self.image_tracked = None
        self.clear = False
        self.is_clear = 1 # right mouse click for removing drawing
        self.tmp_line = []
        self.drag_flag = None

        # variables for recording things
        self.object_name = {}
        self.results_dict = {}
        self.dist_records = {}
        self.label_dict = {}
        self.tmp_results_dict = {}
        self.min_label_ind = None
        self.undone_pts = None
        self.fp_pts = []
        self.undo_records = []
        self.tracked_frames = []
        self.suggest_ind = []
        self.current_pts = None
        self.current_pts_n = None
        self.safe = True
        self.is_calculate = False
        self.is_manual = False
        self.rat_cnt_dict = {}
        self.hit_condi = None

        # variables for breaking from calculating loop
        self.n_run = None
        self.cancel_id = None

        # tkinter widgets
        self.root = None
        self.all_buttons = []
        self._init_width = None
        self._init_height = None
        self._r_height = None
        self._r_width = None
        self._c_height = None
        self._c_width = None

    def _init_main_viewer(self):
        # skeleton and root setting
        self.init_windows()
        self.root.title(self.win_name)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.option_add('*tearOff', False)
        self.root.option_add("*Font", "verdana 10")
        self.root.aspect(1, 1, 1, 1)
        if os.name == 'nt':
            self.root.state('zoomed')
        else:
            self.root.attributes('-zoomed', True)

        # binding
        self.key_binding()

    def _get_timestring(self, n_frame, fps):
        sec = round(n_frame / fps, 2)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _get_photo_from_path(self, path):
        image = Image.open(path)
        return ImageTk.PhotoImage(image)

    # read the video frame by given ind
    def update_frame(self, ind=None):
        ind = ind if ind is not None else self.n_frame - 1
        self.video.set(cv2.CAP_PROP_POS_FRAMES, ind)
        ok, self._frame = self.video.read()
        self._orig_frame = self._frame.copy()

        if not ok:
            self.msg("Can't open the video")

    # update object information table
    def update_info(self):
        if self.treeview_object:
            for n in sorted([k for k, v in self.object_name.items() if v['on']]):
                if self.is_manual:
                    rd = self.tmp_results_dict[n]
                else:
                    rd = self.results_dict[n]

                # check if beetle is detected in current frame
                if 'n_frame' in rd and self.n_frame in rd['n_frame']:
                    is_detected = True
                else:
                    is_detected = False

                # update info table
                color_name = self.color_name[self.object_name[n]['ind']][0]
                last_detected_frame = rd['n_frame'][-1]
                if self.treeview_object.exists(n):
                    self.treeview_object.item(
                        n,
                        text=self.object_name[n]['display_name'],
                        values=(color_name, is_detected, last_detected_frame)
                    )
                else:
                    self.object_name[n]['on'] = True
                    self.treeview_object.insert(
                        '', 'end', n,
                        text=n,
                        values=(color_name, is_detected, last_detected_frame)
                    )

    # update video info
    def update_label(self):
        if self.root:
            text_video_name = self.video_path.split('/')[-1]
            text_time = self._get_timestring(self.n_frame, self.fps)

            self.label_video_name.configure(text='影像檔名: %s' % text_video_name)
            self.label_video_time.configure(text='影像時間: %s' % text_time)
            self.label_nframe_v.configure(text="當前幀數: %s/%s" % (self.n_frame, self.total_frame))
            self.scale_nframe_v.set(self.n_frame)

            self.label_display.after(200, self.update_label)

    # get the current frame when self.last_n_frame != self.n_frame (redundant)
    def update_draw(self, tup=None):
        if self.root:
            if self.last_n_frame != self.n_frame:
                self.update_frame()
                self.last_n_frame = self.n_frame
            else:
                self._frame = self._orig_frame.copy()
            self.update_info()
            self.draw(tup)
            try:
                if not self.is_calculate:
                    self.image = ImageTk.PhotoImage(Image.fromarray(self._frame))
                if not self.safe:
                    self.label_display.config(image=self.image)
            except Exception as e:
                LOGGER.exception(e)

            self.label_display.after(20, self.update_draw)

    def update_track(self, ind):
        if len(self.tracked_frames) > 0 and ind < (len(self.tracked_frames) - 1) and self.safe:
            frame = self.tracked_frames[ind]
            if ind < (len(self.tracked_frames) - 1):
                ind += 1
                self.scale_nframe_v.set(20*(ind) + 1)
                self.label_display.configure(image=frame)
                self.root.after(200, self.update_track, ind)
        else:
            self.scale_nframe_v.set(self.stop_n_frame)
            self.safe = False
            self.label_display.configure(image=self.image)

    # entry window to load video
    def start(self):
        root = tk.Tk()
        if os.name == 'nt':
            root.iconbitmap('beetle.ico')
        root.title(self.win_name)
        label = ttk.Label(root, text='請載入埋葬蟲影像。\n* 影像路徑底下請附上包含埋葬蟲偵測結果並和影像檔名相同的 txt。', font=LARGE_FONT)
        label.pack(padx=10, pady=10)

        label2 = ttk.Label(root, text='當下沒有影像。')
        label2.pack(padx=10, pady=10)

        button = ttk.Button(root, text='載入影像', command=lambda l = label2: self.get_path(l))
        button.pack(padx=10, pady=10)

        button2 = ttk.Button(root, text='開始', command=lambda r = root: self.ready(r))
        button2.pack(padx=10, pady=10)

        self.center(root)
        root.bind('<Escape>', lambda event: root.destroy())
        root.mainloop()

    # load video
    def init_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        self.width = int(self.video.get(3))
        self.height = int(self.video.get(4))
        self.fps = int(self.video.get(5))
        self.resolution = (self.width, self.height)
        self.total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__yolo_results__ = self.read_yolo_result()
        self.__total_n_frame__ = len(self.__yolo_results__)
        self.is_finish = False

        # init related params
        self.object_name = {}
        self.results_dict = {}
        self.tmp_results_dict = {}
        self.dist_records = {}
        self.label_dict = {}
        self.undo_records = []
        self.fp_pts = []
        self.drag_flag = None

    # start > 開始 (press) > ready > run
    def ready(self, r):
        if self.video_path is not None:
            yolo_results_path = self.video_path.split('.avi')[0] + '.txt'
            res = os.path.isfile(yolo_results_path)
        else:
            res = True

        if not res:
            self.msg('路徑底下沒有對應的埋葬蟲偵測結果的 txt。')
        elif self.video_path is None:
            self.msg('請載入影像檔案。')
        else:
            r.destroy()
            self.init_video()
            self.run()

    # save records for undo, max_length=15
    def save_records(self):
        records = (
            copy.deepcopy(self.results_dict),
            copy.deepcopy(self.tmp_results_dict),
            {**self.dist_records},
            copy.deepcopy(self.hit_condi),
            self.stop_n_frame,
            self.undone_pts,
            self.current_pts,
            self.current_pts_n,
            copy.deepcopy(self.suggest_ind),
            copy.deepcopy(self.object_name)
        )

        if len(self.undo_records) > 0:
            if self.stop_n_frame != self.undo_records[-1][1]:
                self.undo_records.append(records)
        else:
            self.undo_records.append(records)
        # remove undo record if it is too long
        if len(self.undo_records) >= 15:
            self.undo_records = self.undo_records[-15:]

    # set root window at the center
    def center_root(self, r=0):
        self.root.update_idletasks()
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()

        size = (self.root.winfo_width(), self.root.winfo_height())
        x = w/2 - size[0]/2
        y = h/2.25 - size[1]/2
        r = 0 if self.root.state() == 'zoomed' else r
        self.root.geometry("%dx%d+%d+%d" % (size[0], size[1]+r, x, y))

    # load .dat file if exists
    def load_history(self):
        self.history_file = self.video_path.split('.avi')[0] + '.dat'
        if os.path.isfile(self.history_file):
            with open(self.history_file, 'rb') as f:
                self.results_dict, self.tmp_results_dict, self.dist_records, self.hit_condi, self.stop_n_frame, self.undone_pts, self.current_pts, self.current_pts_n, self.suggest_ind, self.object_name = pickle.load(f)
            self.n_frame = self.stop_n_frame
        else:
            self.calculate_path()

    def key_binding(self):
        # root
        self.root.bind('<Left>', self.on_left)
        self.root.bind('<Right>', self.on_right)
        self.root.bind('<Up>', self.on_up)
        self.root.bind('<Down>', self.on_down)
        self.root.bind('<Prior>', self.on_page_up)
        self.root.bind('<Next>', self.on_page_down)
        self.root.bind('<Return>', self.on_return)
        self.root.bind('<z>', self.undo)
        self.root.bind('<BackSpace>', self.undo)
        self.root.bind('<Escape>', self.on_close)
        self.root.bind('<h>', self.on_settings)
        self.root.bind('1', self.on_key)
        self.root.bind('2', self.on_key)
        self.root.bind('3', self.on_key)
        self.root.bind('4', self.on_key)
        self.root.bind('5', self.on_key)
        self.root.bind('6', self.on_key)
        self.root.bind('<Delete>', self.on_key)
        self.root.bind('<a>', self.on_key)
        self.root.bind('<d>', self.on_key)
        self.root.bind('<r>', self.on_key)
        self.root.bind('<j>', self.on_key)
        self.root.bind('<q>', self.on_key)
        self.root.bind('<Control-s>', self.ask_save)

        # menu
        self.menu_file.add_command(label='載入新影像', command=self.on_load)
        self.menu_file.add_command(label='暫存操作', command=self.ask_save)
        self.menu_file.add_command(label='匯出資料', command=self.export)
        self.menu_help.add_command(label='設定', command=self.on_settings)

        # widget
        self.label_display.bind('<B1-Motion>', self.on_mouse_drag)
        self.label_display.bind('<Button-1>', self.on_mouse)
        self.label_display.bind('<Button-3>', self.on_mouse)
        self.label_display.bind('<Motion>', self.on_mouse_mv)
        self.scale_nframe_v.config(from_=1, to_=self.total_frame, command=self.set_nframe)
        self.scrollbar_object.config(command=self.treeview_object.yview)
        self.treeview_object.config(yscrollcommand=self.scrollbar_object.set)
        self.treeview_object.bind('<Double-Button-1>', self.tvitem_click)
        self.btn_return.config(command=self.on_return)
        self.btn_manual.config(command=self.on_manual_label)
        self.btn_reset.config(command=self.on_reset)
        self.btn_remove.config(command=self.on_remove)
        self.btn_replay.config(command=self.on_view)
        self.scale_max_path.config(command=self.set_max)

    def run(self):

        self.n_frame = 1
        self.last_n_frame = self.n_frame

        self._init_main_viewer()
        self.load_history()
        self.update_frame()
        self.update_info()
        self.draw()

        # convert to format that ImageTk require
        self.image = ImageTk.PhotoImage(Image.fromarray(self._frame))
        self.label_display.config(image=self.image)

        # load fixed legends
        legend_images = self._get_fixed_legends()
        legend_photos = map(Image.fromarray, legend_images)
        legend_photos = list(map(ImageTk.PhotoImage, legend_photos))
        for i in range(4):
            self.label_legend_images[i].config(image=legend_photos[i])

        # frame for display buttons
        self._render_op_buttons()

        # fixed op button
        photo = self._get_photo_from_path('button.png')
        self.label_suggest.config(image=photo)

        # operation buttons
        self.var_max_path.set(self.maximum)
        self.scale_max_path.set(self.maximum)

        # suggest default option
        if self.suggest_ind[0][0] == 'fp':
            ind = 0
        elif self.suggest_ind[0][0] == 'new':
            ind = 1
        else:
            ind = self.object_name[self.suggest_ind[0][0]]['ind'] + 2
        self.all_buttons[ind].focus_force()
        self.label_suggest.grid(row=ind, column=1, sticky="news", padx=5, pady=5)

        self.update_track(0)
        self.update_label()
        self.update_draw()
        self.center_root()
        self.root.update()
        self._init_height = self.root.winfo_height()
        self._init_width = self.root.winfo_width()

        self._r_height = self._frame.shape[0] / self._init_height
        self._r_width = self._frame.shape[1] / self._init_width

        self.root.mainloop()
