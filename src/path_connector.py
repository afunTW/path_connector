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
VIDEO_PATH = 'videos/[CH04] 2016-09-28 20.20.00_x264.avi'
# COLOR_NAME = ['limegreen', 'deepskyblue', 'YELLOW2', 'ORANGE', 'PURPLE', 'PINK', 'CYAN', 'BLACK', 'RED', 'WHITE']
COLOR_NAME = [('green', 'limegreen'), ('blue', 'deepskyblue'), ('yellow', 'gold'), ('purple', 'blueviolet'), ('orange', 'orange'), ('pink', 'pink'), ('cyan', 'cyan'), ('black', 'black'), ('red', 'red'), ('white', 'white')]
COLOR = [(50, 205, 50), (255, 191, 0), (0, 215, 255), (211, 85, 186), (0, 165, 255), (255, 102, 255), (255, 255, 0), (0, 0, 0), (100, 10, 255), (255, 255, 255)]

# UI required variables
# letter = [chr(i) for i in range(ord('A'), ord('Z')+1)]
letter = [str(i) for i in range(1, 20)]
LARGE_FONT= ("Verdana", 12)
MULTI = False

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
        self.video_path = None
        self.video = None
        self.video = None
        self.width = None
        self.height = None
        self.fps = None
        self.resolution = None
        self.total_frame = None
        self.__yolo_results__ = None
        self.__total_n_frame__ = None
        self.is_root_exist = True
        self.is_finish = False

        # basic variables
        self.color = COLOR
        self.color_name = COLOR_NAME
        # self.maximum = maximum
        self.maximum = 150
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
        self.object_name = dict()
        self.results_dict = dict()
        self.dist_records = dict()
        self.label_dict = dict()
        self.tmp_results_dict = dict()
        self.min_label_ind = None
        self.undone_pts = None
        self.fp_pts = []
        self.undo_records = []
        self.tracked_frames = []
        self.multi = MULTI
        self.suggest_ind = []
        self.current_pts = None
        self.current_pts_n = None
        self.safe = True
        self.is_calculate = False
        self.is_manual = False
        self.rat_cnt_dict = dict()
        self.hit_condi = None

        # variables for breaking from calculating loop
        self.n_run = None
        self.cancel_id = None

        # tkinter widgets
        self.root = None
        self.display_label = None
        self.tv = None
        self.label_nframe = None
        self.label_fps = None
        self.label_elapsed = None
        self.label_mvpts = None
        self.scale_nframe = None
        self.n_frame_var = None
        self.maximum_var = None
        self.tol_var = None
        self.BUTTON_FRAME = None
        self.check_show_yolo = None
        self.check_show_drawing = None
        self.check_show_rat = None
        self.all_buttons = []
        self.suggest_label = None
        self._init_width = None
        self._init_height = None
        self._r_height = None
        self._r_width = None
        self._c_height = None
        self._c_width = None

    def update_frame(self, ind=None):
        ind = ind if ind is not None else self.n_frame - 1
        self.video.set(cv2.CAP_PROP_POS_FRAMES, ind)
        ok, self._frame = self.video.read()
        self._orig_frame = self._frame.copy()

    def thread_update(self, ind):
        td = threading.Thread(target=self.update_frame, args=(ind, ))
        td.start()

    def thread_pop(self):
        td = threading.Thread(target=self.pop_msg, args=())
        td.start()

    def update_info(self):
        # update object information table
        if self.tv is not None:
            for n in sorted([k for k, v in self.object_name.items() if v['on']]):
                if self.is_manual:
                    rd = self.tmp_results_dict[n]
                else:
                    rd = self.results_dict[n]
                try:
                    is_detected = rd['n_frame'].index(self.n_frame)
                    is_detected = True
                except:
                    is_detected = False
                try:
                    self.tv.item(n, text=self.object_name[n]['display_name'], values=(self.color_name[self.object_name[n]['ind']][0], is_detected, rd['n_frame'][-1]))
                except:
                    self.object_name[n]['on'] = True
                    self.tv.insert('', 'end', n, text=n, values=(self.color_name[self.object_name[n]['ind']][0], is_detected, rd['n_frame'][-1]))

    def update_label(self):
        # text_nframe = 'Current Frame: '
        if self.is_root_exist:
            text_video_name = self.video_path.split('/')[-1]
            sec = round(self.n_frame / self.fps, 2)
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            text_time = "%d:%02d:%02d" % (h, m, s)

            self.label_video_name.configure(text='影像檔名: %s' % text_video_name)
            self.label_nframe_v.configure(text="當前幀數: %s/%s" % (self.n_frame, self.total_frame))
            self.label_time.configure(text='影像時間: %s' % text_time)
            self.scale_nframe.set(self.n_frame)

            self.display_label.after(200, self.update_label)

    def update_draw(self, tup=None):
        if self.is_root_exist:
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
                    self.display_label.configure(image=self.image)
                    self.display_label.image = self.image
            except:
                pass

            self.display_label.after(20, self.update_draw)

    def update_track(self, ind):
        if len(self.tracked_frames) > 0 and ind < (len(self.tracked_frames) - 1) and self.safe: #  and self.safe:
            frame = self.tracked_frames[ind]
            if ind < (len(self.tracked_frames) - 1):
                ind += 1
                self.scale_nframe.set(20*(ind) + 1)
                self.display_label.configure(image=frame)
                self.root.after(200, self.update_track, ind)
        else:
            self.scale_nframe.set(self.stop_n_frame)
            self.safe = False
            self.display_label.configure(image=self.image)

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
            # start
            self.run()

    def save_records(self):

        records = (copy.deepcopy(self.results_dict), copy.deepcopy(self.tmp_results_dict), {**self.dist_records}, copy.deepcopy(self.hit_condi), self.stop_n_frame, self.undone_pts, self.current_pts, self.current_pts_n, copy.deepcopy(self.suggest_ind), copy.deepcopy(self.object_name))

        if len(self.undo_records) > 0:
            if self.stop_n_frame != self.undo_records[-1][1]:
                self.undo_records.append(records)
        else:
            self.undo_records.append(records)
        # remove undo record if it is too long
        if len(self.undo_records) >= 15:
            self.undo_records = self.undo_records[-15:]

    def center_root(self, r=0):
        # self.root.update()
        self.root.update_idletasks()
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()

        size = (self.root.winfo_width(), self.root.winfo_height())
        x = w/2 - size[0]/2
        y = h/2.25 - size[1]/2
        r = 0 if self.root.state() == 'zoomed' else r
        self.root.geometry("%dx%d+%d+%d" % (size[0], size[1]+r, x, y))

    # main logic for runing UI
    def run(self):

        self.n_frame = 1
        self.last_n_frame = self.n_frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.n_frame - 1)
        ok, self._frame = self.video.read()
        self._orig_frame = self._frame.copy()
        self.draw_legend()

        if not ok:
            self.msg("Can't open the video")

        self.root = tk.Tk()
        if os.name == 'nt':
            self.root.iconbitmap('beetle.ico')
        self.root.title(self.win_name)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.bind('<Left>', self.on_left)
        self.root.bind('<Right>', self.on_right)
        self.root.bind('<Up>', self.on_up)
        self.root.bind('<Down>', self.on_down)
        self.root.bind('<Prior>', self.on_page_up)
        self.root.bind('<Next>', self.on_page_down)
        self.root.focus_force()
        self.root.option_add('*tearOff', False)
        self.root.option_add("*Font", "verdana 10")
        self.root.aspect(1, 1, 1, 1)
        tk.Grid.rowconfigure(self.root, 0, weight=1)
        tk.Grid.columnconfigure(self.root, 0, weight=1)
        # tk.Grid.columnconfigure(self.root, 1, weight=1)

        file = self.video_path.split('.avi')[0] + '.dat'
        if os.path.isfile(file):
            with open(file, "rb") as f:
                self.results_dict, self.tmp_results_dict, self.dist_records, self.hit_condi, self.stop_n_frame, self.undone_pts, self.current_pts, self.current_pts_n, self.suggest_ind, self.object_name = pickle.load(f)
            self.n_frame = self.stop_n_frame
        else:
            self.calculate_path()
        self.update_frame()
        self.update_info()
        self.draw()

        # create a menu instance
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        # create the file object
        file = tk.Menu(menu)
        file.add_command(label='載入新影像', command=self.on_load)
        file.add_command(label='暫存操作', command=self.ask_save)
        file.add_command(label='匯出資料', command=self.export)
        menu.add_cascade(label='File', menu=file)

        # create the help object
        help = tk.Menu(menu)
        help.add_command(label='設定', command=self.on_settings)
        menu.add_cascade(label='Help', menu=help)

        # label for frame information
        self.n_frame_var = tk.IntVar()
        self.n_frame_var.set(self.n_frame)
        self.maximum_var = tk.IntVar()
        self.maximum_var.set(self.maximum)
        self.tol_var = tk.DoubleVar()
        self.tol_var.set(self.tol)
        self.check_show_yolo = tk.IntVar()
        self.check_show_yolo.set(1)
        self.check_show_drawing = tk.IntVar()
        self.check_show_drawing.set(1)
        self.check_show_rat = tk.IntVar()
        self.check_show_rat.set(0)
        self.check_is_clear = tk.IntVar()
        self.check_is_clear.set(1)

        text_video_name =self.video_path.split('/')[-1]
        sec = round(self.n_frame / self.fps, 2)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        text_time = "%d:%02d:%02d" % (h, m, s)

        # convert to format that ImageTk require
        self.image = ImageTk.PhotoImage(Image.fromarray(self._frame))

        # frame for displaying image label
        IMAGE_FRAME = ttk.Frame(self.root)
        IMAGE_FRAME.grid(row=0, column=0)
        IMAGE_FRAME.grid_columnconfigure(0, weight=1)
        IMAGE_FRAME.grid_rowconfigure(0, weight=1)
        IMAGE_FRAME.grid_rowconfigure(1, weight=1)

        self.display_frame = tk.Frame(IMAGE_FRAME)
        self.display_frame.grid(row=0, column=0)
        self.display_label = ttk.Label(self.display_frame, image=self.image)
        self.display_label.grid()
        self.display_label.bind('<B1-Motion>', self.on_mouse_drag)
        self.display_label.bind('<Button-1>', self.on_mouse)
        self.display_label.bind('<Button-3>', self.on_mouse)
        self.display_label.bind('<Motion>', self.on_mouse_mv)

        IMAGE_LABEL_FRAME = ttk.LabelFrame(IMAGE_FRAME)
        IMAGE_LABEL_FRAME.grid(sticky='news', padx=5, pady=5)
        tk.Grid.rowconfigure(IMAGE_LABEL_FRAME, 0, weight=1)
        tk.Grid.rowconfigure(IMAGE_LABEL_FRAME, 1, weight=1)
        tk.Grid.columnconfigure(IMAGE_LABEL_FRAME, 0, weight=1)
        self.label_nframe_v = ttk.Label(IMAGE_LABEL_FRAME, text="當前幀數: %s/%s" % (self.n_frame, self.total_frame))
        self.label_nframe_v.grid(row=0, column=0)
        self.scale_nframe = ttk.Scale(IMAGE_LABEL_FRAME, from_=1, to_=self.total_frame, command=self.set_nframe)
        self.scale_nframe.configure(cursor='hand2')
        self.scale_nframe.set(self.n_frame)
        self.scale_nframe.grid(row=1, column=0, sticky='news', padx=10)

        # operation frame that will display all application relevent information
        self.OP_FRAME = ttk.Frame(self.root)
        self.OP_FRAME.grid(row=0, column=1, sticky="news", padx=5, pady=5)
        self.OP_FRAME.grid_rowconfigure(0, weight=1)
        self.OP_FRAME.grid_rowconfigure(1, weight=1)
        self.OP_FRAME.grid_rowconfigure(2, weight=1)
        self.OP_FRAME.grid_columnconfigure(0, weight=1)

        # frame for displaying current statement
        STATE_FRAME = ttk.Frame(self.OP_FRAME)
        STATE_FRAME.grid(row=0, column=0, sticky="news", padx=5, pady=5)
        STATE_FRAME.grid_rowconfigure(0, weight=1)
        STATE_FRAME.grid_rowconfigure(1, weight=1)
        STATE_FRAME.grid_rowconfigure(2, weight=1)
        STATE_FRAME.grid_columnconfigure(0, weight=1)

        # subframe for displaying frame information
        INFO_FRAME = ttk.LabelFrame(STATE_FRAME, text='影像訊息')
        INFO_FRAME.grid(sticky="news", padx=5, pady=5)
        INFO_FRAME.grid_rowconfigure(0, weight=1)
        INFO_FRAME.grid_rowconfigure(1, weight=1)
        INFO_FRAME.grid_columnconfigure(0, weight=1)

        self.label_video_name = ttk.Label(INFO_FRAME, text='影像檔名: %s' % text_video_name)
        self.label_video_name.grid(row=0, column=0, sticky=tk.W)
        self.label_time = ttk.Label(INFO_FRAME, text='影像時間: %s' % text_time)
        self.label_time.grid(row=1, column=0, sticky=tk.W)

        # subframe for displaying object information
        OBJ_FRAME = ttk.LabelFrame(STATE_FRAME, text='目標資訊')
        OBJ_FRAME.grid(sticky="news", padx=5, pady=5)
        OBJ_FRAME.grid_rowconfigure(0, weight=1)
        OBJ_FRAME.grid_columnconfigure(0, weight=1)
        OBJ_FRAME.grid_columnconfigure(1, weight=1)

        self.tv = ttk.Treeview(OBJ_FRAME, height = 3)
        self.tv['columns'] = ('color', 'lastpoint', 'lastdetectedframe')
        self.tv.heading('#0', text='名稱', anchor='center')
        self.tv.column('#0', anchor='w', width=50)
        self.tv.heading('color', text='顏色')
        self.tv.column('color', anchor='center', width=60)
        self.tv.heading('lastpoint', text='在本幀是否有被偵測到')
        self.tv.column('lastpoint', anchor='center', width=130)
        self.tv.heading('lastdetectedframe', text='最後被偵測到的幀數')
        self.tv.column('lastdetectedframe', anchor='center', width=130)
        self.tv.grid(row=0, column=0, sticky='news', pady=3)
        vsb = ttk.Scrollbar(OBJ_FRAME, orient='vertical', command=self.tv.yview)
        vsb.grid(row=0, column=1, sticky='news', pady=3)
        self.tv.configure(yscrollcommand=vsb.set)

        for n in sorted(self.object_name.keys()):
            rd = self.results_dict[n]
            try:
                is_detected = rd['n_frame'].index(self.n_frame)
                is_detected = True
            except:
                is_detected = False
            self.tv.insert('', 'end', n, text=n, values=(self.color_name[self.object_name[n]['ind']][0], is_detected, rd['n_frame'][-1]))
        self.tv.bind('<Double-Button-1>', self.tvitem_click)

        # frame for legend
        LEGENG_FRAME = ttk.LabelFrame(STATE_FRAME, text='圖例說明')
        LEGENG_FRAME.grid(sticky="news", padx=5, pady=5)
        for x in range(4):
            LEGENG_FRAME.grid_columnconfigure(x, weight=1)
        LEGENG_FRAME.grid_rowconfigure(0, weight=1)
        LEGENG_FRAME.grid_rowconfigure(1, weight=1)

        l_1 = ImageTk.PhotoImage(Image.fromarray(self.legend_1))
        l_2 = ImageTk.PhotoImage(Image.fromarray(self.legend_2))
        l_3 = ImageTk.PhotoImage(Image.fromarray(self.legend_3))
        l_4 = ImageTk.PhotoImage(Image.fromarray(self.legend_4))

        legend_2 = ttk.Label(LEGENG_FRAME, image=l_2, width=10)
        legend_2.grid(row=0, column=0, padx=3, pady=3)
        legend_1 = ttk.Label(LEGENG_FRAME, image=l_1, width=10)
        legend_1.grid(row=0, column=1, padx=3, pady=3)
        legend_3 = ttk.Label(LEGENG_FRAME, image=l_3, width=10)
        legend_3.grid(row=0, column=2, padx=3, pady=3)
        legend_4 = ttk.Label(LEGENG_FRAME, image=l_4, width=10)
        legend_4.grid(row=0, column=3, padx=3, pady=3)

        label_legend_2 = ttk.Label(LEGENG_FRAME, text='需被標註的\nbbox', width=10, anchor='center')
        label_legend_2.grid(row=1, column=0, padx=3, pady=3)
        label_legend_1 = ttk.Label(LEGENG_FRAME, text='目標路徑起點', width=10, anchor='center')
        label_legend_1.grid(row=1, column=1, padx=3, pady=3)
        label_legend_3 = ttk.Label(LEGENG_FRAME, text='目標在本幀的位置', width=13, anchor='center')
        label_legend_3.grid(row=1, column=2, padx=3, pady=3)
        label_legend_4 = ttk.Label(LEGENG_FRAME, text='目標路徑終點', width=10, anchor='center')
        label_legend_4.grid(row=1, column=3, padx=3, pady=3)

        # frame for display buttons
        self.BUTTON_FRAME = ttk.LabelFrame(self.OP_FRAME, text="需被標註的 bbox 應該是哪一個目標呢？")
        self.BUTTON_FRAME.grid(row=1, column=0, sticky="news", padx=5, pady=5)
        self.BUTTON_FRAME.grid_columnconfigure(0, weight=1)
        for x in range(8):
            self.BUTTON_FRAME.grid_rowconfigure(x, weight=1)

        for i, k in enumerate(['誤判 (d)', '新目標 (a)'] + sorted(self.object_name.keys())):
            if i in [0, 1]:
                bg = None
                b = ttk.Button(self.BUTTON_FRAME, text=k, command=lambda clr=k: self.on_button(clr), bg=bg, width=40)
                b.grid(row=i, column=0, sticky="news", padx=10, pady=2)

            else:
                bg = self.color_name[self.object_name[k]['ind']][1].lower()
                b = tk.Button(self.BUTTON_FRAME, text=self.object_name[k]['display_name'], command=lambda clr=k: self.on_button(clr), bg=bg)
                b.grid(row=i, column=0, sticky="news", padx=10, pady=2)
            b.config(cursor='hand2')

            self.all_buttons.append(b)

        image = Image.open('button.png')
        photo = ImageTk.PhotoImage(image)
        self.suggest_label = ttk.Label(self.BUTTON_FRAME, image=photo)
        # self.suggest_label

        self.BUTTON_FRAME_2 = ttk.LabelFrame(self.OP_FRAME, text='操作')
        self.BUTTON_FRAME_2.grid(row=2, column=0, sticky="news", padx=5, pady=3)
        self.BUTTON_FRAME_2.grid_columnconfigure(0, weight=1)
        self.BUTTON_FRAME_2.grid_columnconfigure(1, weight=1)
        for x in range(8):
            self.BUTTON_FRAME_2.grid_rowconfigure(x, weight=1)

        # operation buttons
        button_go = ttk.Button(self.BUTTON_FRAME_2, text='回到需被標註的幀數 (Enter)', command=self.on_return, cursor='hand2')
        button_go.grid(row=0, columnspan=2, sticky="news", padx=10, pady=2)

        button_manual = ttk.Button(self.BUTTON_FRAME_2, text='進入 / 離開 Manual Label (q)', command=self.on_manual_label, cursor='hand2')
        button_manual.grid(row=1, columnspan=2, sticky="news", padx=10, pady=2)

        button_reset = ttk.Button(self.BUTTON_FRAME_2, text='重置 (r)', command=self.on_reset, cursor='hand2')
        button_reset.grid(row=2, columnspan=2, sticky="news", padx=10, pady=2)

        button_remove = ttk.Button(self.BUTTON_FRAME_2, text='刪除目標', command=self.on_remove, cursor='hand2')
        button_remove.grid(row=3, columnspan=2, sticky="news", padx=10, pady=2)

        button_replay = ttk.Button(self.BUTTON_FRAME_2, text='回放已追踪路徑', command=self.on_view, cursor='hand2')
        button_replay.grid(row=4, columnspan=2, sticky="news", padx=10, pady=2)

        label_max = ttk.Label(self.BUTTON_FRAME_2, text='顯示路徑的長度: ')
        label_max.grid(row=5, column=0, sticky=tk.W)
        label_max_v = ttk.Label(self.BUTTON_FRAME_2, textvariable=self.maximum_var)
        label_max_v.grid(row=5, column=1)
        scale_max = ttk.Scale(self.BUTTON_FRAME_2, from_=2, to_=3000, length=200, command=self.set_max)
        scale_max.set(self.maximum)
        scale_max.configure(cursor='hand2')
        scale_max.grid(row=5, column=1)

        # checkboxes
        check_show_box = ttk.Checkbutton(self.BUTTON_FRAME_2, variable=self.check_show_yolo, onvalue=1, offvalue=0, text='顯示埋葬蟲 bounding box')
        check_show_box.grid(row=6, column=0, sticky="news", padx=10, pady=5)

        check_is_clear = ttk.Checkbutton(self.BUTTON_FRAME_2, variable=self.check_is_clear, onvalue=1, offvalue=0, text='透鏡')
        check_is_clear.grid(row=7, column=0, sticky="news", padx=10, pady=5)

        check_show_rat = ttk.Checkbutton(self.BUTTON_FRAME_2, variable=self.check_show_rat, onvalue=1, offvalue=0, text='顯示老鼠輪廓')
        check_show_rat.grid(row=6, column=1, sticky="news", padx=10, pady=5)

        check_show_drawing = ttk.Checkbutton(self.BUTTON_FRAME_2, variable=self.check_show_drawing, onvalue=1, offvalue=0, text='顯示已追踪路徑')
        check_show_drawing.grid(row=7, column=1, sticky="news", padx=10, pady=5)

        # suggest default option
        LOGGER.info(self.suggest_ind)
        if self.suggest_ind[0][0] == 'fp':
            ind = 0
        elif self.suggest_ind[0][0] == 'new':
            ind = 1
        else:
            ind = self.object_name[self.suggest_ind[0][0]]['ind'] + 2
        self.all_buttons[ind].focus_force()
        self.suggest_label.grid(row=ind, column=1, sticky="news", padx=5, pady=5)

        self.update_track(0)
        self.update_label()
        self.update_draw()
        # center
        self.center_root()
        self.root.update()
        self._init_height = self.root.winfo_height()
        self._init_width = self.root.winfo_width()

        self._r_height = self._frame.shape[0] / self._init_height
        self._r_width = self._frame.shape[1] / self._init_width

        # self.root.minsize(width=self._init_width, height=self._init_height)

        # bind key and corresponding function
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
        if os.name == 'nt':
            self.root.state('zoomed')
        else:
            self.root.attributes('-zoomed', True)
        self.root.mainloop()
