import json
import logging
import os
import pickle
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import askokcancel
from tkinter.messagebox import askyesno
from tkinter.messagebox import showerror
from tkinter.messagebox import showinfo
from tkinter.messagebox import showwarning

import numpy as np
import pandas as pd

from src.listbox import Tk_Table

LOGGER = logging.getLogger(__name__)

class Interface(object):

    # show warning
    def msg(self, string):
        showinfo('Info', string)

    # confirm leave
    def on_close(self, event=None):
        if not self.is_calculate:
            if askokcancel('離開', '你確定要關閉程式了嗎？'):
                if len(self.undo_records) > 1:
                    with open( "%s.dat" % self.video_path.split('.avi')[0], "wb" ) as f:
                        pickle.dump(self.undo_records[-1], f)
                self.root.quit()
                self.root.destroy()
        else:
            self.cancel_calc()

    def ask_save(self, style='replace'):
        filename = "%s.dat" % self.video_path.split('.avi')[0]
        if style == 'replace':
            if os.path.isfile(filename):
                if askyesno('存檔', '檔案已存在, 是否要覆蓋原本的檔案?'):
                    with open(filename, "wb" ) as f:
                        pickle.dump(self.undo_records[-1], f)
            else:
                with open(filename, "wb" ) as f:
                    pickle.dump(self.undo_records[-1], f)
        elif style == 'confirm':
            if askyesno('存檔', '要把操作結果存檔嗎?'):
                with open(filename, "wb" ) as f:
                    pickle.dump(self.undo_records[-1], f)

    # confirm for replacement
    def ask_yes_no(self, clr, title='確認', icontype='warning'):
        string = '%s 在本幀已經有被分配的 bounding box 咯!\n你確定要取代原本的嗎？' % self.object_name[clr]['display_name']
        result = askyesno(title, string, icon=icontype)

        return result

    # ask for desired file name
    def get_path(self, label=None, res=False):
        if os.name == 'nt':
            path = askopenfilename(
                title='請選擇影像路徑 (*.avi;*.mp4)',
                filetypes=[('video file (*.avi;*.mp4)', '*.avi;*.mp4')])
        else:
            path = askopenfilename(title='請選擇影像路徑 (*.avi;*.mp4)')

        if path != "":
            if not res:
                self.video_path = path
            else:
                return path
            if label is not None:
                label.configure(text='%s' % self.video_path.split('/')[-1])

    def pop_msg(self):
        self.temp_root = tk.Tk()
        label = tk.Label(self.temp_root, text='Calculating...please wait...')
        label.pack()
        self.temp_root.mainloop()

    def setting(self):
        settings_root = tk.Tk()
        tk.Grid.rowconfigure(settings_root, 0, weight=1)
        tk.Grid.columnconfigure(settings_root, 0, weight=1)
        tk.Grid.columnconfigure(settings_root, 1, weight=1)

        def exit(event):
            settings_root.destroy()

        settings_root.focus_force()
        settings_root.title('設定')

        ACTION = ['標註對應的目標', '誤判', '新目標', '返回', '重置', '前一幀', '後一幀', '前五幀', '後五幀', '選擇 default 選項', '進入/離開 Manual Label', '回到需被標註的幀數', '移動幀數', '設定']
        HOTKEY = ['1/2/3/4', 'd/DELETE', 'a', 'z/BACKSPACE', 'r', 'LEFT', 'RIGHT', 'PAGE DOWN', 'PAGE UP', 'DOWN', 'q', 'ENTER',  'j', 'h']

        hotkey = ttk.LabelFrame(settings_root, text="快捷鍵")
        action = ttk.LabelFrame(settings_root, text="操作")
        hotkey.grid_columnconfigure(0, weight=1)
        action.grid_columnconfigure(0, weight=1)
        hotkey.grid(row=0, column=0, padx=5, pady=5, sticky='news')
        action.grid(row=0, column=1, padx=5, pady=5, sticky='news')

        # action description section
        for i, a in enumerate(ACTION):
            ttk.Label(action, text=a).grid(column=0, row=i, sticky=tk.W, padx=5, pady=5)
            ttk.Label(hotkey, text=HOTKEY[i]).grid(column=0, row=i, padx=5, pady=5)

            hotkey.grid_rowconfigure(i, weight=1)
            action.grid_rowconfigure(i, weight=1)


        settings_root.bind('<Escape>', exit)
        settings_root.bind('<h>', exit)
        self.center(settings_root)
        settings_root.mainloop()

    def export(self):
        root = '/'.join(self.video_path.split('/')[:-1])
        filename = self.video_path.split('/')[-1]
        filename = "%s" % filename.split('.avi')[0]
        filename = asksaveasfilename(initialdir='%s' % (root),
                                     defaultextension=".csv",
                                     filetypes=(("CSV", "*.csv"),("All Files", "*.*")),
                                     initialfile=filename,
                                     title='匯出路徑')

        if filename not in ["", None]:
            with open(filename.split(".csv")[0]+".json", 'w+') as f:
                json.dump({"res": self.results_dict, "name": self.object_name}, f)
            df = {'name': [], 'nframe': [], 'x_center': [], 'y_center': [], 'width': [], 'height': []}
            data = self.results_dict
            for k in sorted(data.keys()):
                frame = data[k]['n_frame']
                paths = np.array(data[k]['path'])
                wh = np.array(data[k]['wh'])
                x, y = paths[:, 0].tolist(), paths[:, 1].tolist()
                w, h = wh[:, 0].tolist(), wh[:, 1].tolist()
                df['name'].extend([self.object_name[k]['display_name']]*len(frame))
                df['nframe'].extend(frame)
                df['x_center'].extend(x)
                df['y_center'].extend(y)
                df['width'].extend(w)
                df['height'].extend(h)
            df = pd.DataFrame(df)
            df = df.reindex_axis(['name', 'nframe', 'x_center', 'y_center', 'width', 'height'], axis=1)
            df.to_csv(filename, index=False)

            self.msg("標註軌跡已存檔於 %s" % filename)
        # filename = "%s_rat_contour.json" % self.video_path.split('.avi')[0]
        # save rat contour
        # with open(filename, 'a') as f:
        #     json.dump(self.rat_cnt_dict, f)

    def pop_behavior_table(self):

        top = tk.Toplevel(self.root)
        top.title("行為記錄")
        table = Tk_Table(top, ["幀數","名稱", "行為"], row_numbers=True, stripped_rows = ("white","#f2f2f2"), select_mode="none")
        table.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky='news')
        table.focus_force()

        button = ttk.Button(top, text='Add', command=lambda: table.insert_row([100, 'A', 'Attack']))
        button.grid(column=0, row=1, padx=10, pady=10, sticky='news')
        table.insert_row([1, 'A', 'Attack'])
        table.insert_row([2, 'B', 'Wrestle'])
        table.insert_row([30, 'C', 'Chase'])
        table.insert_row([45, 'D', 'Escape'])

        self.center(top)
        top.bind('<Escape>', lambda event: top.destroy())
        top.protocol('WM_DELETE_WINDOW', lambda : top.destroy())

    class popupEntry(object):

        def __init__(self,master, title, string, validnum=False):
            top=self.top= tk.Toplevel(master)
            top.title(title)
            tk.Grid.rowconfigure(top, 0, weight=1)
            tk.Grid.columnconfigure(top, 0, weight=1)
            top.transient(master)
            top.grab_set()

            self.l=tk.Label(top,text=string, font=("Verdana", 12))
            self.l.pack(expand=tk.YES, fill=tk.BOTH, padx=5, pady=5)
            if validnum:
                vcmd = (master.register(self.validate),
                    '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
                self.e =ttk.Entry(top, validate='key', validatecommand=vcmd)
            else:
                self.e = ttk.Entry(top)
            self.e.config(width=9)
            self.e.pack(expand=tk.YES, padx=5, pady=5)
            self.e.focus_force()
            self.e.bind('<Return>', lambda event: self.cleanup())
            self.b = ttk.Button(top,text='Ok',command=self.cleanup, width=5)
            self.b.pack(expand=tk.YES, padx=5, pady=5)

            top.update_idletasks()
            width = top.winfo_reqwidth() + 10
            height = top.winfo_reqheight() + 10
            x = (top.winfo_screenwidth() // 2.25) - (width // 2)
            y = (top.winfo_screenheight() // 2) - (height // 2)
            top.geometry('+%d+%d' % (x, y))
            top.geometry('260x120')

            top.bind('<Escape>', lambda event: top.destroy())

            # pending; bind return, a decent name judge
        def cleanup(self):
            self.value=self.e.get()
            self.top.destroy()

        def validate(self, action, index, value_if_allowed,
                           prior_value, text, validation_type, trigger_type, widget_name):
            if text in '0123456789':
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    return False
            else:
                return False
