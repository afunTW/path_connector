import cv2
from src.interface import Interface
from src.utils import Common
from PIL import Image, ImageTk
import tkinter as tk
import threading
from tkinter import ttk
import imageio
from tkinter.messagebox import askyesno
import numpy as np
import copy, os

letter = [chr(i) for i in range(ord('a'), ord('z')+1)]

class KeyHandler(Interface, Common):

    # reload the interface after reloading a new video; pending, add judgement for new video (like if a YOLO txt file existed)
    def on_load(self):
        path = self.get_path(res=True)
        yolo_results_path = path.split('.avi')[0] + '.txt'
        res = os.path.isfile(yolo_results_path)
        
        if not res:
            self.msg('影像檔案路徑底下沒有對應的 YOLO txt。')
        elif path in [None, ""]:
            self.msg('請載入影像檔案。')
        else:
            self.video_path = path
            self.video.release()
            self.init_video()
            self.object_name = dict()
            self.results_dict = dict()
            self.tmp_results_dict = None
            self.dist_records = dict()
            self.label_dict = dict()
            self.undo_records = []
            self.drag_flag = None
            if self.is_manual:
                self.chg_mode()
            
            self.n_frame = 1
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.n_frame - 1)
            ok, self._frame = self.video.read()
            self._orig_frame = self._frame.copy()
            
            self.calculate_path(self.n_frame)
            
            # reset button
            for b in self.all_buttons:
                b.grid_forget()
            self.all_buttons = []

            on_ind = [v['ind'] for k, v in self.object_name.items() if v['on']]
            for i, k in enumerate(['誤判', '新目標'] + sorted(self.object_name.keys())):
                if i in [0, 1]:
                    bg = None
                    fg = None
                    b = ttk.Button(self.BUTTON_FRAME, text=k, command=lambda clr=k: self.on_button(clr), bg=bg, fg=fg, width=40)
                    b.grid(row=i, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)

                else:
                    bg = self.color_name[self.object_name[k]['ind']].lower()
                    fg = 'white'
                    b = tk.Button(self.BUTTON_FRAME, text=self.object_name[k]['display_name'], command=lambda clr=k: self.on_button(clr), bg=bg, fg=fg)
                    b.grid(row=i, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)

                self.all_buttons.append(b)

            # reset table information
            x = self.tv.get_children()
            for item in x:
                self.tv.delete(item)

    # popup a description widget
    def on_settings(self, event=None):
        self.setting()

    def reset(self, event):
        self.tmp_line = []

    # record position of current mosue cursor and cursor's type while it's on any object
    def on_mouse_mv(self, event):
        self.clear = True
        self.last_x = self.mv_x
        self.last_y = self.mv_y
        self.mv_x = event.x
        self.mv_y = event.y

        # update cursor
        if self.is_manual:
            if self.drag_flag != 'new':
                self.drag_flag = None
                for k, v in self.tmp_results_dict.items():
                    path = v['path']
                    flag = v['n_frame']
                    try:
                        ind = max([flag.index(f) for f in flag if f <= self.n_frame])
                    except:
                        ind = 0
                    if self.in_circle((self.mv_x, self.mv_y), path[ind], 15):
                        self.drag_flag = k
                        break

                if self.drag_flag is not None:
                    self.display_label.config(cursor='hand2')
                else:
                    self.display_label.config(cursor='arrow')
            else:
                self.display_label.config(cursor='hand2')

    # switch between normal mode and manual label mode
    def chg_mode(self):
        self.is_manual = not self.is_manual
        # modify state of buttons for object index assignment
        for i, b in enumerate(self.all_buttons):
            if i != 1:
                b['state'] = 'disabled' if self.is_manual else 'normal'

        if self.is_manual:
            # temporally results for manual label
            self.tmp_results_dict = copy.deepcopy(self.results_dict)
            self.save_records()
            # switch off drawing removal
            self.check_is_clear.set(0)
        else:
            # switch on drawing removal
            self.check_is_clear.set(1)

    def on_manual_label(self):
        if self.is_manual:
            self.leave_manual_label()
        else:
            self.chg_mode()

    # left click enter manual label mode, right click remove label
    def on_mouse(self, event):
        n = event.num
        # if double click while normal mode, enter manual label mdoe
        if n == 1 and not self.is_manual:
            pass
            # self.chg_mode()
        # if right click while manual label mode
        elif n == 3 and self.is_manual:
            self.undo_manual()
        # if double left click while manual label mode
        elif self.is_manual and self.drag_flag == 'new':
            # pending; add a UI to confirm adding object
            p = (event.x, event.y)
            new_key = letter[len(self.object_name)]
            self.min_label_ind = self.n_frame if self.min_label_ind is None else min(self.n_frame, self.min_label_ind)
            self.label_dict[new_key] = {'path': [p], 'n_frame': [self.n_frame]}
            self.tmp_results_dict[new_key] = {'path': [p], 'n_frame': [self.n_frame]}
            self.object_name[new_key] = {'ind': len(self.object_name), 'on': True, 'display_name': new_key}

            try:
                self.dist_records[self.n_frame][new_key] = dict()
            except:
                self.dist_records[self.n_frame] = dict()
                self.dist_records[self.n_frame][new_key] = dict()
            self.dist_records[self.n_frame][new_key]['dist'] = [0]
            self.dist_records[self.n_frame][new_key]['center'] = [p]
            self.dist_records[self.n_frame][new_key]['below_tol'] = [True]

            # add buttons
            bg = self.color_name[self.object_name[new_key]['ind']].lower()
            b = tk.Button(self.BUTTON_FRAME, text=new_key, command=lambda clr=new_key: self.on_button(clr), bg=bg, fg='white')
            b.grid(row=len(self.all_buttons) + 2, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
            b.config(state='disabled')
            self.all_buttons.append(b)
            self.center_root(r=35)

            self.display_label.config(cursor='arrow')
            self.drag_flag = None

    # drag method for manual label mode
    def on_mouse_drag(self, event):
        # drag is available while the cursor changed
        if self.is_manual and self.drag_flag not in [None, 'new']:
            flags = self.tmp_results_dict[self.drag_flag]['n_frame']
            path = self.tmp_results_dict[self.drag_flag]['path']
            # get index of path if existed and then update with new coordinate
            try:
                ind = flags.index(self.n_frame)
                self.tmp_results_dict[self.drag_flag]['path'][ind] = (event.x, event.y)
                self.min_label_ind = self.n_frame if self.min_label_ind is None else min(self.min_label_ind, self.n_frame)

            # otherwise, find the minimum index that are bigger than self.frame
            except:
                tmp = [flags.index(v) for v in flags if v >= self.n_frame]
                # concatenate the coordinate among corresponding index if the minimum exists
                if len(tmp) > 0:
                    ind = min(tmp)
                    self.tmp_results_dict[self.drag_flag]['n_frame'] = flags[:ind] + [self.n_frame] + flags[ind:]
                    self.tmp_results_dict[self.drag_flag]['path'] = path[:ind] + [(event.x, event.y)] + path[ind:]
                    self.min_label_ind = self.n_frame if self.min_label_ind is None else min(self.min_label_ind, self.n_frame)
                # otherwise, append the coordinate on the end of the path
                else:
                    self.tmp_results_dict[self.drag_flag]['n_frame'].append(self.n_frame)
                    self.tmp_results_dict[self.drag_flag]['path'].append((event.x, event.y))

            # record for label
            if self.drag_flag not in self.label_dict.keys():
                self.label_dict[self.drag_flag] = {'path': [], 'n_frame': []}
            flags = self.label_dict[self.drag_flag]['n_frame']
            path = self.label_dict[self.drag_flag]['path']
            try:
                ind = flags.index(self.n_frame)
                self.label_dict[self.drag_flag]['path'][ind] = (event.x, event.y)
            except:
                tmp = [flags.index(v) for v in flags if v >= self.n_frame]
                if len(tmp) > 0:
                    ind = min(tmp)
                    self.label_dict[self.drag_flag]['n_frame'] = flags[:ind] + [self.n_frame] + flags[ind:]
                    self.label_dict[self.drag_flag]['path'] = path[:ind] + [(event.x, event.y)] + path[ind:]
                # otherwise, append the coordinate on the end of the path
                else:
                    self.label_dict[self.drag_flag]['n_frame'].append(self.n_frame)
                    self.label_dict[self.drag_flag]['path'].append((event.x, event.y))

    # left click append label record; not used anymore
    def on_mouse_manual_label(self, event):
        # execute only if it is manual label mode
        if self.is_manual:
            # manual label for new object
            p = (event.x, event.y)
            if self.label_ind == 0:
                # pending; ask new object UI
                new_key = letter[len(self.object_name)]
                self.label_dict[new_key] = {'path': [p], 'n_frame': [self.n_frame]}
                self.tmp_results_dict[new_key] = {'path': [p, p], 'n_frame': [self.n_frame, self.n_frame]}
                self.object_name[new_key] = {'ind': len(self.object_name), 'on': True, 'display_name': new_key}

                # self.results_dict[new_key] = {'path': [p, p], 'n_frame': [n, n]}
                try:
                    self.dist_records[self.n_frame][new_key] = dict()
                except:
                    self.dist_records[self.n_frame] = dict()
                    self.dist_records[self.n_frame][new_key] = dict()
                self.dist_records[self.n_frame][new_key]['dist'] = [0]
                self.dist_records[self.n_frame][new_key]['center'] = [p]
                self.dist_records[self.n_frame][new_key]['below_tol'] = [True]

                # add buttons
                bg = self.color_name[self.object_name[new_key]['ind']].lower()
                b = tk.Button(self.BUTTON_FRAME, text=new_key, command=lambda clr=new_key: self.on_button(clr), bg=bg, fg='white')
                b.grid(row=len(self.all_buttons) + 2, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
                b.config(state='disabled')
                self.all_buttons.append(b)
                self.label_ind = 1 + max([v['ind'] for k, v in self.object_name.items() if v['on']])

            # manual label for existing object
            else:
                k = [k for k, v in self.object_name.items() if v['ind'] == self.label_ind - 1][0]
                # record label points
                if self.n_frame not in self.label_dict[k]['n_frame']:
                    self.label_dict[k]['n_frame'].append(self.n_frame)
                    self.label_dict[k]['path'].append(p)
                else:
                    self.label_dict[k]['path'][self.label_dict[k]['n_frame'].index(self.n_frame)] = p

                # modify label point if it conflicts with original result dicts
                try:
                    ind = self.tmp_results_dict[k]['n_frame'].index(self.n_frame)
                    # pending; conflict with existing path, ask reassign UI?
                    self.tmp_results_dict[k]['path'][ind] = p
                except:
                    self.tmp_results_dict[k]['n_frame'].append(self.n_frame)
                    self.tmp_results_dict[k]['path'].append(p)

    # button event
    def on_button(self, clr):
        
        if not self.is_manual:
            self.save_records()
            self.n_frame = self.stop_n_frame
            p, n = self.current_pts, self.current_pts_n
        run = True
        replace = False

        if clr in [k for k, v in self.object_name.items() if v['on']]:
            is_assigned = self.results_dict[clr]['n_frame'][-1] == self.stop_n_frame
            if not is_assigned:
                self.results_dict[clr]['path'].append(p)
                self.results_dict[clr]['n_frame'].append(n)
                print('appended!')
            else:
                res = self.ask_yes_no(clr)
                if res:
                    self.undone_pts.append((self.results_dict[clr]['path'][-1], n))
                    print(self.undone_pts)
                    self.results_dict[clr]['path'][-1] = p
                    print('appended!')
                    run = True
                    replace = True
                else:
                    run = False
        elif clr == '新目標':
            if len(self.object_name) < 6:
                if not self.is_manual:
                    # append results
                    new_key = letter[len(self.object_name)]
                    self.results_dict[new_key] = {'path': [p], 'n_frame': [n]}
                    print(len(self.object_name), self.object_name)
                    self.object_name[new_key] = {'ind': len(self.object_name), 'on': True, 'display_name': new_key}
                    print(len(self.object_name))
                    try:
                        self.dist_records[n][new_key] = dict()
                    except:
                        self.dist_records[n] = dict()
                        self.dist_records[n][new_key] = dict()
                    self.dist_records[n][new_key]['dist'] = [0]
                    self.dist_records[n][new_key]['center'] = [p]
                    self.dist_records[n][new_key]['below_tol'] = [True]

                    # add buttons
                    bg = self.color_name[self.object_name[new_key]['ind']].lower()
                    b = tk.Button(self.BUTTON_FRAME, text=new_key, command=lambda clr=new_key: self.on_button(clr), bg=bg, fg='white')
                    b.grid(row=len(self.all_buttons) + 2, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
                    self.all_buttons.append(b)
                    self.center_root(r=35)
                    # add table info
                    rd = self.results_dict[new_key]
                    self.tv.insert('', 'end', new_key, text=new_key, values=(self.color_name[self.object_name[new_key]['ind']], rd['path'][-1], rd['n_frame'][-1]))
                    print('added!')
                else:
                    self.drag_flag = 'new'
                    run = False
            else:
                self.msg('目標數量太多咯!')
                run = False
        elif clr == '誤判':
            self.fp_pts.append(p)
            print('deleted!')
                
        if run:
            if len(self.undone_pts) == 0:
                self.root.update()
                self.calculate_path(self.stop_n_frame + 1)
            else:
                self.current_pts, self.current_pts_n = self.undone_pts.pop(0)
                self.suggest_ind.pop(0)
                if len(self.suggest_ind) > 0:
                    print(self.suggest_ind[0])
                    if self.suggest_ind[0][0] == 'fp':
                        self.all_buttons[0].focus_force()
                    elif self.suggest_ind[0][0] == 'new':
                        self.all_buttons[1].focus_force()
                    else:
                        self.all_buttons[self.object_name[self.suggest_ind[0][0]]['ind']].focus_force()
                else:
                    self.all_buttons[0].focus_force()

    # move to previous frame
    def on_left(self, event):
        if self.n_frame > 1:
            self.n_frame -= 1
        else:
            self.msg('Already the first frame!')
    
    # move to next frame
    def on_right(self, event):
        if self.n_frame == self.total_frame:
            self.msg('Already the last frame!')
        else:
            self.n_frame += 1

    # move to previous 5 frames
    def on_page_up(self, event):
        if self.n_frame > 1:
            self.n_frame -= 5
            self.n_frame = max(self.n_frame, 1)
        else:
            self.msg('Already the first frame!')

    # move to next 5 frames
    def on_page_down(self, event):
        if self.n_frame == self.total_frame:
            self.msg('Already the last frame!')
        else:
            self.n_frame += 5
            self.n_frame = min(self.n_frame, self.total_frame)

    def on_up(self, event):
        print('up')

    def on_down(self, event):
        print('down')

    # on some key pressed event
    def on_key(self, event):
        sym = event.keysym
        if not self.is_manual:
            if sym not in ['n', 'Delete', 'd', 'm', 'j']:
                try:
                    i = int(event.char)
                    self.on_button([k for k, v in self.object_name.items() if v['ind'] == i - 1][0])
                except Exception as e:
                    print('on_key', e)

            elif sym == 'n':
                self.on_button('新目標')
            elif sym in ['Delete', 'd']:
                self.on_button('誤判')
            # enter manual label mode
            elif sym == 'm':
                self.chg_mode()
            elif sym == 'j':
                self.jump_frame()
        else:
            if sym == 'n':
                if self.drag_flag is None:
                    if len(self.object_name) < 6:
                        self.drag_flag = 'new'
                    else:
                        self.msg('目標數量太多咯!')
                else:
                    self.drag_flag = None
                self.drag_flag = 'new' if self.drag_flag is None else None
            elif sym == 'm':
                self.leave_manual_label()
            elif sym == 'j':
                self.jump_frame()
            else:
                print('on_key error %s' % type(sym))

    def set_max(self, s):
        v = int(float(s))
        self.maximum_var.set(v)
        self.maximum = v

    def set_tol(self, s):
        v = round(float(s), 1)
        self.tol_var.set(v)
        self.tol = v

    def set_nframe(self, s):
        v = int(float(s))
        self.n_frame_var.set(v)
        self.n_frame = v

    def on_return(self, event=None):
        self.n_frame = self.stop_n_frame

    def leave_manual_label(self, event=None):
        
        if self.is_manual:
            # if exists label record
            if self.tmp_results_dict != self.results_dict:
                string = '是否把以上標註加入目前的目標路徑？'
                result = askyesno('確認', string, icon='warning')
                if result:
                    self.results_dict = self.tmp_results_dict
                    if self.min_label_ind is not None:
                        
                        for k, v in self.results_dict.items():
                            try:
                                flag = min([v['n_frame'].index(f) for f in v['n_frame'] if f >= self.min_label_ind])
                                v['path'] = v['path'][:(flag + 1)]
                                v['n_frame'] = v['n_frame'][:(flag + 1)]
                            except:
                                pass

                        self.calculate_path(self.min_label_ind)
                    else:
                        pass
                else:
                    self.object_name = self.undo_records[-1][-1]

            self.chg_mode()
        # else:
        #     self.n_frame = self.stop_n_frame

    def on_remove(self):
        
        # pending; a better workflow for undo
        def destroy(i):
            self.save_records()
            # root.grab_release()
            k = [k for k, v in self.object_name.items() if v['ind'] == i][0]
            result = askyesno('確認', '確定要刪除 %s 嗎？' %k, icon='warning')
            if result:
                button = self.all_buttons[i+2]
                button.grid_forget()
                self.center_root(r=-35)
                root.destroy()
                top.destroy()

                # delete all info
                self.tv.delete(k)
                self.object_name[k]['on'] = False

                del self.results_dict[k]
                if k in self.dist_records.keys():
                    del self.dist_records[k]
            else:
                pass
        def close():
            root.destroy()

        root = tk.Tk()
        root.protocol('WM_DELETE_WINDOW', close)
        root.withdraw()
        top = tk.Toplevel()
        top.grab_set()
        ## Display the window and wait for it to close
        root.title('Remove object')
        self.center(top)
        for k in sorted([k for k, v in self.object_name.items() if v['on']]):
            b = ttk.Button(top, text=k, command=lambda i = self.object_name[k]['ind']: destroy(i))
            b.pack(expand=True, fill=tk.BOTH)
        self.root.wait_window(top)
        root.mainloop()

    def on_show_boxes(self):
        self.is_show_boxes = not self.is_show_boxes

    def break_loop(self, event=None):
        self.safe = False

    def on_view(self):
        results_dict = self.results_dict
        object_name = self.object_name
        start_pt = self.n_frame if self.n_frame != self.stop_n_frame else 1
        break_pt = max([max(v['n_frame']) for k, v in results_dict.items()])

        for i in range(start_pt, break_pt):
            if (i % 20 == 0 or i == start_pt):
                self.n_frame = i
                self.update_frame()
                frame = self._orig_frame.copy()
                for k in sorted([k for k, v in object_name.items() if v['on']]):
                    flag = results_dict[k]['n_frame']
                    color = self.color[object_name[k]['ind']]
                    try:
                        ind = np.where(np.array(flag) > i)[0][0]
                    except Exception as e:
                        # print(e)
                        ind = None
                    if ind is not None:
                        path = results_dict[k]['path'][:ind][-150:]
                        for l in range(1, len(path)):
                            thickness = int(np.sqrt((1 + l * 0.01) * 2) * 1.5)
                            cv2.line(frame, path[l - 1], path[l], color, thickness)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.display_label.configure(image=image)
                self.scale_nframe.set(i)
                self.root.update_idletasks()

        self.n_frame = self.stop_n_frame

    def on_view_results(self):
        """
        TODO:
        interpolation for results.
        """
        results_dict = self.results_dict
        video = imageio.get_reader(self.video_path)
        COLOR = self.color
        object_name = self.object_name
        start_pt = self.n_frame
        break_pt = max([max(v['n_frame']) for k, v in results_dict.items()])

        # nested function
        def stream(label):

            for i, frame in enumerate(video.iter_data()):
                if i % 20 == 0 and (i+1) >= start_pt:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    for k in sorted([k for k, v in object_name.items() if v['on']]):
                        flag = results_dict[k]['n_frame']
                        color = self.color[object_name[k]['ind']]
                        try:
                            ind = np.where(np.array(flag) > i)[0][0]
                        except Exception as e:
                            # print(e)
                            ind = None
                        if ind is not None:
                            path = results_dict[k]['path'][:ind][-150:]
                            for l in range(1, len(path)):
                                thickness = int(np.sqrt((1 + l * 0.01) * 2) * 1.5)
                                cv2.line(frame, path[l - 1], path[l], color, thickness)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = ImageTk.PhotoImage(Image.fromarray(frame))
                    label.configure(image=image)
                    label.image = image
                if i == break_pt:
                    master.destroy()
                    temp_root.destroy()
                    break
        def exit(event):
            master.destroy()
            temp_root.destroy()

        temp_root = tk.Tk()
        temp_root.withdraw()
        self.center(temp_root)
        master = tk.Toplevel()
        master.focus_force()
        master.title('Results')
        my_label = tk.Label(master)
        my_label.pack()
        thread = threading.Thread(target=stream, args=(my_label,))
        thread.daemon = 1
        thread.start()
        master.bind('<Escape>', exit)
        # temp_root.destroy()
        temp_root.mainloop()

    # for changing name
    def tvitem_click(self, event, item=None):
        self.save_records()

        sel_items = self.tv.selection() if item is None else item
        if sel_items:
            popup = Interface.popupEntry(self.root, title="更改 object 名稱", string="請輸入新的名稱。")
            self.root.wait_window(popup.top)
            sel_item = sel_items[0]

            try:
                new_key = popup.value
                if new_key in [v['display_name'] for k, v in self.object_name.items()]:
                    self.msg('%s 已經被使用了。' % new_key)
                    self.tvitem_click(item=sel_items)
                elif new_key in [" " * i for i in range(10)]:
                    self.msg('請輸入空白鍵以外的字串。')
                    self.tvitem_click(item=sel_items)
                elif len(new_key) > 6:
                    self.msg('請輸入長度小於 6 的字串。')
                    self.tvitem_click(item=sel_items)
                else:
                    self.object_name[sel_item]['display_name'] = new_key
                    self.all_buttons[self.object_name[sel_item]['ind'] + 2].config(text=new_key)
            except:
                pass

    # undo method
    def undo(self, event=None):

        if len(self.undo_records)  == 0:
            self.msg('Nothing can undo.')
        elif self.is_manual:
            self.undo_manual()
        else:
            old_name = self.object_name

            self.results_dict, self.tmp_results_dict, self.stop_n_frame, self.undone_pts, self.current_pts, self.current_pts_n, self.suggest_ind, self.object_name = self.undo_records[-2 if len(self.undo_records) > 1 else -1]

            if old_name != self.object_name:
                keys = set(self.object_name.keys()).difference(set(old_name.keys()))
                for k in keys:
                    self.object_name.pop(k)
                    self.tmp_results_dict.pop(k)
                    self.results_dict.pop(k)
                    self.dist_records[self.n_frame].pop(k)

                    self.tv.delete(k)
                    self.all_buttons[-1].grid_forget()
                    self.all_buttons.pop()

            self.undo_records = self.undo_records[:-1]
            self.n_frame = self.stop_n_frame
            if len(self.suggest_ind) > 0:
                if self.suggest_ind[0][0] == 'fp':
                    self.all_buttons[0].focus_force()
                elif self.suggest_ind[0][0] == 'new':
                    self.all_buttons[1].focus_force()
                else:
                    self.all_buttons[self.object_name[self.suggest_ind[0][0]]['ind']].focus_force()

            # update buttons number
            on_ind = [v['ind'] for k, v in self.object_name.items() if v['on']]
            for i, b in enumerate(self.all_buttons):
                if i in [0, 1]:
                    pass
                elif i - 2 in on_ind:
                    b.grid(row=i + 2, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
                    b.config(text=self.object_name[letter[i-2]]['display_name'])
                else:
                    try:
                        self.tv.delete(letter[i-2])
                    except:
                        print(letter[i-2])

                    b.grid_forget()

    def undo_manual(self):
        self.drag_flag = None
        for k in sorted(self.object_name.keys()):
            # remove current label if any exists
            try:
                ind = self.label_dict[k]['n_frame'].index(self.n_frame)
                self.label_dict[k]['n_frame'].pop(ind)
                self.label_dict[k]['path'].pop(ind)
            except:
                pass

            try:
                ind = self.results_dict[k]['n_frame'].index(self.n_frame)
                self.tmp_results_dict[k]['path'][ind] = self.results_dict[k]['path'][ind]
            except:
                try:
                    ind = self.tmp_results_dict[k]['n_frame'].index(self.n_frame)
                    self.tmp_results_dict[k]['n_frame'].pop(ind)
                    self.tmp_results_dict[k]['path'].pop(ind)

                    if len(self.tmp_results_dict[k]['path']) == 0:
                        self.object_name.pop(k)
                        self.tmp_results_dict.pop(k)
                        self.dist_records[self.n_frame].pop(k)

                        self.tv.delete(k)
                        self.all_buttons[-1].grid_forget()
                        self.all_buttons.pop()
                except Exception as e:
                    # print('undo_manual', e)
                    pass
        self.center_root(r=-35)

    def jump_frame(self):
        popup = Interface.popupEntry(self.root, title="移動幀數", string="請輸入介於 %s ~ %s 的數字。" % (1, self.total_frame), validnum=True)
        self.root.wait_window(popup.top)
        try:
            n = int(popup.value)
            if n >= 1 and n <= self.total_frame:
                self.n_frame = n
            else:
                self.msg("請輸入介於 %s ~ %s 的數字。" % (1, self.total_frame))
                self.jump_frame()
        except Exception as e:
            print(e)
