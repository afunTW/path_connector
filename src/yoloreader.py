import copy
import logging

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.measure import compare_ssim

from src.utils import catchtime

LOGGER = logging.getLogger(__name__)
letter = [str(i) for i in range(1, 20)]

N_SHOW = 25 # thres for update image frame

# threshold
THRES_FORWARD_DIST = 30 # append 到 assigned key 的最小距離
THRES_FORWARD_N_MAX = 50 # 往後看多少 frame 是不是有符合最小距離的條件
THRES_FORWARD_N = 10 # 往後 N_MAX 裡有符合最小距離的最少數量, 如果有就當作新 key
THRES_NEAR_DIST = 20 # 對於沒有被分配 key 的bbox, 距離多進就直接 pass

THRES_NEAR_DIST_NOT_ASSIGN = 70 # 對於沒有被分配 key 的 bbox, 和其他沒被分配到 bbox 的 key 可以符合 append 條件的距離
THRES_NOT_ASSIGN_FORWARD_N_MAX = 100 # 沒有被分配 key 的 bbox, 往之後再看多少 frame 數
THRES_NOT_ASSIGN_FORWARD_DIST = 50 # 沒有被分配 key 的 bbox, 符合停下來讓 user 判斷的最小距離
THRES_NOT_ASSIGN_FORWARD_N = 15 # 往後 N_MAX 裡符合最小距離的最少數量
THRES_NOT_ASSIGN_FP_DIST = 30 # 比較和已經是 false positive 的距離

SEPARATE_N_FRAME = 40
THRES_SEP_DIST = 50

class YOLOReader(object):

    def read_yolo_result(self):
        # yolo_results_path = 'testing_0928.txt'
        yolo_results_path = self.video_path.split('.avi')[0] + '.txt'
        with open(yolo_results_path, 'r') as f:
            lines = f.readlines()
        return lines

    # store new results dict
    def add_res(self, ind, c, wh, nframe):
        if ind in self.results_dict:
            self.results_dict[ind]['n_frame'].append(nframe)
            self.results_dict[ind]['path'].append(c)
            self.results_dict[ind]['wh'].append(wh)
        else:
            self.results_dict[ind] = {
                'path': [c],
                'n_frame': [nframe],
                'wh': [wh]
            }


    def calculate_path(self, ind=None, n_show=N_SHOW):
        """
        A algorithm that connects YOLO bounding info as continuous bounding box
        -----------------------------------------------------------------------
        ind: run this function start from ind-th frames
        n_show: records every n_show for displaying tracked results

        self.__yolo_results__ = [frame index, [[y1, x1, y2, x2, prob1, {class: prob}], [y1, ...], ...]]
        """
        # n_key_used: 有出現過的 label 個數
        # object name = class label
        n_key_used = len(self.object_name.keys())

        self.is_calculate = True
        self.suggest_ind = []
        undone_pts = []
        n_frame = ind or self.n_frame

        while self.is_calculate:
            
            # 取得當前 frame 的紀錄 (frame index, bbox+label probability)
            if n_frame < self.__total_n_frame__:
                nframe, boxes = eval(self.__yolo_results__[n_frame - 1])
                # classes = np.random.choice(np.arange(1, 6),size=len(boxes), 
                #     p=[0.25, 0.25, 0.25, 0.23, 0.02], replace=False if len(boxes) < 5 else True)
                # 假設有 5 種埋葬蟲的 index, 4 種 mark + 1 unknown
            else:
                self.is_calculate = False
                self.is_finish = True
                break
            assert nframe == n_frame

            # 過濾 detection model 判斷為蟲的機率 < 0.75 的 bbox
            boxes = [b for b in boxes if b[4] > 0.75]
            boxes = np.array(boxes)

            # append history manual label result
            label_ind = [k for k, v in self.label_dict.items() if n_frame in v['n_frame']]
            for k in label_ind:
                if n_frame in self.label_dict:
                    i = self.label_dict[k]['n_frame'].index(n_frame)
                    LOGGER.info(self.results_dict[k])
                    self.add_res(k, self.label_dict[k]['path'][i], self.results_dict[k]['wh'][-1], n_frame)

            # 每 n_show 個 frame 更新顯示畫面
            if n_frame % n_show == 0:
                self.update_frame(n_frame)

            # 如果 detection results 有蟲的話
            if len(boxes) > 0:
                self.dist_records[n_frame] = dict()

                # on_keys: 有出現過而且有偵測到的 label
                on_keys = sorted([k for k, v in self.object_name.items() if v['on']])

                # boxes: 現在這個 frame 被 model 判斷是蟲, 而且機率大於 75% 的 bounding box
                for i, box in enumerate(boxes):
                    # classes
                    # chrac = str(classes[i])
                    ymin, xmin, ymax, xmax, score, label_prob = box
                    chrac, chrac_prob = max(label_prob.items(), key=lambda x: x[-1])
                    LOGGER.info('{} - bbox ({}, {}, {}, {}) label ({}, {})'.format(
                        nframe, ymin, xmin, ymax, xmax, chrac, chrac_prob
                    ))
                    x_c = int((xmin+xmax) / 2 + 0.5)
                    y_c = int((ymin+ymax) / 2 + 0.5)
                    p = (x_c, y_c)  # bbox center
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)

                    # 沒有出現過任何 label 或是第一個 frame, init
                    if (n_key_used == 0 or n_frame == 1) and not self.is_manual:
                        # temp: 符合最小距離的數量
                        # fp_n: 看到未來第 n 個 frame 的 index (為了檢查是否有符合最小距離條件的 frame)
                        # forward_points: 未來 n 個 frame 的 model 預測 bounding box 的結果
                        temp = 0
                        fp_n = min(n_frame + THRES_FORWARD_N_MAX, len(self.__yolo_results__))
                        forward_points = [eval(self.__yolo_results__[i])[1] for i in range(n_frame, fp_n)]
                        p_tmp = p
                        for i, res in enumerate(forward_points):
                            min_dist = 99999
                            for b in res:
                                # dist: default l2-norm (sqrt(x**2, y**2))
                                ymin, xmin, ymax, xmax, score, label_prob = b
                                x_c = int((xmin+xmax) / 2 + 0.5)
                                y_c = int((ymin+ymax) / 2 + 0.5)
                                p_forward = (x_c, y_c)
                                dist = np.linalg.norm(np.array(p_forward) - np.array(p_tmp))
                                if dist <= min(THRES_FORWARD_DIST, min_dist):
                                    min_dist = dist
                                    p_tmp = p_forward
                            if min_dist < THRES_FORWARD_DIST:
                                temp += 1

                        # 假如在未來 THRES_FORWARD_N_MAX (最多 50) 個 frame 以內
                        # 與 box 的最小距離在 THRES_FORWARD_DIST (30) 以內的次數超過 THRES_FORWARD_N (10) 次
                        # 自動分配 label name 並紀錄
                        if temp > THRES_FORWARD_N:
                            # append first point to results
                            # chrac = letter[n_key_used]
                            self.add_res(chrac, p, (w, h), n_frame)

                            self.object_name[chrac] = {
                                'ind': n_key_used,
                                'on': True,
                                'display_name': chrac
                            }
                            n_key_used += 1

                            # record distance history
                            self.dist_records[n_frame][chrac] = {
                                'dist': [0],
                                'center': [p],
                                'below_tol': [True],
                                'wh': [(w, h)]
                            }

                    # 有出現過 label 而且不是第一個 frame
                    # 對這個 frame 有被 detect 到的 label
                    # 紀錄最近一次被 detect 到的座標到這個 bbox 的距離 (self.dist_records)
                    else:
                        # record all distance history first
                        for i, k in enumerate(on_keys):
                            v = self.results_dict[k]['path']
                            dist = np.linalg.norm(np.array(v[-1]) - np.array(p))

                            if k not in self.dist_records[n_frame].keys():
                                self.dist_records[n_frame][k] = dict()
                                self.dist_records[n_frame][k]['dist'] = [dist]
                                self.dist_records[n_frame][k]['center'] = [p]
                                self.dist_records[n_frame][k]['below_tol'] = [dist <= self.tol]
                                self.dist_records[n_frame][k]['wh'] = [(w, h)]
                            else:
                                self.dist_records[n_frame][k]['dist'].append(dist)
                                self.dist_records[n_frame][k]['center'].append(p)
                                self.dist_records[n_frame][k]['below_tol'].append(dist <= self.tol)
                                self.dist_records[n_frame][k]['wh'].append((w, h))

                tmp_dist_record = copy.deepcopy(self.dist_records[n_frame])
                # start judgement
                # sorted dist index by dist
                # sorted_indexes: 在 n_frame 時, 每個 label 到每個 bbox 距離的排序
                # e.g. {'1': [0, 2, 1], '2': [2, 0, 1], '3': [1, 2, 0], '4': [2, 1, 0]}
                sorted_indexes = {k: sorted(range(len(v['dist'])), key=lambda k: v['dist'][k]) for k, v in tmp_dist_record.items()}
                hit_condi = [(k, sorted_indexes[k][0]) for k in on_keys if tmp_dist_record[k]['below_tol'][sorted_indexes[k][0]]]
                ######

                # 把 classifier 分類好
                # 目前就是按照 classification 的結果分配 box 對應哪一隻蟲
                # 需要注意的 TODO:
                # - 同一個 class 在同一個 frame 有多個 boxes
                # - 有太多 unknown, 需要利用未來的資訊來做篩選

                for i, box in enumerate(boxes):
                    # classes
                    # ind = str(classes[i])
                    ymin, xmin, ymax, xmax, score, label_class = box
                    chrac, chrac_prob = max(label_prob.items(), key=lambda x: x[-1])
                    x_c = int((xmin+xmax) / 2 + 0.5)
                    y_c = int((ymin+ymax) / 2 + 0.5)
                    p = (x_c, y_c)  # bbox center
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)

                    # if classes is not unknown, 把 bbox 分配給對應的蟲
                    if chrac != '5':
                        self.add_res(chrac, p, (w, h), n_frame)

                    # 如果撞到 unknown classes 就停下來
                    else:
                        LOGGER.info('Ready to stop because of charc == "{}"'.format(chrac))
                        self.is_calculate = False
                        undone_pts.append((tmp_dist_record[on_keys[0]]['center'][i], n_frame))

                assigned_keys = [k for k, chrac in hit_condi]
                assigned_boxes = [chrac for k, chrac in hit_condi]
                not_assigned_boxes = set(range(len(boxes))).difference(assigned_boxes)
                not_assigned_indices = []
                ######

            if self.is_calculate:
                n_frame += 1
            else:
                LOGGER.info('paths connecting stops at %s' % n_frame)

            # record animation
            if n_frame % n_show == 0 and len(self.all_buttons) != 0:

                cv2.putText(self._frame, 'Calculating...', (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
                for k in on_keys:
                    color = self.color[self.object_name[k]['ind']]
                    flag = self.results_dict[k]['n_frame']
                    try:
                        last = np.where(np.array(flag) > (n_frame - 300))[0][0]
                    except:
                        last = None
                    # draw only for last 300 frame
                    if last is not None:
                        pts = self.results_dict[k]['path'][last:]
                        for i in range(1, len(pts)):
                            thickness = int(np.sqrt((1 + i * 0.01)  * 2) * 1.5)
                            cv2.line(self._frame, pts[i - 1], pts[i], color, thickness)

                if self.root.state() == 'zoomed':
                    shape = self._frame.shape

                    r1 = (shape[1] / self.root.winfo_width())
                    r2 = (shape[0] / self.root.winfo_height())
                    shrink_r = max(r1, r2)

                    _c_height = self._r_height/shrink_r
                    _c_width = self._r_width/shrink_r

                    if r1 == shrink_r:
                        nw = int(shape[1] * _c_width)
                        nh = int(shape[0] * nw / shape[1])
                    else:
                        nh = int(shape[0] * _c_height)
                        nw = int(shape[1] * nh / shape[0])

                    df_w = self.frame_display.winfo_width()
                    if df_w == 1284:
                        pass
                    elif nw > df_w:
                        nn_w = df_w - 4
                        r = nn_w / nw
                        nn_h = int(nh * r)
                        nh = nn_h
                        nw = nn_w
                    else:
                        LOGGER.info(df_w)

                    newsize = (nw, nh)
                    self._frame = cv2.resize(self._frame, newsize)

                self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
                self.image = ImageTk.PhotoImage(Image.fromarray(self._frame))
                if self.label_display is not None:
                    self.label_display.configure(image=self.image)
                    self.scale_nframe_v.set(n_frame)
                    self.root.update_idletasks()
                else:
                    self.tracked_frames.append(ImageTk.PhotoImage(Image.fromarray(self._frame)))
                # break from calculating
                break

        if self.is_calculate:
            self.cancel_id = self.label_display.after(0, self.calculate_path, n_frame)
        else:
            # if the labeling process is not finish
            if not self.is_finish and undone_pts:
                undone_pts = list(set(undone_pts))
                self.hit_condi = hit_condi
                self.suggest_options(undone_pts, nframe)

                # update new value
                self.n_frame = n_frame
                self.stop_n_frame = n_frame
                self.current_pts, self.current_pts_n = undone_pts.pop(0)
                self.undone_pts = undone_pts

                # record value for undoing
                with catchtime("saving record took time", "info") as f:
                    self.save_records()

                # ensure don't enter manual mode and reset relevant variables
                self.min_label_ind = None
                self.cancel_id = None
            else:
                self.n_frame = n_frame
                self.current_pts = None
                self.msg("你已完成本影片的所有軌跡標註, 辛苦了!")
                self.export()

    # logic for suggesting a reasonable option
    def suggest_options(self, undone_pts, nframe):
        on_keys = [k for k, v in self.object_name.items() if v['on']]
        hit_condi = self.hit_condi
        LOGGER.info('Active label - {}'.format(on_keys))
        LOGGER.info('Suggest options (label name, bbox index) - {}'.format(hit_condi))
        for i, tup in enumerate(undone_pts):
            p, nframe = tup
            tmp_record = self.dist_records[nframe]
            min_dist_not_assigned = 9999
            min_key_not_assigned = None
            # compare with not assigned key
            for k, v in tmp_record.items():
                if k in [tmp for tmp in on_keys if tmp not in [j for j, _ in hit_condi]]:
                    try:
                        ind = tmp_record[k]['center'].index(p)
                    # should only occur while there are multi undone points, calculate dist record
                    except Exception as e:
                        dist = np.linalg.norm(np.array(self.results_dict[k]['path'][-1]) - np.array())
                        _, boxes = eval(self.__yolo_results__[self.n_frame - 1])
                        for b in boxes:
                            ymin, xmin, ymax, xmax, score = b
                            x_c = int((xmin+xmax) / 2 + 0.5)
                            y_c = int((ymin+ymax) / 2 + 0.5)
                            w = int(xmax - xmin)
                            h = int(ymax - ymin)

                            if p == (x_c, y_c):
                                break

                        tmp_record[k]['center'].append(p)
                        tmp_record[k]['dist'].append(dist)
                        tmp_record[k]['below_tol'].append(True if dist <= self.tol else False)
                        tmp_record[k]['wh'].append((w, h))
                        ind = tmp_record[k]['center'].index(p)

                    if tmp_record[k]['dist'][ind] < min_dist_not_assigned:
                        min_dist_not_assigned = tmp_record[k]['dist'][ind]
                        min_key_not_assigned = k
            # compare with assigned key
            min_dist_assigned = 9999
            min_key_assigned = None
            for k, v in tmp_record.items():
                if k in [tmp for tmp in on_keys if tmp in [j for j, _ in hit_condi]]:
                    try:
                        ind = tmp_record[k]['center'].index(p)
                    except:
                        dist = np.linalg.norm(np.array(self.results_dict[k]['path'][-1]) - np.array(p))
                        _, boxes = eval(self.__yolo_results__[self.n_frame - 1])
                        for b in boxes:
                            ymin, xmin, ymax, xmax, score = b
                            x_c = int((xmin+xmax) / 2 + 0.5)
                            y_c = int((ymin+ymax) / 2 + 0.5)
                            w = int(xmax - xmin)
                            h = int(ymax - ymin)

                            if p == (x_c, y_c):
                                break

                        tmp_record[k]['center'].append(p)
                        tmp_record[k]['dist'].append(dist)
                        tmp_record[k]['below_tol'].append(True if dist <= self.tol else False)
                        tmp_record[k]['wh'].append((w, h))
                        ind = tmp_record[k]['center'].index(p)

                    if tmp_record[k]['dist'][ind] < min_dist_assigned:
                        min_dist_assigned = tmp_record[k]['dist'][ind]
                        min_key_assigned = k

            # suggest new object if far with both assigned and not assigned keys
            if min_dist_not_assigned >= 160 and min_dist_assigned > 100:
                self.suggest_ind.append(('new', {'assigned': (min_key_assigned, min_dist_assigned), 'not_assigned': (min_key_not_assigned, min_dist_not_assigned)}))
            # suggest false positive if far with not assigned keys but near with assigned keys
            elif min_dist_not_assigned >= 160 and min_dist_assigned < 100:
                self.suggest_ind.append(('fp', {'assigned': (min_key_assigned, min_dist_assigned), 'not_assigned': (min_key_not_assigned, min_dist_not_assigned)}))
            # suggest the nearest not assigned key for other cases
            else:
                self.suggest_ind.append((min_key_not_assigned, {'assigned': (min_key_assigned, min_dist_assigned), 'not_assigned': (min_key_not_assigned, min_dist_not_assigned)}))

        # update default option if button has been already created
        if len(self.all_buttons) > 0:
            if self.suggest_ind[0][0] == 'fp':
                ind = 0
            elif self.suggest_ind[0][0] == 'new':
                ind = 1
            else:
                ind = self.object_name[self.suggest_ind[0][0]]['ind'] + 2
            self.all_buttons[ind].focus_force()
            self.label_suggest.grid(row=ind, column=1, sticky="nwes", padx=5, pady=5)
