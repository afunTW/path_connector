import numpy as np
import cv2
import copy
from PIL import Image, ImageTk

# from mahotas.features import haralick, zernike
# from skimage.feature import hog, local_binary_pattern
from skimage.measure import compare_ssim

# letter = [chr(i) for i in range(ord('A'), ord('Z')+1)]
letter = [str(i) for i in range(1, 20)]

N_SHOW = 25

# threshold
THRES_FORWARD_DIST = 30
THRES_FORWARD_N_MAX = 30
THRES_FORWARD_N = 10
THRES_NEAR_DIST = 80 # for pass false positive

THRES_NEAR_DIST_NOT_ASSIGN = 48 # for append if very near not assigned key
THRES_NOT_ASSIGN_FORWARD_N_MAX = 100
THRES_NOT_ASSIGN_FORWARD_DIST = 35
THRES_NOT_ASSIGN_FORWARD_N = 10
THRES_NOT_ASSIGN_FP_DIST = 20

class YOLOReader(object):

    def read_yolo_result(self):
        # yolo_results_path = 'testing_0928.txt'
        yolo_results_path = self.video_path.split('.avi')[0] + '.txt'
        with open(yolo_results_path, 'r') as f:
            lines = f.readlines()
        return lines

    def calculate_path(self, ind=None, n_show=N_SHOW):
        """
        A algorithm that connects YOLO bounding info as continuous bounding box
        -----------------------------------------------------------------------
        ind: run this function start from ind-th frames
        n_show: records every n_show for displaying tracked results
        """
        n_key_used = len(self.object_name.keys())
        
        self.is_calculate = True
        n_frame = ind if ind is not None else self.n_frame
        undone_pts = []
        self.suggest_ind = []

        while self.is_calculate:
            # self.root.update_idletasks()
            nframe, boxes = eval(self.__yolo_results__[n_frame - 1])
            assert nframe == n_frame
            boxes = np.array(boxes)

            # append history manual label result
            label_ind = [k for k, v in self.label_dict.items() if n_frame in v['n_frame']]
            for k in label_ind:
                i = self.label_dict[k]['n_frame'].index(n_frame)
                print(self.results_dict[k])
                self.results_dict[k]['n_frame'].append(n_frame)
                self.results_dict[k]['path'].append(self.label_dict[k]['path'][i])
                self.results_dict[k]['wh'].append(self.label_dict[k]['wh'][i])

            # initiate frame for recording animation
            if n_frame % n_show == 0:
                self.update_frame(n_frame)
            if len(boxes) > 0:
                self.dist_records[n_frame] = dict()

                on_keys = [k for k, v in self.object_name.items() if v['on']]

                for i, box in enumerate(boxes):
                    ymin, xmin, ymax, xmax, score = box
                    x_c = int((xmin+xmax) / 2 + 0.5)
                    y_c = int((ymin+ymax) / 2 + 0.5)
                    p = (x_c, y_c)
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)
                    # if there is no keys, initiate
                    if (n_key_used == 0 or n_frame == 1) and not self.is_manual:
                        temp = 0
                        forward_points = [eval(self.__yolo_results__[i])[1] for i in range(n_frame, n_frame + THRES_FORWARD_N_MAX)]
                        p_tmp = p
                        for i, res in enumerate(forward_points):
                            min_dist = 99999
                            for b in res:
                                ymin, xmin, ymax, xmax, score = b
                                x_c = int((xmin+xmax) / 2 + 0.5)
                                y_c = int((ymin+ymax) / 2 + 0.5)
                                p_forward = (x_c, y_c)
                                dist = np.linalg.norm(np.array(p_forward) - np.array(p_tmp))
                                if dist <= min(THRES_FORWARD_DIST, min_dist):
                                    min_dist = dist
                                    p_tmp = p_forward
                            if min_dist < THRES_FORWARD_DIST:
                                temp += 1

                        # if was connected 10 frames in next 30 frames
                        if temp > THRES_FORWARD_N:
                            # append first point to results
                            chrac = letter[n_key_used]
                            self.results_dict[chrac] = dict()
                            self.results_dict[chrac]['path'] = [p]
                            self.results_dict[chrac]['n_frame'] = [n_frame]
                            self.results_dict[chrac]['wh'] = [(w, h)]

                            self.object_name[chrac] = {'ind': n_key_used, 'on': True, 'display_name': chrac}
                            n_key_used += 1

                            # record distance history
                            self.dist_records[n_frame][chrac] = dict()
                            self.dist_records[n_frame][chrac]['dist'] = [0]
                            self.dist_records[n_frame][chrac]['center'] = [p]
                            self.dist_records[n_frame][chrac]['below_tol'] = [True]
                            self.dist_records[n_frame][chrac]['wh'] = [(w, h)]

                    else:
                        # record all distance history first
                        for i, k in enumerate(on_keys):
                            v = self.results_dict[k]['path']
                            dist = np.linalg.norm(np.array(v[-1]) - np.array(p))

                            if k not in self.dist_records[n_frame].keys():
                                self.dist_records[n_frame][k] = dict()
                                self.dist_records[n_frame][k]['dist'] = [dist]
                                self.dist_records[n_frame][k]['center'] = [p]
                                self.dist_records[n_frame][k]['below_tol'] = [True if dist <= self.tol else False]
                                self.dist_records[n_frame][k]['wh'] = [(w, h)]
                            else:
                                self.dist_records[n_frame][k]['dist'].append(dist)
                                self.dist_records[n_frame][k]['center'].append(p)
                                self.dist_records[n_frame][k]['below_tol'].append(True if dist <= self.tol else False)
                                self.dist_records[n_frame][k]['wh'].append((w, h))

                # start judgement
                tmp_dist_record = copy.deepcopy(self.dist_records[n_frame])
                # sorted dist index by dist
                sorted_indexes = {k: sorted(range(len(v['dist'])), key=lambda k: v['dist'][k]) for k, v in tmp_dist_record.items()}
                hit_condi = [(k, sorted_indexes[k][0]) for k in on_keys if tmp_dist_record[k]['below_tol'][sorted_indexes[k][0]]]
                
                # if n_frame > 450 and n_frame < 455:
                #     print(n_frame)
                #     print(tmp_dist_record)

                # the easiest part: the length of hit_condi is same as the number of objects
                if n_frame == 1:
                    pass
                elif len(set([v[1] for v in hit_condi])) == len(on_keys):

                    # pending assessment
                    for k, ind in hit_condi:
                        if k not in label_ind:
                            self.results_dict[k]['path'].append(tmp_dist_record[k]['center'][ind])
                            self.results_dict[k]['n_frame'].append(n_frame)
                            self.results_dict[k]['wh'].append(tmp_dist_record[k]['wh'][ind])

                # the length of hit_condi is same as the number of nearest indexes
                elif len(set([v for k, v in hit_condi])) == len(hit_condi):

                    # pending, assessment
                    for k, ind in hit_condi:
                        if k not in label_ind:
                            self.results_dict[k]['path'].append(tmp_dist_record[k]['center'][ind])
                            self.results_dict[k]['n_frame'].append(n_frame)
                            self.results_dict[k]['wh'].append(tmp_dist_record[k]['wh'][ind])

                    # if there are boxes were not assigned
                    if len(hit_condi) != len(boxes):
                        assigned_boxes = [ind for k, ind in hit_condi]
                        assigned_keys = [k for k, ind in hit_condi]
                        not_assigned_boxes = set([i for i in range(len(boxes))]).difference(assigned_boxes)
                        not_assigned_keys = [k for k in on_keys if k not in assigned_keys]                        

                        for ind in not_assigned_boxes:

                            # if the not assigned boxes are too near with assigned keys, got thres
                            if any([v['dist'][ind] <= THRES_NEAR_DIST for k, v in tmp_dist_record.items() if k in assigned_keys]):
                                pass
                            else:
                                # less strict distance condition for not assigned object
                                min_dist = 9999
                                min_key = None
                                for k in not_assigned_keys:
                                    if tmp_dist_record[k]['dist'][ind] < min_dist:
                                        min_dist = tmp_dist_record[k]['dist'][ind]
                                        if min_dist <= THRES_NEAR_DIST_NOT_ASSIGN:
                                            min_key = k
                                # append the record if any object was found
                                if min_key is not None:
                                    if min_key not in label_ind:
                                        self.results_dict[min_key]['path'].append(tmp_dist_record[min_key]['center'][ind])
                                        self.results_dict[min_key]['n_frame'].append(n_frame)
                                        self.results_dict[min_key]['wh'].append(tmp_dist_record[min_key]['wh'][ind])
                                        not_assigned_keys.pop(-1)
                                else:
                                    # forward next 100 points
                                    temp = 0
                                    forward_points = [eval(self.__yolo_results__[i])[1] for i in range(n_frame - 1, n_frame + (THRES_NOT_ASSIGN_FORWARD_N_MAX - 1))]
                                    p = tmp_dist_record[on_keys[0]]['center'][ind]
                                    for i, res in enumerate(forward_points):
                                        min_dist = 99999
                                        for b in res:
                                            ymin, xmin, ymax, xmax, score = b
                                            x_c = int((xmin+xmax) / 2 + 0.5)
                                            y_c = int((ymin+ymax) / 2 + 0.5)
                                            p_forward = (x_c, y_c)
                                            dist = np.linalg.norm(np.array(p_forward) - np.array(p))
                                            if dist <= min(THRES_NOT_ASSIGN_FORWARD_DIST, min_dist):
                                                min_dist = dist
                                                p = p_forward
                                        if min_dist < THRES_NOT_ASSIGN_FORWARD_DIST:
                                            temp += 1

                                    # if this center is potential, compare it with false positive point.
                                    if temp > THRES_NOT_ASSIGN_FORWARD_N:
                                        compare = False
                                        for fp in self.fp_pts:
                                            fp_dist = np.linalg.norm(np.array(fp) - np.array(tmp_dist_record[on_keys[0]]['center'][ind]))
                                            if fp_dist < THRES_NOT_ASSIGN_FP_DIST:
                                                compare = True
                                        # stop the function and ask user only if this center is not near with false positive points
                                        if not compare:
                                            undone_pts.append((tmp_dist_record[on_keys[0]]['center'][ind], n_frame))

                                            print('not assigned boxes')
                                            print('distance records %s' % tmp_dist_record)
                                            print('object and bbox match pairs %s' % hit_condi)
                                            print('index of not assigned bounding boxes: %s' % ind)
                                            lost_box_key = [k for k in on_keys if k not in [j for j, _ in hit_condi]]
                                            print('lost boxes key %s' % lost_box_key)
                                            if (tmp_dist_record[lost_box_key[0]]['center'][ind], n_frame) not in undone_pts:
                                                undone_pts.append((tmp_dist_record[lost_box_key[0]]['center'][ind], n_frame))
                                            self.is_calculate = False
                                    # if this center of bounding box is not potentially connected in next 100 frames, just ignored it.
                                    else:
                                        pass
                # there are boxes that hit condition with multi objects
                else:
                    hit_boxes = [v for k, v in hit_condi]
                    non_duplicate_key = [k for k, v in hit_condi if hit_boxes.count(v) == 1]
                    duplicate_ind = set([x for x in hit_boxes if hit_boxes.count(x) > 1]) # boxes indexes with multi objects
                    duplicate_key = [k for k, v in hit_condi if hit_boxes.count(v) > 1] # boxes indexes with multi objects

                    # if this is duplicate indexes case, assign the nearest one
                    if len(duplicate_ind) > 0 :
                        """
                        Try compare difference.
                        """
                        min_key = []

                        # just use nearest distance
                        for ind in duplicate_ind:
                            duplicate_key = [k for k, v in hit_condi if v == ind]
                            sorted_keys_by_dist = sorted(range(len(duplicate_key)), key=lambda k: tmp_dist_record[duplicate_key[k]]['dist'][ind])
                            min_dist_key = sorted_keys_by_dist[0]
                            min_key.append(duplicate_key[min_dist_key])

                        hit_condi_reduced = [(k, v) for k, v in hit_condi if k in non_duplicate_key + min_key]
                        for k, ind in hit_condi_reduced:
                            if k not in label_ind:
                                self.results_dict[k]['path'].append(tmp_dist_record[k]['center'][ind])
                                self.results_dict[k]['n_frame'].append(n_frame)
                                self.results_dict[k]['wh'].append(tmp_dist_record[k]['wh'][ind])

                        # pending, not assigned key

                    # if there is any condition that wasn't considered
                    else:
                        self.is_calculate = False
                        print("A not considered case happened!")
                        print(tmp_dist_record)
                        print(hit_condi)
            # just ignored if there is no bounding box in this frame
            else:
                on_keys = [k for k, v in self.object_name.items() if v['on']]
                pass

            if self.is_calculate:
                n_frame += 1
            else:
                print('paths connecting stops at %s' % n_frame)

            # record animation
            if n_frame % n_show == 0 and self.display_label is not None:

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

                    df_w = self.display_frame.winfo_width()
                    if df_w == 1284:
                        pass
                    elif nw > df_w:
                        nn_w = df_w - 4
                        r = nn_w / nw
                        nn_h = int(nh * r)
                        nh = nn_h
                        nw = nn_w
                    else:
                        print(df_w)

                    newsize = (nw, nh)
                    self._frame = cv2.resize(self._frame, newsize)

                self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
                self.image = ImageTk.PhotoImage(Image.fromarray(self._frame))
                if self.display_label is not None:
                    self.display_label.configure(image=self.image)
                    self.scale_nframe.set(n_frame)
                    self.root.update_idletasks()
                else:
                    self.tracked_frames.append(ImageTk.PhotoImage(Image.fromarray(self._frame)))
                # break from calculating
                break

        if self.is_calculate:
            self.cancel_id = self.display_label.after(0, self.calculate_path, n_frame)
        else:
            self.hit_condi = hit_condi
            self.suggest_options(undone_pts, nframe)

            # update new value
            self.n_frame = n_frame
            self.stop_n_frame = n_frame
            self.current_pts, self.current_pts_n = undone_pts.pop(0)
            self.undone_pts = undone_pts
            
            # record value for undoing
            self.save_records()

            # ensure don't enter manual mode and reset relevent variables
            self.min_label_ind = None
            self.cancel_id = None

    # algorithm for suggesting a reasonable option
    def suggest_options(self, undone_pts, nframe):
        on_keys = [k for k, v in self.object_name.items() if v['on']]
        print("on keys: ", on_keys)
        hit_condi = self.hit_condi
        print("suggest_options", hit_condi)
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
            self.suggest_label.grid(row=ind, column=1, sticky="nwes", padx=5, pady=5)
