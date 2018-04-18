import logging
import time
from functools import wraps

import cv2
import numpy as np
from PIL import Image, ImageTk

LENGTH_ARROW = 20
LOGGER = logging.getLogger(__name__)

class catchtime(object):
    def __init__(self, name, log_type="debug"):
        self.name = name
        self.lt = log_type
    def __enter__(self):
        self.t = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.clock() - self.t
        if self.lt == "info":
            LOGGER.info("{} - {:.4f}".format(self.name, self.t))
        elif self.lt == "debug":
            LOGGER.debug("{} - {:.4f}".format(self.name, self.t))

class Utils(object):

    def draw(self, tup=None):
        if self.root:
            results_dict = self.results_dict if not self.is_manual else self.tmp_results_dict
            with catchtime("0") as f:
                # draw connected paths
                for i, k in enumerate(sorted([k for k, v in self.object_name.items() if v['on']])):
                    pts = np.array(results_dict[k]['path'])
                    flag = results_dict[k]['n_frame']
                    color = self.color[self.object_name[k]['ind']]
                    try:
                        # ind = [flag.index(v) for v in flag if v <= self.n_frame]
                        ind = max(np.where(np.array(flag) <= self.n_frame)[0])
                    except Exception as e:
                        ind = None

                    # if the path is not conneted in the current frame yet, show only the first coordinate
                    if ind is not None:
                        if self.var_show_tracked is None or self.var_show_tracked.get() == 1 and not self.is_calculate:
                            # show until current if ind is not None
                            lb = (ind+1-self.maximum)
                            ub = (ind + 1) if ind is not None else None
                            pts = pts[(lb if lb > 0 else 0):ub]
                            if len(pts) > 0:
                                # start point
                                cv2.circle(self._frame, tuple(pts[0]), 10, color, 1)
                                cv2.circle(self._frame, tuple(pts[0]), 13, color, 1)

                                for i in range(1, len(pts)):
                                    p1 = pts[i-1]
                                    p2 = pts[i]
                                    dist = np.linalg.norm(p1 - p2)
                                    # draw dotted line
                                    if dist < 48:
                                        cv2.line(self._frame, tuple(p1), tuple(p2), color, 1)
                                    else:
                                        drawline(self._frame, tuple(p1), tuple(p2), color, 1, style='dotted', gap=7)
                                    # draw arrow
                                    if i % 6 == 0:
                                        if dist > 3:
                                            draw_arrow(self._frame, tuple(p1), tuple(p2), color, dist=dist, thickness=2, line_type=16)
                            else:
                                pass
            with catchtime("1") as f:
                # draw names after paths
                for i, k in enumerate(sorted([k for k, v in self.object_name.items() if v['on']])):
                    pts = np.array(results_dict[k]['path'])
                    flag = results_dict[k]['n_frame']
                    color = self.color[self.object_name[k]['ind']]

                    try:
                        # ind = max([flag.index(v) for v in flag if v <= self.n_frame])
                        ind = max(np.where(np.array(flag) <= self.n_frame)[0])
                    except Exception as e:
                        ind = None
                    if ind is None:
                        cv2.circle(self._frame, tuple(pts[0]), 10, color, 1)
                        cv2.circle(self._frame, tuple(pts[0]), 13, color, 1)
                    else:
                        if flag[ind]  == self.n_frame:
                            width = 4
                        else:
                            width = 1
                        if ind != 0:
                            last_pt = tuple(pts[ind - 1])
                        else:
                            last_pt = tuple(pts[ind])
                        pt = tuple(pts[ind])
                        tri_pts = tri(pt)
                        # draw path end point triangle
                        cv2.polylines(self._frame, tri_pts, True, color, width)

                        # position of text info
                        c = color # (50, 50, 255)
                        if last_pt[1] > pt[1] and pt[1] > 50:
                            if width == 4:
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] - 30, pt[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 4)
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] - 30, pt[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1)
                            else:
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] - 30, pt[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 4)
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] - 30, pt[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
                                cv2.putText(self._frame, '?', (pt[0] - 18, pt[1] - 38), cv2.FONT_HERSHEY_TRIPLEX, 0.6, c, 1)

                        else:
                            if width == 4:
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] + 20, pt[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 4)
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] + 20, pt[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1)
                            else:
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] + 20, pt[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 4)
                                cv2.putText(self._frame, self.object_name[k]['display_name'], (pt[0] + 20, pt[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
                                cv2.putText(self._frame, '?', (pt[0] + 32, pt[1] + 12), cv2.FONT_HERSHEY_TRIPLEX, 0.6, c, 1)

            # draw coordinate (stop point) that needed to be assigned
            if self.current_pts is not None:
                p, nframe = self.current_pts, self.current_pts_n
                color = (50, 50, 255)
                x, y = p
                thickness = 2 if nframe == self.n_frame else 1
                cv2.circle(self._frame, p, 15, color, thickness)
                cv2.putText(self._frame, '?', (x - 9, y + 9), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, thickness)

            # draw YOLO bounding boxes
            if self.var_show_bbox is None or self.var_show_bbox.get() == 1:
                _, boxes = eval(self.__yolo_results__[self.n_frame - 1])

                for b in boxes:
                    ymin, xmin, ymax, xmax, score = b
                    x_c = int((xmin+xmax) / 2 + 0.5)
                    y_c = int((ymin+ymax) / 2 + 0.5)
                    p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
                    color = None

                    # for not showing history false positive points
                    compare = False
                    for fp in self.fp_pts:
                        fp_dist = np.linalg.norm(np.array(fp) - np.array((x_c, y_c)))
                        if fp_dist < 1:
                            compare = True
                            break

                    if not compare:
                        # find corresponding color of YOLO bounding boxes
                        for k in sorted([k for k, v in self.object_name.items() if v['on']]):
                            pts = results_dict[k]['path']
                            flag = results_dict[k]['n_frame']
                            ind = None
                            try:
                                ind = flag.index(self.n_frame)
                            except:
                                pass

                            if ind is not None:
                                if pts[ind] == (x_c, y_c):
                                    color = self.color[self.object_name[k]['ind']]
                                    break

                        if color:
                            cv2.rectangle(self._frame, p1, p2, color, 1)
                        else:
                            cv2.rectangle(self._frame, p1, p2, (255, 255, 255), 1)
            # remove drawing on specific region
            n = 60
            e = 100
            if self.clear and self.var_clear.get() == 1:
                x, y = self.mv_x, self.mv_y
                xmin, xmax = max(0, (x-n)), min((x+n), self.width)
                ymin, ymax = max(0, (y-n)), min((y+n), self.height)
                self._frame[ymin:ymax, xmin:xmax] = self._orig_frame[ymin:ymax, xmin:xmax].copy()
                drawrect(self._frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1, style='dotted')
                if not (x >= (self.last_x - e) and x <= (self.last_x + e) and y >= (self.last_y - e) and  y <= (self.last_y + e)):
                    self.clear = False

            # draw status
            if not self.is_manual:
                string = 'Label' if self.stop_n_frame == self.n_frame else 'Prev.' if self.stop_n_frame > self.n_frame else 'Af.'
            else:
                string = 'Manual Label' if self.stop_n_frame == self.n_frame else 'Prev. (Manual)' if self.stop_n_frame > self.n_frame else 'Af. (Manual)'
            cv2.putText(self._frame, string, (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)

            # draw manual label paths
            if len(self.tmp_line) > 1:
                color = (255, 255, 255)
                for i in range(1, len(self.tmp_line)):
                    cv2.line(self._frame, self.tmp_line[i - 1], self.tmp_line[i], color, 2)

            # draw rat contour
            rat_detector = RatDetector()
            rat_detector.detect_rat_contour(cv2.cvtColor(self._orig_frame, cv2.COLOR_BGR2GRAY))

            if self.n_frame not in self.rat_cnt_dict.keys():
                self.rat_cnt_dict[self.n_frame] = rat_detector.rat_cnt.tolist()

            if len(rat_detector.rat_cnt) > 0 and self.var_show_rat is not None and self.var_show_rat.get() == 1:
                cv2.drawContours(self._frame, rat_detector.rat_cnt, -1, (216, 233, 62), 2)

            # adjust the frame aspect ratio if the window is maximized
            if self.root.state() == 'zoomed':
                try:
                    shape = self._frame.shape
                    self.root.update()
                    r1 = (shape[1] / self.root.winfo_width())
                    r2 = (shape[0] / self.root.winfo_height())
                    shrink_r = max(r1, r2)
                    # shrink_r = r1
                    self._c_height = self._r_height/shrink_r
                    self._c_width = self._r_width/shrink_r

                    if r1 == shrink_r:
                        nw = int(shape[1] * self._c_width)
                        nh = int(shape[0] * nw / shape[1])
                        self._c_height = nw / shape[1]
                    else:
                        nh = int(shape[0] * self._c_height)
                        nw = int(shape[1] * nh / shape[0])
                        self._c_width = nh / shape[0]
                    df_w = self.display_frame.winfo_width()
                    if df_w == 1284:
                        pass
                    elif nw > df_w:
                        nn_w = df_w - 4
                        r = nn_w / nw
                        self._c_width = r * self._c_width

                        nn_h = int(nh * r)
                        self._c_height = nn_h / shape[0]
                        nh = nn_h
                        nw = nn_w
                    else:
                        LOGGER.info(df_w)

                    newsize = (nw, nh)
                    self._frame = cv2.resize(self._frame, newsize)
                except:
                    pass

            # convert frame into rgb
            self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

class Common(object):

    # see if a points is inside a circle
    def in_circle(self, pt, center, radius=10):
        x, y = pt
        x_c, y_c = center
        dx, dy = (x - x_c)**2, (y - y_c)**2
        if dx + dy <= radius**2:
            return True
        else:
            return False

    # see if a points is inside a rectangle
    def in_rect(pt, rect):
        x_condition = pt[0] > rect[0][0] and pt[0] < rect[1][0]
        y_condition = pt[1] > rect[0][1] and pt[1] < rect[1][1]

        if x_condition and y_condition:
            return True
        else:
            return False
    # center a tkinter window
    def center(self, win, rx=2, ry=2):
        win.update_idletasks()
        width = win.winfo_reqwidth()
        height = win.winfo_reqheight()
        x = (win.winfo_screenwidth() // rx) - (width // 2)
        y = (win.winfo_screenheight() // ry) - (height // 2)
        win.geometry('+%d+%d' % (x, y))

    def calc_iou(self, boxA, boxB):
        # boxA: xmin, ymin, xmax, ymax
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


class RatDetector(object):

    def detect_rat_contour(self, gray):

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # find contour with the biggest area
        self.rat_cnt = sorted(cnts, key=cv2.contourArea)[-1]

    def detect_on_rat(self, bbox):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        try:
            cnt = self.rat_cnt.reshape(len(self.rat_cnt), 2)
            poly = mplPath.Path(cnt)
            on_rat = False
            for x in [x1, x2]:
                for y in [y1, y2]:
                    on_rat = on_rat or poly.contains_point((x, y))
        except Exception as e:
            LOGGER.info('Error in detect_on_rat method', e)
            on_rat = False
        return on_rat

# return a triangle with pt as center
def tri(pt):
    x, y = pt
    x1 = 13
    y1 = 13
    return [np.array([(x-x1, y+y1), (x+x1, y+y1), (x, y-y1)]).reshape((-1, 1, 2))]

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=15):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style)

def draw_arrow(image, p, q, color, dist, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    if dist > 48:
        # draw arrow tail
        drawline(image, p, q, color, thickness, gap=7)
    else:
        cv2.line(image, p, q, color, thickness, line_type, shift)

    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
