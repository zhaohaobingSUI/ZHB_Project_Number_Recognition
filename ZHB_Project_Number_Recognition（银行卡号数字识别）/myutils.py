import cv2


"""""""""""""""""""""绘图展示open"""""""""""""""""""""
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
"""""""""""""""""""""绘图展示end"""""""""""""""""""""


"""""""""""""""""""""定义轮廓排序函数open"""""""""""""""""""""
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    #计算每一个 外接矩形boundingBoxes
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    #对轮廓们根据左上角的坐标进行排序操作
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes
"""""""""""""""""""""定义轮廓排序函数end"""""""""""""""""""""



def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized