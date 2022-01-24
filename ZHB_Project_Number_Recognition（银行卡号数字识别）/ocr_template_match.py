# 导入工具包
from imutils import contours
import numpy as np
import cv2
import myutils
from myutils import cv_show


# 设置参数
"""image = cv2.imread("images\credit_card_01.png")
template = cv2.imread("ocr_a_reference.png")"""

"""ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())"""

# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

"""""""""""""""""""""模板图像预处理open"""""""""""""""""""""
"""img = cv2.imread(args["template"])"""
img = cv2.imread("ocr_a_reference.png")
cv_show('img',img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)
"""""""""""""""""""""模板图像预处理end"""""""""""""""""""""


"""""""""""""""""""""模板图像数字轮廓计算open"""""""""""""""""""""
"""
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.findContours在opencv旧版本中返回3个参数，在新版本中返回2个参数
"""
#cv2.findContours()轮廓检测函数，接受的参数为二值图，即黑白的（不是灰度图）
#返回的list中每个元素都是图像中的一个轮廓
#轮廓，终点坐标 = （（复制二值图像进行处理），（只检测外轮廓），（只保留终点坐标））
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#将轮廓画到img中
cv2.drawContours(img,refCnts,-1,(0,0,255),3)  #-1表示画所有轮廓
cv_show('img',img)
print (np.array(refCnts).shape)  #显示图中圈出的是几个轮廓
#将轮廓排序
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #排序，从左到右，从上到下，返回排好序的轮廓
"""""""""""""""""""""模板图像数字轮廓计算end"""""""""""""""""""""

"""""""""""""""""""""对模板图像数字进行标记open"""""""""""""""""""""
#定义空字典，后面对每个轮廓进行标记对应的数字
digits = {}
# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):  # i是数字，c是第几个轮廓
	# 计算外接矩形并且resize成合适大小
	(x, y, w, h) = cv2.boundingRect(c)
	#把目标区域抠出来
	roi = ref[y:y + h, x:x + w]
	#把框放大
	roi = cv2.resize(roi, (57, 88))
	#每一个数字对应每一个抠出来的轮廓模板,放到模板字典里
	digits[i] = roi
"""""""""""""""""""""对模板图像数字进行标记end"""""""""""""""""""""

"""""""""""""""""""""输入图像预处理open"""""""""""""""""""""
# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  #9*3的核
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))    #5*5的核

#读取输入图像，预处理
"""image = cv2.imread(args["image"])"""
image = cv2.imread("images\credit_card_01.png")
cv_show('image',image)
#对图像大小设置
image = myutils.resize(image, width=300)
#灰度处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

#X的梯度,进行灰度图的边缘检测
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)#ksize=-1相当于用3*3的
#梯度取绝对值
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
#归一化
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

#需要将信用卡中四个一组的数字提取出来
#通过闭操作（先膨胀，再腐蚀，就模糊到一起了）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX',gradX)

#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0，让opencv自动做判断
thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

#再来一个闭操作，因为有些模糊区域中间有漏洞
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
cv_show('thresh',thresh)
"""""""""""""""""""""输入图像预处理end"""""""""""""""""""""


"""""""""""""""""""""输入图像标记轮廓open"""""""""""""""""""""
# 计算轮廓
"""thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)"""
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 将轮廓的框框再画到原图像上
cnts = threshCnts
cur_img = image.copy()
#将轮廓的框框cnts再画到原图像cur_img上
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
"""""""""""""""""""""输入图像标记轮廓end"""""""""""""""""""""

"""""""""""""""""""""从大轮廓（4个数字）中取每个数字并进行匹配open"""""""""""""""""""""
locs = []
# 遍历轮廓，先得到轮廓的外接矩形
for (i, c) in enumerate(cnts):
	# 计算矩形
	(x, y, w, h) = cv2.boundingRect(c)
	#计算轮廓外接矩形的宽高比（因为我们的目标区域的外接矩形宽高比和其他误识区域的宽高比不同）
	ar = w / float(h)
	# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
	if ar > 2.5 and ar < 4.0:
		if (w > 40 and w < 55) and (h > 10 and h < 20):
			#符合的留下来存到locs[]中
			locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序（4个大轮廓）
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个大轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs): #轮廓的xywh值取出来
	# initialize the list of group digits
	groupOutput = []

	# 根据坐标提取每一个大轮廓 ，+—5是把轮廓周围多取一点
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	cv_show('group',group) #得到大轮廓（4个数字）

	# 预处理；
	# 二值化
	group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)
	# 计算每一组的轮廓，排序
	"""group_,digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"""
	digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

	# 计算每一组中的每一个数值，每个数字都已框出
	for c in digitCnts:
		# 找到当前数值的轮廓，resize成合适的的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		#这里每个数字的轮廓要跟模板每个数字大小一样(57, 88)
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)

		#数字都框出以后，要和模板中的十个数挨个匹配
		# 计算匹配得分空list，谁匹配得分高就是谁
		scores = []

		# 和十个模板进行匹配，计算每一个得分
		for (digit, digitROI) in digits.items():
			# 模板匹配 roi是现在要检测的数字，digitROI是模板做好的
			result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result) #记录得分
			scores.append(score)

		# 得到最合适的数字，即得分最大的
		groupOutput.append(str(np.argmax(scores)))

	# 画出来
	cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# 得到结果
	output.extend(groupOutput)
"""""""""""""""""""""从大轮廓（4个数字）中取每个数字并进行匹配open"""""""""""""""""""""

"""""""""""""""""""""打印结果open"""""""""""""""""""""
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
"""""""""""""""""""""打印结果end"""""""""""""""""""""