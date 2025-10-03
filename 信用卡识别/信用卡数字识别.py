from imutils import contours
import numpy as np
import myutils, cv2, argparse, os, imutils

#参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",      default=r"credit_card_01.png")
ap.add_argument("-t", "--template",   default=r"ocr_a_reference.png")
args = vars(ap.parse_args())

FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#制作模板字典 
img  = cv2.imread(args["template"])
#灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#二值处理
ref = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
#找轮廓
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
refCnts, _ = myutils.sort_contours(refCnts, method="left-to-right")

digits = {i: cv2.resize(ref[y:y+h, x:x+w], (57, 88))
          for i, (x, y, w, h) in enumerate(map(cv2.boundingRect, refCnts))}

#2. 读信用卡图
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#3. 预处理 
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
#礼帽
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#梯度计算
gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (gradX.min(), gradX.max())
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#再次二值处理，消除空块
thresh   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

#4. 找数字块 
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#基于一般信用卡的数据
locs = [(x, y, w, h) for (x, y, w, h) in map(cv2.boundingRect, cnts)
        if 2.5 < w / float(h) < 4.0 and 40 < w < 55 and 10 < h < 20]
locs = sorted(locs, key=lambda x: x[0])
#安全检查
if not locs:
    print("未检测到任何数字块，请检查预处理或换图！")
    exit()

# 5. 识别每一组
output = []
for (gX, gY, gW, gH) in locs:
    group = gray[max(gY-5, 0):gY+gH+5, max(gX-5, 0):gX+gW+5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    digitCnts, _ = contours.sort_contours(digitCnts, method="left-to-right")

    groupDigits = []
    #找单个数字
    for c in digitCnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = cv2.resize(group[y:y+h, x:x+w], (57, 88))
        scores = [cv2.matchTemplate(roi, digits[d], cv2.TM_CCOEFF)[0, 0] for d in range(10)]
        groupDigits.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupDigits), (gX, gY-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupDigits)

#6. 结果
print("信用卡卡号:", "".join(output))
print("发卡机构  :", FIRST_NUMBER.get(output[0], "Unknown"))
cv_show("Output", image)
