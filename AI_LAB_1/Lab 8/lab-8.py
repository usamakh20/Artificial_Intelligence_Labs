from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

im = Image.open('image.JPG')
sift = cv2.xfeatures2d.SIFT_create()

while(True):
    op = int(input("\n1. Task 1\n2. Task 2\n3. Task 3\n4. Task 4\n5. Task 5\n6. Task 6\n7. Exit\nChoose Option:"))
    if(op==1):
        im.resize([512,384],Image.ANTIALIAS).rotate(-45).convert('L').save("task 1.png","png")
    elif(op==2 or op==3 or op==4):
        mat = np.array(im)
        grey = (np.dot(mat[..., :3], [0.299, 0.587, 0.114]))
        if (op == 2):
            Image.fromarray(np.uint8(grey)).save("task 2.png", "png")
        elif (op == 3):
            plt.hist(grey.ravel())
            plt.savefig('task 3_1.png')
            equ = cv2.equalizeHist(grey.astype(np.uint8))
            cv2.imwrite('task 3_2.png', equ)
        elif (op==4):
            grey = np.float32(grey)
            dst = cv2.cornerHarris(grey, 2, 3, 0.04)
            mat[dst > 0.01 * dst.max()] = [0, 0, 255]
            cv2.imwrite('task 4.png', mat)
    elif (op==5 or op==6):
        img = cv2.imread('image.jpg',0)
        gray=np.empty(img.shape)
        kp, des = sift.detectAndCompute(img, None)
        if(op==5):
            gray = cv2.drawKeypoints(img, kp, gray)
            cv2.imwrite('task 5.png', gray)
        elif(op==6):
            MIN_MATCH_COUNT = 10

            img1 = cv2.imread('small.png', 0)  # trainImage

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des, des1, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w = img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                img2 = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            else:
                print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
                matchesMask = None

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img, kp, img1, kp1, good, None, **draw_params)

            cv2.imwrite("task 6.png",img3)

    elif(op==7):
        break

    else: print "Invalid Option!!"



