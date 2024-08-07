import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# 输入相机固有参数
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]])

# 假设如果计算量很大，那么图像可以下采样一次。请注意，下采样以 2 的幂数完成，即 1,2,4,8,...
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

# 当前路径目录
path = os.getcwd()

# 输入保存图片的目录。请注意，必须为该特定实现命名图像
#img_dir = 路径 + '/样本数据集/'
img_dir = './images/'

# 在保存到点云之前，对于 PnP 新添加的点，添加了捆绑调整的规定。请注意，它仍然非常慢
bundle_adjustment = False

# 一个函数，用于缩小图像，以防 SfM 管道需要时间执行。
#下采样：减少计算时间，提高效率；丢失细节，会减少特征点
def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 2
	# cv2.GaussianBlur(img,(5,5),0)
	while(i <= downscale):
		img = cv2.pyrDown(img)
		# img = cv2.pyrUp(img)
		i = i + 1
	return img
	
# 一个函数，用于三角测量，给定图像对及其相应的投影矩阵
def Triangulation(P1, P2, pts1, pts2, K, repeat):
    #OpenCV的triangulatePoints函数以及其他一些函数通常要求输入的点集矩阵的维度顺序是先点的维度（列），再是点集的大小（行）。这种顺序与NumPy中数组的常规顺序（先行后列）不同。
    if not repeat:
        points1 = np.transpose(pts1)
        points2 = np.transpose(pts2)
    else:
        points1 = pts1
        points2 = pts2
#输出的3D坐标是齐次坐标，共四个维度，因此需要将前三个维度除以第四个维度以得到非齐次坐标xyz。这个坐标是在相机坐标系下的坐标，以输入的两个相机位姿所在的坐标系为准。
    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = cloud / cloud[3]

    return points1, points2, cloud


# 透视管道 -n -点
# X：一个Nx3的数组，包含N个3D点的坐标（x, y, z）。
# p：一个Nx2的数组，包含N个2D点的坐标（u, v），这些点是X中对应3D点在图像上的投影。
# K：相机的内参矩阵，一个3x3的矩阵。
# d：相机的畸变系数，通常是一个包含4个或5个元素的向量（取决于畸变模型）。
# p_0：另一个图片的特征点。
# initial：一个标志位，用于指示是否需要对输入数据进行初始处理（如转置）。
def PnP(X, p, K, d, p_0, initial):
    # print(X.shape, p.shape, p_0.shape)
    #初始化对特征点转置
    if initial == 1:
        # X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T
#返回旋转向量rvecs、平移向量t、一个布尔数组inliers（表示哪些点是内点，即符合模型的点），以及一个标志位ret（表示求解是否成功）
    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    # print(X.shape, p.shape, t, rvecs)
    #使用cv2.Rodrigues函数将旋转向量rvecs转换为旋转矩阵R
    R, _ = cv2.Rodrigues(rvecs)

# 如果存在内点（inliers非空），则根据inliers索引更新p、X和p_0，只保留内点对应的部分。
#返回旋转矩阵R、平移向量t、更新后的2D点p、3D点X和原始2D点p_0（可能已根据内点更新）。
    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0

# 主管道中重投影误差的计算
def ReprojectionError(X, pts, Rt, K, homogenity):
    total_error = 0
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    r, _ = cv2.Rodrigues(R)
    if homogenity == 1:
        X = cv2.convertPointsFromHomogeneous(X.T)

    p, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p = p[:, 0, :]
    p = np.float32(p)
    pts = np.float32(pts)
    if homogenity == 1:
        total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
    else:
        total_error = cv2.norm(p, pts, cv2.NORM_L2)
    pts = pts.T
    tot_error = total_error / len(p)
    #print(p[0], pts[0])

    return tot_error, X, p


# 计算束调整的重投影误差
def OptimReprojectionError(x):
	Rt = x[0:12].reshape((3,4))
	K = x[12:21].reshape((3,3))
	rest = len(x[21:])
	rest = int(rest * 0.4)
	p = x[21:21 + rest].reshape((2, int(rest/2)))
	X = x[21 + rest:].reshape((int(len(x[21 + rest:])/3), 3))
	R = Rt[:3, :3]
	t = Rt[:3, 3]
	
	total_error = 0
	
	p = p.T
	num_pts = len(p)
	error = []
	r, _ = cv2.Rodrigues(R)
	
	p2d, _ = cv2.projectPoints(X, r, t, K, distCoeffs = None)
	p2d = p2d[:, 0, :]
	#print(p2d[0], p[0])
	for idx in range(num_pts):
		img_pt = p[idx]
		reprojected_pt = p2d[idx]
		er = (img_pt - reprojected_pt)**2
		error.append(er)
	
	err_arr = np.array(error).ravel()/num_pts
	
	print(np.sum(err_arr))
	#err_arr = np.sum(err_arr)
	

	return err_arr

def BundleAdjustment(points_3d, temp2, Rtnew, K, r_error):

	# 设置要优化的优化变量
	opt_variables = np.hstack((Rtnew.ravel(), K.ravel()))
	opt_variables = np.hstack((opt_variables, temp2.ravel()))
	opt_variables = np.hstack((opt_variables, points_3d.ravel()))

	error = np.sum(OptimReprojectionError(opt_variables))
	corrected_values = least_squares(fun = OptimReprojectionError, x0 = opt_variables, gtol = r_error)

	corrected_values = corrected_values.x
	Rt = corrected_values[0:12].reshape((3,4))
	K = corrected_values[12:21].reshape((3,3))
	rest = len(corrected_values[21:])
	rest = int(rest * 0.4)
	p = corrected_values[21:21 + rest].reshape((2, int(rest/2)))
	X = corrected_values[21 + rest:].reshape((int(len(corrected_values[21 + rest:])/3), 3))
	p = p.T
	
	return X, p, Rt
	

def Draw_points(image, pts, repro):
    if repro == False:
        image = cv2.drawKeypoints(image, pts, image, color=(0, 255, 0), flags=0)
    else:
        for p in pts:
            image = cv2.circle(image, tuple(p), 2, (0, 0, 255), -1)
    return image


def to_ply(path, point_cloud, colors, densify):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    #print( verts.shape)
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    if not densify:
        with open(path + '/Point_Cloud/sparse_pyrDown.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open(path + '/Point_Cloud/dense_.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
            
# 相机姿势注册。这对于可视化每个相机的姿势以及点云非常有用。目前它已被禁用。稍后会修复。
def camera_orientation(path, mesh, R_T, i):
    T = np.zeros((4, 4))
    T[:3, ] = R_T
    T[3, :] = np.array([0, 0, 0, 1])
    new_mesh = copy.deepcopy(mesh).transform(T)
    # print(new_mesh)
    #new_mesh.scale(0.5, center=new_mesh.get_center())
    o3d.io.write_triangle_mesh(path + "/Point_Cloud/camerapose" + str(i) + '.ply', new_mesh)
    return


def common_points(pts1, pts2, pts3):
# '''这里pts1代表图像2在1-2匹配时找到的点
#     pts2 是在 2-3 ''' 匹配过程中在图像 2 中找到的点
    indx1 = []
    indx2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i, :])
        if a[0].size == 0:
            pass
        else:
            indx1.append(i)
            indx2.append(a[0][0])

#    '''temp_array1和temp_array2将不常见'''
    temp_array1 = np.ma.array(pts2, mask=False)
    temp_array1.mask[indx2] = True
    temp_array1 = temp_array1.compressed()
    temp_array1 = temp_array1.reshape(int(temp_array1.shape[0] / 2), 2)

    temp_array2 = np.ma.array(pts3, mask=False)
    temp_array2.mask[indx2] = True
    temp_array2 = temp_array2.compressed()
    temp_array2 = temp_array2.reshape(int(temp_array2.shape[0] / 2), 2)
    print("Shape New Array", temp_array1.shape, temp_array2.shape)
    return np.array(indx1), np.array(indx2), temp_array1, temp_array2

# 对两张图像进行特征检测，然后进行特征匹配
def find_features(img0, img1):
    ##########################SIFT##########################
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # t1 = time.perf_counter()  
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    
    #lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    # t2= time.perf_counter()  
    # elapsed_time = t2 - t1  
    # print(f"SIFT特征提取所需时间: {elapsed_time} 秒")
    #pts0 = np.float32([m.pt for m in kp0])
    #pts1, st, err = cv2.calcOpticalFlowPyrLK(img0gray, img1gray, pts0, None, **lk_params)
    #pts0 = pts0[st.ravel() == 1]
    #pts1 = pts1[st.ravel() == 1]
    #print(pts0.shape, pts1.shape)

    bf = cv2.BFMatcher()
    # t1 = time.perf_counter() 
    #返回与样本每个特征嗲最相似的两个特征点[<DMatch 0x7f117af995f0>, <DMatch 0x7f117af99610>]
    #DMatch数据结构
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    matches = bf.knnMatch(des0, des1, k=2)
    # 比例筛选
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    # t2= time.perf_counter()  
    # elapsed_time = t2 - t1  
    # print(f"SIFT提取下暴力匹配所需时间: {elapsed_time} 秒")
    # print(f"SIFR提取下特征点对数: {len(good)} 对")
    # h, w = img0.shape[:2]  
    # # 创建一个足够大的图像来显示两个图像和匹配项  
    # # 这里我们简单地将两个图像并排放置，并留出一些空间  
    # result_img_match = np.zeros((h, 2*w + 100, 3), dtype=np.uint8)  # +100是为了留出一些空间
    # result_img_match =cv2.drawMatches(img0, kp0, img1, kp1, good,result_img_match,matchesThickness=1);
    # cv2.imshow("all good matches", result_img_match)
    # cv2.imshow("all matches", result_img_goodmatch)
    # cv2.waitKey(0) 
##############################################################################

##########################ORB##########################
    # img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # orb = cv2.ORB.create()
    # t1 = time.perf_counter()  
    # kp0, des0 = orb.detectAndCompute(img0gray, None)
    # kp1, des1 = orb.detectAndCompute(img1gray, None)
    # t2= time.perf_counter()  
    # elapsed_time = t2 - t1  
    # print(f"ORB特征提取所需时间: {elapsed_time} 秒")
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    # t1 = time.perf_counter()  
    # matches = bf.match(des0, des1) 
    # # 对匹配结果进行排序  
    # matches = sorted(matches, key=lambda x: x.distance)  
    # min_dist = min(m.distance for m in matches) 
    # good_matches = [m for m in matches if m.distance < max((min_dist*2),40)]  
    # t2= time.perf_counter()  
    # elapsed_time = t2 - t1  
    # print(f"ORB提取下暴力匹配所需时间: {elapsed_time} 秒")
    # print(f"ORB提取下特征点对数: {len(good_matches)} 对")
    # h, w = img0.shape[:2]  
    # # 创建一个足够大的图像来显示两个图像和匹配项  
    # # 这里我们简单地将两个图像并排放置，并留出一些空间  
    # result_img_match = np.zeros((h, 2*w + 100, 3), dtype=np.uint8)  # +100是为了留出一些空间
    # result_img_match =cv2.drawMatches(img0, kp0, img1, kp1, good,result_img_match,matchesThickness=1);
    # cv2.imshow("all good matches", result_img_match)
    # # cv2.imshow("all matches", result_img_goodmatch)
    # cv2.waitKey(0) 
    ########################################################
    return pts0, pts1


#创建窗口
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#摊平多维为一维，与原数据共享数据，修改任意一个，二者都会改
posearr = K.ravel()

R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
R_t_1 = np.empty((3, 4))
#矩阵相乘
P1 = np.matmul(K, R_t_0)
Pref = P1
P2 = np.empty((3, 4))

Xtot = np.zeros((1, 3))
colorstot = np.zeros((1, 3))

#排序
img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    #.lower将文件名全部小写
    if '.jpg' in img.lower() or '.png' in img.lower():
        images = images + [img]
i = 0
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()


densify = False  # 添加以防我们将合并此代码中的致密化步骤。大多数情况下，它会被单独考虑，但仍然添加在这里。

# 设置参考两帧
img0 = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
img1 = img_downscale(cv2.imread(img_dir + '/' + images[i + 1]), downscale)

pts0, pts1 = find_features(img0, img1)

# 求取本征矩阵，使用RANSAC方法，进一步排除失配点。
#mask是判断是内外点的数组
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
#留下内点，排除外点
pts0 = pts0[mask.ravel() == 1]
pts1 = pts1[mask.ravel() == 1]
# 获得的姿态是第二张图像相对于第一张图像的姿态
_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)  # |找到姿势
#返回的mask,mask将更新，只包含那些通过手性检验的内点，也就是那些在3D空间中位于相机前方的点
pts0 = pts0[mask.ravel() > 0]
pts1 = pts1[mask.ravel() > 0]

R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

#空间P2坐标
P2 = np.matmul(K, R_t_1)

# 对第一对图像进行三角测量。位姿将被设置为参考，用于增量 SfM
pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat=False)


#*************************************待优化******************************************
# 将 3D 点回溯到图像上并计算重投影误差。理想情况下它应该小于一。
# 如果发现点云不正确的罪魁祸首，请启用捆绑调整
# 返回平均误差、调整后的非齐次三维点集X（如果进行了齐次坐标转换）、以及重投影点p。
error, points_3d, repro_pts = ReprojectionError(points_3d, pts1, R_t_1, K, homogenity = 1)  
print("REPROJECTION ERROR: ", error)
#返回旋转矩阵R、平移向量t、更新后的2D点p、3D点X和原始2D点p_0（可能已根据内点更新）。
Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
#Xtot = np.vstack((Xtot, points_3d))

#*************************************************************************************
#生成3维对角矩阵
R = np.eye(3)
t = np.array([[0], [0], [0]], dtype=np.float32)

# 这里，要考虑的总图像可以变化。理想情况下，可以使用整套，也可以使用其中的一部分。对于整批：使用 tot_imgs = len(images) -2
tot_imgs = len(images) - 2 

posearr = np.hstack((posearr, P1.ravel()))
posearr = np.hstack((posearr, P2.ravel()))

gtol_thresh = 0.5
#camera_orientation(path, mesh, R_t_0, 0)
#camera_orientation(path, mesh, R_t_1, 1)

#tqdm是一个快速、可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)
for i in tqdm(range(tot_imgs)):
   # 获取要添加到管道中的新图像并获取与图像对的匹配
    img2 = img_downscale(cv2.imread(img_dir + '/' + images[i + 2]), downscale)

    # pts0,pts1 = find_features(img1,img2)

    pts_, pts2 = find_features(img1, img2)
    if i != 0:
        pts0, pts1, points_3d = Triangulation(P1, P2, pts0, pts1, K, repeat = False)
        pts1 = pts1.T
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)#齐次转非齐
        points_3d = points_3d[:, 0, :] #转换为二维数组，将points_3d转换回了一个形状为(N, 3)的二维数组，其中每行代表一个三维空间中的点。
    
   # pts1 和 pts_ 之间有一些共同点
    # 我们需要找到 pts1 的 indx1 与 pts_ 中的 indx2 匹配
    indx1, indx2, temp1, temp2 = common_points(pts1, pts_, pts2)
    com_pts2 = pts2[indx2]
    com_pts_ = pts_[indx2]
    com_pts0 = pts0.T[indx1]
    # 我们有新图像的 3D -2D 对应关系以及之前获得的点云。公共点可用于找到新图像的世界坐标
    # 使用透视 -n -点 (PnP)
    Rot, trans, com_pts2, points_3d, com_pts_ = PnP(points_3d[indx1], com_pts2, K, np.zeros((5, 1), dtype=np.float32), com_pts_, initial = 0)
    # 找到新图像的等效投影矩阵
    Rtnew = np.hstack((Rot, trans))
    Pnew = np.matmul(K, Rtnew)

    #print(Rtnew)
    error, points_3d, _ = ReprojectionError(points_3d, com_pts2, Rtnew, K, homogenity = 0)
   
    
    temp1, temp2, points_3d = Triangulation(P2, Pnew, temp1, temp2, K, repeat = False)
    error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 1)
    print("Reprojection Error: ", error)
   # 我们正在存储每个图像的姿势。这在多视图立体期间非常有用，因为应该知道这一点
    posearr = np.hstack((posearr, Pnew.ravel()))

# 如果考虑捆绑调整。 gtol_thresh 表示梯度阈值或更新中可能发生的最小跳跃。如果跳跃较小，则终止优化。
    # 请注意，大多数时候，管道产生的重投影误差小于 0.1！然而，它往往太慢，往往每帧接近半分钟！
# 对于点云注册，点在 NumPy 数组中更新。为了使对象可视化，相应的 BGR 颜色也会更新，并将被合并
    # 最后有 3D 点
    if bundle_adjustment:
        print("Bundle Adjustment...")
        points_3d, temp2, Rtnew = BundleAdjustment(points_3d, temp2, Rtnew, K, gtol_thresh)
        Pnew = np.matmul(K, Rtnew)
        error, points_3d, _ = ReprojectionError(points_3d, temp2, Rtnew, K, homogenity = 0)
        print("Minimized error: ",error)
        Xtot = np.vstack((Xtot, points_3d))
        pts1_reg = np.array(temp2, dtype=np.int64)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg],dtype=np.int64)
        colorstot = np.vstack((colorstot, colors))
    else:
        Xtot = np.vstack((Xtot, points_3d[:, 0, :]))
        pts1_reg = np.array(temp2, dtype=np.int32)
        colors = np.array([img2[l[1], l[0]] for l in pts1_reg.T])
        colorstot = np.vstack((colorstot, colors)) 
    # camera_orientation(path, mesh, Rtnew, i + 2)    


    R_t_0 = np.copy(R_t_1)
    P1 = np.copy(P2)
    plt.scatter(i, error)
    plt.pause(0.05)

    img0 = np.copy(img1)
    img1 = np.copy(img2)
    pts0 = np.copy(pts_)
    pts1 = np.copy(pts2)
    #P1 = np.copy(P2)
    P2 = np.copy(Pnew)
    cv2.imshow('image', img2)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

plt.show()
cv2.destroyAllWindows()

# 最后使用open3d对得到的点云进行注册并保存。以.ply形式保存，可以使用meshlab查看
print("Processing Point Cloud...")
print(Xtot.shape, colorstot.shape)
to_ply(path, Xtot, colorstot, densify)
print("Done!")
# 保存所有图像的投影矩阵
np.savetxt('pose.csv', posearr, delimiter = '\n')

