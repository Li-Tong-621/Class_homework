import numpy as np
import cv2
import drawMesh
import os
import time


# 矩阵A的极分解
def polar_decomposition(A):
	u, s, vh = np.linalg.svd (A)
	smat = np.diag (s)
	r = u@vh
	S = vh.T@smat@vh
	return (r, S)


# 预先求出每个三角形理想变换矩阵A,存贮将A极分解后得到旋转矩阵的theta值和S
def get_A(v_start, v_end, faces, theta, S):
	for i, face in enumerate (faces):
		# 根据公式1先得到p,q
		p = np.zeros ((3, 3))
		q = np.zeros ((3, 2))
		for j, v_id in enumerate (face):
			p[j][0] = v_start[v_id][0]
			p[j][1] = v_start[v_id][1]
			p[j][2] = 1
			q[j][0] = v_end[v_id][0]
			q[j][1] = v_end[v_id][1]
		A1 = np.dot (np.linalg.pinv (p), q)
		A = A1[:2]
		r, S[i] = polar_decomposition (A)
		# 根据旋转矩阵r求出旋转角度theta,以便对theta插值
		if r[0][1] < 0:
			theta[i] = np.arccos(r[0][0])
		else:
			theta[i] = -np.arccos(r[0][0])


# pPoint, qPoint  每行对应一个点索引的坐标x,y;
# index是二维数组，一行三个点索引表示一个三角形, 索引必须从0开始！
# tSize表示三角形总数
def get_H(v_start, v_end, index, tSize):
	L = np.array ([[1], [1], [1]])
	P = np.zeros ((3, 2))
	Q = np.zeros ((3, 2))
	h = np.zeros ((tSize, 6))  # 论文中每个三角形对应的H1~H6
	H = np.zeros ((2 * len (v_start), 2 * len (v_start)))
	for i in range (tSize):  # 每个三角形
		# 得到h
		for j in range (3):
			P[j] = v_start[index[i][j]]
			Q[j] = v_end[index[i][j]]
		P1 = np.hstack ((P, L))
		h1 = np.linalg.pinv (P1)
		# 计算小h1-h6
		for j in range (3):
			h[i][j] = h1[0][j]
			h[i][j + 3] = h1[1][j]
		# 计算H
		for j in range (3):
			for k in range (3):
				H[index[i][j] * 2][index[i][k] * 2] += h[i][j] * h[i][k] + h[i][j + 3] * h[i][k + 3]
				H[index[i][j] * 2 + 1][index[i][k] * 2 + 1] += h[i][j] * h[i][k] + h[i][j + 3] * h[i][k + 3]
	Hv = np.linalg.pinv (H[2:][:,2:])
	return Hv, H, h


# 获取t时刻的所有点的坐标
def process(t, theta, S, Hv, H, h, index, v_start0, v_end0):
	g = np.zeros (H.shape[0])
	I = np.eye(2)
	vt0 = (1 - t) * v_start0 + t * v_end0
	for i, s in enumerate (S):  # 第i个三角形，角度是th0~4
		# 计算g中的a
		# 对角度theta, S进行线性插值
		A = np.array ([[np.cos (t * theta[i]), -np.sin (t * theta[i])], [np.sin (t * theta[i]), np.cos (t * theta[i])]])
		A = np.dot (A, (1 - t) * I + t * s)
		A1, A2 = -A[:, 0], -A[:, 1]
		# 计算g
		for j in range (3):  # 第i个三角形的第j个点
			g[index[i][j] * 2] += np.array ([h[i][j], h[i][j + 3]]).dot (A1)
			g[index[i][j] * 2 + 1] += np.array ([h[i][j], h[i][j + 3]]).dot (A2)
	g += vt0[0] * H[0] + vt0[1] * H[1]
	v_t = -Hv.dot (g[2:])
	v_t = v_t.reshape (-1, 2)
	return np.vstack((vt0, v_t))


def frame_by_frame(event, x, y, flags, param):
	global frames, theta, S, hv, H, h, triangle, V_start, V_end, img, img2, count
	drawMesh.draw_mesh_red(V_end,edges,img2)
	if count == (frames+1):
		count = 1
	if event == cv2.EVENT_LBUTTONDOWN:
		t = count/frames
		print('Time:', t)
		start_time = time.time()
		V_t = process(t, theta, S, hv, H, h, triangle, V_start[0], V_end[0])
		elapsed_time = time.time() - start_time
		print('Time taken for ARAP Interplation:', elapsed_time)
		count = count+1
		img = img2.copy()
		drawMesh.draw_mesh(V_t,edges,img)
		cv2.imshow(windowName, img)

if __name__ == "__main__":
	num = 0
	f=open('name.txt','r')
	L=f.readline()
	openfile_name1,openfile_name2=L.split(',')
	"""print(openfile_name1)
	print(openfile_name2)"""
	openfile_name1='horse_S.obj'
	openfile_name2 = 'horse_T.obj'
	openfile_name1, openfile_name2 = L.split(',')
	windowName = 'ARAP'
	cv2.setUseOptimized (True)

	__location__ = os.path.realpath (os.path.join (os.getcwd (), os.path.dirname (__file__)))
	print("please wait a few second...")

	# CHANGE SOURCE AND TARGET MESH HERE
	start = open (os.path.join (__location__, openfile_name1), 'r')
	end = open (os.path.join (__location__, openfile_name2), 'r')

	no_vertices, no_faces, V_start, faces = drawMesh.read_file (start)
	_, _, V_end, faces_end = drawMesh.read_file (end)

	triangle = (faces - 1).astype("int32")
	hv, H, h = get_H(V_start, V_end, triangle, no_faces)

	theta = np.zeros(no_faces)
	S = np.zeros((no_faces, 2, 2))
	get_A(V_start, V_end, triangle, theta, S)

	# CHANGE NUMBER OF FRAMES HERE
	frames = 20

	img = np.zeros ((800, 1280, 3), np.uint8)
	img2 = img.copy ()
	img_clear = img.copy ()

	edges = drawMesh.get_edges (no_faces, faces)

	cv2.namedWindow (windowName)
	cv2.setMouseCallback (windowName, frame_by_frame)

	count = 1

	drawMesh.draw_mesh_red (V_end, edges, img)
	drawMesh.draw_mesh (V_start, edges, img)
	cv2.namedWindow (windowName)

	V_t = np.zeros ((no_vertices, 2))
	print("done")

	while (True):
		cv2.imshow (windowName, img)
		# PRESS SPACE BAR TO RECORD A VIDEO
		tt = cv2.waitKey (1)  # esc键退出
		if tt == 27:
			break
		if tt == 32:  # 空格键开始播放
			count = 1
			drawMesh.draw_mesh (V_start, edges, img)
			fourcc = cv2.VideoWriter_fourcc ('M', 'J', 'P', 'G')
			# 将运动的视频写入文件中
			out = cv2.VideoWriter ('arap_interpolation.avi', fourcc, 20, (1280, 800), isColor=True)

			for k in range (frames):
				t = count / frames
				print ('Time:', t)
				V_t = process (t, theta, S, hv, H, h, triangle, V_start[0], V_end[0])
				count = count + 1
				img = img_clear.copy ()
				drawMesh.draw_mesh (V_t, edges, img)
				cv2.imshow (windowName, img)
				cv2.waitKey (1)
				out.write (img)

	try:
		out.release()
	except:
		print("close")
	cv2.destroyAllWindows ()
