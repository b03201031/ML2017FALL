import numpy as np 
from skimage import io
from skimage.viewer import ImageViewer
from skimage import transform
import sys

target_img = sys.argv[1]+'/' + sys.argv[2]
new_shape = (600, 600)
def get_imgs(folder_path):
	path = folder_path + '/{}.jpg'
	imgs = []
	for idx in range(415):
		img = io.imread(path.format(idx)) 
		img = transform.resize(img, new_shape)
		img = img.flatten()
		#print(img.shape)
		#break
		imgs.append(img)
	imgs = np.asarray(imgs)
	return imgs	

def get_eigenface(U, X, idx):
	M = np.reshape(U[:, idx], X.shape[0], 1)
	M -= np.min(M)
	M /= np.max(M)
	M = (M*255*-1).astype(np.uint8)
	M = M.reshape(new_shape + (3, ))
	return M

def reconstruct(img_path, U, X_mean, nb_eigenface):
	img = io.imread(img_path)
	img = transform.resize(img, new_shape)
	img = img.flatten()
	img = img.reshape(1, len(img))
	img -= X_mean.T
	#print(img.shape)
	weight = np.dot(img, U[:, :nb_eigenface])
	#print(weight)
	reconstruct_img = np.zeros(img.shape[1]).reshape(img.shape[1], 1)
	#print(reconstruct_img.shape)
	print(weight.shape)
	for idx, w in enumerate(weight.T):
		reconstruct_img = reconstruct_img + U[:, idx].reshape(img.shape[1], 1) * w
	
	print(reconstruct_img.shape)
	print(X_mean.shape)
	reconstruct_img += X_mean
	reconstruct_img -= np.min(reconstruct_img)
	reconstruct_img /= np.max(reconstruct_img)
	reconstruct_img = (reconstruct_img*255).astype(np.uint8)
	reconstruct_img = reconstruct_img.reshape(new_shape + (3, ))
	return reconstruct_img
	
def eigenvalue_ratiob(s, nb_taken):
	s_square = s*s
	s_square_sum = np.sum(s_square)
	for idx in range(nb_taken):
		print(s_square[idx]/s_square_sum)





def main():
	X = get_imgs(sys.argv[1]).T
	X_mean = np.mean(X, axis = 1).reshape(X.shape[0], 1)
	
	#U = np.load('U_{}.npy'.format(new_shape[0]))
	#s = np.load('s_{}.npy'.format(new_shape[0]))
	#V = np.load('V_{}.npy'.format(new_shape[0]))

	
	
	U, s, V = np.linalg.svd(X-X_mean, full_matrices=False)
	#np.save('U_{}'.format(new_shape[0]), U)
	#np.save('s_{}'.format(new_shape[0]), s)
	#np.save('V_{}'.format(new_shape[0]), V)
	
	#viewer = ImageViewer(reconstruct_img)
	reconstruct_img = reconstruct(target_img, U, X_mean, 4)
	io.imsave('reconstruction.jpg', reconstruct_img)
	
	'''
	for idx in range(4):
		img_idx = int(random.random()*414.)
		
		target_img = sys.argv[1]+'/' + '{}.jpg'.format(img_idx)
		print('reconstruct ', target_img)
		reconstruct_img = reconstruct(target_img, U, X_mean, 4)
		io.imsave('reconstruct_{}.jpg'.format(idx), reconstruct_img)
	'''
if __name__ == '__main__':
	main()