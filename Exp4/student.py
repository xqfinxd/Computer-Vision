# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # raise NotImplementedError()
    height, width, channel = images[0].shape
    albedo = np.zeros((height, width, channel))
    normals = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                I = np.array([image[h, w, c] for image in images])
                L = lights
                G = np.dot(np.linalg.inv(np.dot(L.T, L)), np.dot(L.T, I))
                kd = abs(np.linalg.norm(G))
                if kd < 1e-7:
                    N = np.zeros((3, 1))
                else:
                    N = G / kd
                albedo[h, w, c] = kd
                normals[h, w] = N.reshape(-1)
    return albedo, normals

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # raise NotImplementedError()
    height, width, _ = points.shape
    projection = np.zeros((height, width, 2))
    for h in range(height):
        for w in range(width):
            p = points[h, w]
            loc = np.ones(4)
            loc[:3] = p
            loc = np.dot(Rt, loc.T)
            loc = np.dot(K, loc.T)
            if loc[2] < 1e-7:
                projection[h, w] = [0, 0]
                continue
            loc = loc / loc[2]
            projection[h, w] = loc[:2]
    return projection


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # raise NotImplementedError()
    height, width, channel = image.shape
    size2 = ncc_size * ncc_size
    normalized = np.zeros((height, width, channel*size2))
    ncc = ncc_size/2
    for h in range(height):
        for w in range(width):
            if h-ncc < 0 or h+ncc >= height or w-ncc < 0 or w+ncc >= width:
                continue
            temp = []
            for c in range(channel):
                W = image[h-ncc : h+ncc+1, w-ncc : w+ncc+1, c]
                W = (W - np.mean(W)).reshape(-1)
                temp.append(W)
            allW = np.hstack(temp).reshape(-1)
            norm = np.linalg.norm(allW)
            if norm < 1e-7:
                allW = np.zeros_like(allW)
            else:
                allW = allW / norm
            normalized[h, w] = allW
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # raise NotImplementedError()
    height, width, channel = image1.shape
    ncc = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            ncc[h, w] = np.dot(image1[h, w], image2[h, w])
    return ncc


def form_poisson_equation_impl(height, width, alpha, normals, depth_weight, depth):
    """
    Creates a Poisson equation given the normals and depth at every pixel in image.
    The solution to Poisson equation is the estimated depth. 
    When the mode, is 'depth' in 'combine.py', the equation should return the actual depth.
    When it is 'normals', the equation should integrate the normals to estimate depth.
    When it is 'both', the equation should weight the contribution from normals and actual depth,
    using  parameter 'depth_weight'.

    Input:
        height -- height of input depth,normal array
        width -- width of input depth,normal array
        alpha -- stores alpha value of at each pixel of image. 
            If alpha = 0, then the pixel normal/depth should not be 
            taken into consideration for depth estimation
        normals -- stores the normals(nx,ny,nz) at each pixel of image
            None if mode is 'depth' in combine.py
        depth_weight -- parameter to tradeoff between normals and depth when estimation mode is 'both'
            High weight to normals mean low depth_weight.
            Giving high weightage to normals will result in smoother surface, but surface may be very different from
            what the input depthmap shows.
        depth -- stores the depth at each pixel of image
            None if mode is 'normals' in combine.py
    Output:
        constants for equation of type Ax = b
        A -- left-hand side coefficient of the Poisson equation 
            note that A can be a very large but sparse matrix so csr_matrix is used to represent it.
        b -- right-hand side constant of the the Poisson equation
    """

    assert alpha.shape == (height, width)
    assert normals is None or normals.shape == (height, width, 3)
    assert depth is None or depth.shape == (height, width)

    '''
    Since A matrix is sparse, instead of filling matrix, we assign values to a non-zero elements only.
    For each non-zero element in matrix A, if A[i,j] = v, there should be some index k such that, 
        row_ind[k] = i
        col_ind[k] = j
        data_arr[k] = v
    Fill these values accordingly
    '''
    row_ind = []
    col_ind = []
    data_arr = []
    '''
    For each row in the system of equation fill the appropriate value for vector b in that row
    '''
    b = []
    if depth_weight is None:
        depth_weight = 1

    '''
    TODO
    Create a system of linear equation to estimate depth using normals and crude depth Ax = b

    x is a vector of depths at each pixel in the image and will have shape (height*width)

    If mode is 'depth':
        > Each row in A and b corresponds to an equation at a single pixel
        > For each pixel k, 
            if pixel k has alpha value zero do not add any new equation.
            else, fill row in b with depth_weight*depth[k] and fill column k of the corresponding
                row in A with depth_weight.

        Justification: 
            Since all the elements except k in a row is zero, this reduces to 
                depth_weight*x[k] = depth_weight*depth[k]
            you may see that, solving this will give x with values exactly same as the depths, 
            at pixels where alpha is non-zero, then why do we need 'depth_weight' in A and b?
            The answer to this will become clear when this will be reused in 'both' mode

    Note: The normals in image are +ve when they are along an +x,+y,-z axes, if seen from camera's viewpoint.
    If mode is 'normals':
        > Each row in A and b corresponds to an equation of relationship between adjacent pixels
        > For each pixel k and its immideate neighbour along x-axis l
            if any of the pixel k or pixel l has alpha value zero do not add any new equation.
            else, fill row in b with nx[k] (nx is x-component of normal), fill column k of the corresponding
                row in A with -nz[k] and column k+1 with value nz[k]
        > Repeat the above along the y-axis as well, except nx[k] should be -ny[k].

        Justification: Assuming the depth to be smooth and almost planar within one pixel width.
        The normal projected in xz-plane at pixel k is perpendicular to tangent of surface in xz-plane.
        In other word if n = (nx,ny,-nz), its projection in xz-plane is (nx,nz) and if tangent t = (tx,0,tz),
            then n.t = 0, therefore nx/-nz = -tz/tx
        Therefore the depth change with change of one pixel width along x axis should be proporational to tz/tx = -nx/nz
        In other words (depth[k+1]-depth[k])*nz[k] = nx[k]
        This is exactly what the equation above represents.
        The negative sign in ny[k] is because the indexing along the y-axis is opposite of +y direction.

    If mode is 'both':
        > Do both of the above steps.

        Justification: The depth will provide a crude estimate of the actual depth. The normals do the smoothing of depth map
        This is why 'depth_weight' was used above in 'depth' mode. 
            If the 'depth_weight' is very large, we are going to give preference to input depth map.
            If the 'depth_weight' is close to zero, we are going to give preference normals.
    '''
    # TODO Block Begin
    # fill row_ind,col_ind,data_arr,b
    # raise NotImplementedError()
    row = 0
    if depth is not None:
        for i in range(height):
            for j in range(width):
                if alpha[i, j] != 0:
                    row_ind.append(row)
                    row += 1
                    col_ind.append(i*width+j)
                    data_arr.append(depth_weight)
                    b.append(depth_weight*depth[i, j])
    if normals is not None:
        for i in range(height):
            for j in range(width):
                if alpha[i, j] != 0 and \
                        (i+1 < height and j+1 < width) and \
                        (alpha[i+1, j] != 0 and alpha[i, j+1] != 0):
                    row_ind.append(row)
                    col_ind.append(i*width+j)
                    data_arr.append(-normals[i, j, 2])
                    row_ind.append(row)
                    col_ind.append(i*width+j+1)
                    data_arr.append(normals[i, j, 2])
                    row += 1
                    b.append(normals[i, j, 0])
                    row_ind.append(row)
                    col_ind.append(i*width+j)
                    data_arr.append(-normals[i, j, 2])
                    row_ind.append(row)
                    col_ind.append((i+1)*width+j)
                    data_arr.append(normals[i, j, 2])
                    row += 1
                    b.append(-normals[i, j, 1])
    # TODO Block end
    # Convert all the lists to numpy array
    row_ind = np.array(row_ind, dtype=np.int32)
    col_ind = np.array(col_ind, dtype=np.int32)
    data_arr = np.array(data_arr, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Create a compressed sparse matrix from indices and values
    A = csr_matrix((data_arr, (row_ind, col_ind)), shape=(row, width * height))
    return A, b
