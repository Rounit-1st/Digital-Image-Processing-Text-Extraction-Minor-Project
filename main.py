import cv2 as cv
import numpy as np

###########################
image_path = "sample.jpg"
###########################

def getLines(img, threshold, angle_step=1):
    """hough line using vectorized numpy operations,
    may take more memory, but takes much less time"""
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    xcosthetas = np.dot(x_idxs.reshape((-1,1)), cos_theta.reshape((1,-1)))
    ysinthetas = np.dot(y_idxs.reshape((-1,1)), sin_theta.reshape((1,-1)))
    rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
    rhosmat = rhosmat.astype(np.int16)
    for i in range(num_thetas):
        _rhos,counts = np.unique(rhosmat[:,i], return_counts=True)
        accumulator[_rhos,i] = counts
    # Thresholding
    idxs = np.argwhere(accumulator > threshold)
    rho_idxs, theta_idxs = idxs[:,0], idxs[:,1]
    return np.column_stack((rhos[rho_idxs], thetas[theta_idxs]))


def line_intersection(line1, line2):
    """Find intersection point of two lines defined by (rho, theta)"""
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    try:
        x0, y0 = np.linalg.solve(A, b)
        return int(x0[0]), int(y0[0])
    except np.linalg.LinAlgError:
        return None


def get_corners_from_lines(lines, img_shape):
    """Extract 4 corner points from detected lines"""
    if len(lines) < 4:
        return None
    
    # Separate lines into horizontal and vertical based on angle
    horizontal = []
    vertical = []
    
    for rho, theta in lines:
        angle_deg = np.rad2deg(theta)
        # Vertical lines: close to -90° or 90°
        if abs(angle_deg) > 45:
            vertical.append((rho, theta))
        else:
            horizontal.append((rho, theta))
    
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    
    # Sort to get top/bottom horizontal and left/right vertical
    horizontal.sort(key=lambda x: x[0])
    vertical.sort(key=lambda x: x[0])
    
    # Get the two most extreme lines in each direction
    h1, h2 = horizontal[0], horizontal[-1]
    v1, v2 = vertical[0], vertical[-1]
    
    # Find 4 corner intersections
    corners = []
    for h in [h1, h2]:
        for v in [v1, v2]:
            corner = line_intersection(h, v)
            if corner:
                corners.append(corner)
    
    if len(corners) != 4:
        return None
    
    # Sort corners: top-left, top-right, bottom-right, bottom-left
    corners = sorted(corners, key=lambda x: (x[1], x[0]))  # Sort by y, then x
    
    if len(corners) == 4:
        top_pts = sorted(corners[:2], key=lambda x: x[0])  # Top 2, sort by x
        bottom_pts = sorted(corners[2:], key=lambda x: x[0])  # Bottom 2, sort by x
        return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
    
    return None


def perspective_transform(img, corners):
    """Apply perspective transformation to get bird's eye view"""
    # Calculate width and height of the output image
    width_top = np.linalg.norm(corners[0] - corners[1])
    width_bottom = np.linalg.norm(corners[3] - corners[2])
    width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(corners[0] - corners[3])
    height_right = np.linalg.norm(corners[1] - corners[2])
    height = int(max(height_left, height_right))
    
    # Destination points (rectangle)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Get perspective transformation matrix
    M = cv.getPerspectiveTransform(corners, dst)
    
    # Apply transformation
    warped = cv.warpPerspective(img, M, (width, height))
    return warped


# Main execution
img_original = cv.imread(image_path)
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Resize to width = 300px (maintain aspect ratio)
height, width = img.shape
scale = 300 / width
new_width = 300
new_height = int(height * scale)
img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
img_original = cv.resize(img_original, (new_width, new_height), interpolation=cv.INTER_AREA)

# Edge detection
kernel = np.ones((5,5), np.uint8)
dilated = cv.dilate(img, kernel, iterations=5,)
eroded = cv.erode(dilated, kernel, iterations=5)
edges = cv.Canny(eroded, 60, 180)

# Detect lines
lines = getLines(edges, 90)

# Draw lines on image
img_with_lines = img_original.copy()
for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Extract corners from lines
corners = get_corners_from_lines(lines, img.shape)

if corners is not None:
    # Draw corners
    img_with_corners = img_with_lines.copy()
    for corner in corners:
        cv.circle(img_with_corners, tuple(corner.astype(int)), 5, (255, 0, 0), -1)
    
    # Apply perspective transform
    transformed = perspective_transform(img_original, corners)
    
    # Show results
    # cv.imshow('Dialated', dilated)
    # cv.imshow('Eroded', eroded)
    # cv.imshow('1. Original with Lines', img_with_lines)
    # cv.imshow('2. Detected Corners', img_with_corners)
    cv.imshow('3. Transformed Paper', transformed)
else:
    print("Could not detect 4 corners from lines")
    cv.imshow('Lines Detected', img_with_lines)

# cv.imshow('Edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()