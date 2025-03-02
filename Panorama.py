import os
import cv2
import numpy as np

def load_images(folder_path):
    """Load all images (jpg, png, etc.) from a folder."""
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_paths.append(os.path.join(folder_path, filename))
    image_paths.sort()  # Ensure a consistent order
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load {path}")
    return images

def detect_features(images, method="ORB"):
    """
    Detect keypoints and compute descriptors for each image using either ORB or SIFT.
    
    Args:
        images (list): List of images.
        method (str): 'ORB' or 'SIFT'
        
    Returns:
        keypoints_list: List of lists of keypoints for each image.
        descriptors_list: List of descriptor arrays for each image.
    """
    method = method.upper()
    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
    elif method == "SIFT":
        detector = cv2.SIFT_create()
    else:
        raise ValueError("Unsupported method. Use 'ORB' or 'SIFT'.")
    
    keypoints_list = []
    descriptors_list = []
    for img in images:
        kp, des = detector.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    return keypoints_list, descriptors_list

def match_keypoints(descriptors_list, method="ORB", ratio_thresh=0.75):
    """
    Match keypoints between consecutive images.
    - If method='ORB', use Hamming distance.
    - If method='SIFT', use L2 (NORM_L2).
    
    Returns:
        matches_list: List of 'good' matches between consecutive images
                      (i.e., matches_list[i] is for images i and i+1)
    """
    method = method.upper()
    if method == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:  # SIFT
        bf = cv2.BFMatcher(cv2.NORM_L2)
    
    matches_list = []
    for i in range(len(descriptors_list) - 1):
        des1 = descriptors_list[i]
        des2 = descriptors_list[i+1]
        if des1 is None or des2 is None:
            print(f"Warning: Descriptors missing for image {i} or {i+1}")
            matches_list.append(None)
            continue
        
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        matches_list.append(good)
    return matches_list

def compute_homographies(matches_list, keypoints_list):
    """
    Compute homography for each pair of consecutive images using the matches.
    
    Returns:
        homographies: List of homography matrices for pairs (i, i+1)
                      (homographies[i] warps image i+1 to image i's coordinate space)
    """
    homographies = []
    for i, matches in enumerate(matches_list):
        if matches is None or len(matches) < 4:
            print(f"Not enough matches to compute homography between images {i} and {i+1}.")
            homographies.append(None)
            continue
        kp1 = keypoints_list[i]
        kp2 = keypoints_list[i+1]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        homographies.append(H)
    return homographies

def stitch_images_basic(images, method="ORB", ratio_thresh=0.75):
    """
    Stitch images using keypoint matching (ORB or SIFT) and homography,
    without any blending. Finally, crop black areas.
    
    Args:
        images (list of numpy.ndarray): List of images to stitch.
        method (str): 'ORB' or 'SIFT'
        ratio_thresh (float): Lowe's ratio test threshold.
    
    Returns:
        panorama (numpy.ndarray): Final stitched and cropped panorama.
    """
    keypoints_list, descriptors_list = detect_features(images, method=method)
    matches_list = match_keypoints(descriptors_list, method=method, ratio_thresh=ratio_thresh)
    homographies = compute_homographies(matches_list, keypoints_list)
    panorama = images[0]
    for i in range(1, len(images)):
        H = homographies[i - 1]
        if H is None:
            print(f"Skipping image {i}, homography is None.")
            continue
        h1, w1 = panorama.shape[:2]
        h2, w2 = images[i].shape[:2]
        warped_img = cv2.warpPerspective(images[i], H, (w1 + w2, h1))
        warped_img[0:h1, 0:w1] = panorama
        panorama = warped_img
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        panorama = panorama[y:y+h, x:x+w]
    return panorama

def feather_blend(base, warped, blend_width=30):
    """
    Feather blend the overlapping region between base and warped images.
    Assumes base and warped have the same height and partially overlapping width.
    """
    h_base, w_base = base.shape[:2]
    h_warp, w_warp = warped.shape[:2]
    out_w = max(w_base, w_warp)
    out_h = max(h_base, h_warp)
    blended = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    blended[:h_base, :w_base] = base
    overlap_x_start = 0
    overlap_x_end = min(w_base, w_warp)
    for x in range(overlap_x_start, overlap_x_end):
        dist_to_base_edge = w_base - x
        if x >= w_base:
            blended[:h_warp, x] = warped[:h_warp, x]
        elif x >= w_warp:
            continue
        else:
            alpha = 1.0
            if dist_to_base_edge < blend_width:
                alpha = dist_to_base_edge / float(blend_width)
            for y in range(0, min(h_base, h_warp)):
                px_base = base[y, x]
                px_warp = warped[y, x]
                blended[y, x] = alpha * px_base + (1 - alpha) * px_warp
    if w_warp > w_base:
        blended[:h_warp, w_base:w_warp] = warped[:h_warp, w_base:w_warp]
    return blended

def stitch_images_feathered(images, method="ORB", ratio_thresh=0.75, blend_width=30):
    """
    Stitch images using keypoint matching (ORB or SIFT) and homography,
    then apply a feather blend in the overlap region, and finally crop black areas.
    """
    keypoints_list, descriptors_list = detect_features(images, method=method)
    matches_list = match_keypoints(descriptors_list, method=method, ratio_thresh=ratio_thresh)
    homographies = compute_homographies(matches_list, keypoints_list)
    panorama = images[0]
    for i in range(1, len(images)):
        H = homographies[i - 1]
        if H is None:
            print(f"Skipping image {i}, homography is None.")
            continue
        h_pano, w_pano = panorama.shape[:2]
        h2, w2 = images[i].shape[:2]
        warped_img = cv2.warpPerspective(images[i], H, (w_pano + w2, h_pano))
        panorama = feather_blend(panorama, warped_img, blend_width=blend_width)
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        panorama = panorama[y:y+h, x:x+w]
    return panorama


def plot_and_print_keypoints(images, keypoints_list, method="ORB", output_folder="."):
    """
    For each input image, draw the detected keypoints, save the plotted image,
    and print the coordinates (first 10 keypoints for brevity).
    
    Args:
        images (list): List of images.
        keypoints_list (list): List of keypoints for each image.
        method (str): 'ORB' or 'SIFT'.
        output_folder (str): Folder to save keypoint images.
    """
    method = method.upper()
    for idx, (img, kp) in enumerate(zip(images, keypoints_list)):
        img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        output_name = os.path.join(output_folder, f"keypoints_{method}_{idx}.jpg")
        cv2.imwrite(output_name, img_kp)
        print(f"Saved keypoints image: {output_name}")
        print(f"Image {idx} ({method}) - Number of keypoints: {len(kp)}")
        for i, point in enumerate(kp[:10]):
            print(f"  Keypoint {i}: {point.pt}")
        print("-" * 50)

def ensure_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():
    input_folder = "Inputs/Part-2"
    output_folder = os.path.join("Outputs", "Part2")
    ensure_output_folder(output_folder)
    
    images = load_images(input_folder)
    if len(images) < 2:
        print("Not enough images to stitch.")
        return
    
    # Process and display keypoints for SIFT
    print("\nProcessing SIFT keypoints:")
    keypoints_sift, _ = detect_features(images, method="SIFT")
    plot_and_print_keypoints(images, keypoints_sift, method="SIFT", output_folder=output_folder)
    
    # 1) SIFT (no blending)
    print("\nStitching with SIFT (no blending)...")
    panorama_sift_basic = stitch_images_basic(images, method="SIFT", ratio_thresh=0.75)
    cv2.imwrite(os.path.join(output_folder, "final_panorama_sift.jpg"), panorama_sift_basic)
    print("Saved: final_panorama_sift.jpg")
    
    # 2) SIFT + Feather Blending
    print("\nStitching with SIFT + Feather Blending...")
    panorama_sift_feather = stitch_images_feathered(images, method="SIFT", ratio_thresh=0.75, blend_width=50)
    cv2.imwrite(os.path.join(output_folder, "final_panorama_sift_feather.jpg"), panorama_sift_feather)
    print("Saved: final_panorama_sift_feather.jpg")
    
    # Process and display keypoints for ORB
    print("\nProcessing ORB keypoints:")
    keypoints_orb, _ = detect_features(images, method="ORB")
    plot_and_print_keypoints(images, keypoints_orb, method="ORB", output_folder=output_folder)
    
    # 3) ORB (no blending)
    print("\nStitching with ORB (no blending)...")
    panorama_orb_basic = stitch_images_basic(images, method="ORB", ratio_thresh=0.75)
    cv2.imwrite(os.path.join(output_folder, "final_panorama_orb.jpg"), panorama_orb_basic)
    print("Saved: final_panorama_orb.jpg")
    
    # 4) ORB + Feather Blending
    print("\nStitching with ORB + Feather Blending...")
    panorama_orb_feather = stitch_images_feathered(images, method="ORB", ratio_thresh=0.75, blend_width=50)
    cv2.imwrite(os.path.join(output_folder, "final_panorama_orb_feather.jpg"), panorama_orb_feather)
    print("Saved: final_panorama_orb_feather.jpg")
    

if __name__ == "__main__":
    main()
