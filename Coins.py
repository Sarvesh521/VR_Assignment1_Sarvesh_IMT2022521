import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # if it already exists, delete everything in it
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

def ensure_file(file_path):
    if not os.path.exists(file_path):
        return False
    return True

def preprocess_image(image_path,color_space):
    """
    Reads the image from the given path and applies preprocessing:
      - Grayscale conversion / HSV conversion / LAB conversion
      - Gaussian blur / Median blur 
      - Histogram equalization
      - Canny edge detection / Marr-Hildreth edge detection / Sobel edge detection / Roberts edge detection / Prewitt edge detection
      - Morphological closing
    Returns the original image and the processed (closed edges) image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return None, None, None
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    orig = image.copy()

    if color_space == 'gray':
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (5, 5), 0)
      # median_blur = cv2.medianBlur(blurred, 5)
      # equalized = cv2.equalizeHist(blurred)
      return orig, image, blurred


    elif color_space == 'hsv':
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      h, s, v = cv2.split(hsv)
      blurred_v = cv2.GaussianBlur(v, (5, 5), 0)
      return orig, image, blurred_v

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    blurred_l = cv2.GaussianBlur(l, (5, 5), 0)
    return orig, image, blurred_l

def detect_coins_canny(orig, image, blurred, color_space,image_name):
    """
    Read the image and detect coin-like contours.
    Returns the original image and a list of coin contours.
    """

    # Canny edge detection and morphological closing
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # out_dir = f"Outputs/Part1/{color_space}/Canny"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/Canny"
    ensure_dir(out_dir)
    # Display the edges image
    plt.figure(figsize=(10, 10))
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Detected Edges using Canny in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/edges.jpg")
    plt.close()

    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    min_area = 500  # Adjust this value based on the coin scale in your image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)
    
    # Draw contours on image for visualization
    image_with_coins = image.copy()
    cv2.drawContours(image_with_coins, filtered_contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins using Canny in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/coins.jpg")
    plt.close()

    return orig, filtered_contours


# def detect_coins_marr_hildreth(orig, image, blurred, color_space):
#     """
#     Read the image and detect coin-like contours using an approximation of 
#     Marr–Hildreth edge detection (i.e. Laplacian of Gaussian).
#     Returns the original image and a list of coin contours.
#     """
#     # Apply Laplacian of Gaussian (LoG) operator
#     log = cv2.Laplacian(blurred, cv2.CV_64F)
#     log_abs = np.uint8(np.absolute(log))
    
#     # Determine threshold as 15% of the maximum value of the log_abs image
#     thresh_value = int(0.15 * np.max(log_abs))
    
#     # Threshold the Laplacian result to generate a binary edge map
#     _, log_edges = cv2.threshold(log_abs, thresh_value, 255, cv2.THRESH_BINARY)
    
#     kernel = np.ones((3, 3), np.uint8)
#     # dilated_edges = cv2.dilate(log_edges, kernel, iterations=1) 
#     closed_edges = cv2.morphologyEx(log_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     out_dir = f"Outputs/Part1/{color_space}/Marr-Hildreth"
#     ensure_dir(out_dir)
#     # Display the LoG edges image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(closed_edges, cmap='gray')
#     plt.title('Detected Edges (Marr-Hildreth / LoG) in ' + color_space)
#     plt.axis('off')
#     # plt.show()
#     plt.savefig(f"{out_dir}/edges.jpg")
#     plt.close()

#     # Find contours based on the closed edge map
#     contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw contours on a copy of the image for visualization
#     image_with_coins = image.copy()
#     cv2.drawContours(image_with_coins, contours, -1, (0, 255, 0), 2)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Coins using Marr-Hildreth in ' + color_space)
#     plt.axis('off')
#     # plt.show()
#     plt.savefig(f"{out_dir}/coins.jpg")
#     plt.close()

#     return orig, contours


def detect_coins_marr_hildreth(orig, image, blurred, color_space,image_name):
    """
    Read the image and detect coin-like contours using an approximation of 
    Marr–Hildreth edge detection (i.e. Laplacian of Gaussian).
    Filters out contours that are too small.
    Returns the original image and a list of filtered coin contours.
    """
    # Apply Laplacian of Gaussian (LoG) operator
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_abs = np.uint8(np.absolute(log))
    
    # Determine threshold as 10% of the maximum value of the log_abs image
    thresh_value = int(0.1 * np.max(log_abs))
    
    # Threshold the Laplacian result to generate a binary edge map
    _, log_edges = cv2.threshold(log_abs, thresh_value, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(log_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # out_dir = f"Outputs/Part1/{color_space}/Marr-Hildreth"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/Marr-Hildreth"
    ensure_dir(out_dir)
    
    # Save the LoG edges image
    plt.figure(figsize=(10, 10))
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Detected Edges (Marr-Hildreth / LoG) in ' + color_space)
    plt.axis('off')
    plt.savefig(f"{out_dir}/edges.jpg")
    plt.close()

    # Find contours from the closed edge map
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours, for example discarding those with a small area (tweak threshold as needed)
    filtered_contours = []
    min_area = 500  # Adjust this value based on the coin scale in your image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)
    
    # Draw filtered coin contours for visualization
    image_with_coins = image.copy()
    cv2.drawContours(image_with_coins, filtered_contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins using Marr-Hildreth in ' + color_space)
    plt.axis('off')
    plt.savefig(f"{out_dir}/coins.jpg")
    plt.close()

    return orig, filtered_contours


# def detect_coins_sobel(orig, image, blurred, color_space):
#     """
#     Detect coin-like contours using the Sobel edge detector.
#     Returns the original image and a list of coin contours.
#     """
#     # Compute Sobel gradients in x and y direction
#     sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
#     sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
#     # Compute gradient magnitude
#     sobel_mag = cv2.magnitude(sobelX, sobelY)
#     sobel_mag = np.uint8(np.absolute(sobel_mag))
    
#     # Threshold the Sobel magnitude image (adjust the threshold value as needed)
#     _, sobel_edges = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    
#     # Morphological closing to fill gaps in edges
#     kernel = np.ones((3, 3), np.uint8)
#     closed_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     out_dir = f"Outputs/Part1/{color_space}/Sobel"
#     ensure_dir(out_dir)
#     # Display Sobel edges image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(closed_edges, cmap='gray')
#     plt.title('Detected Edges (Sobel) in ' + color_space)
#     plt.axis('off')
#     # plt.show()
#     plt.savefig(f"{out_dir}/edges.jpg")
#     plt.close()

#     # Find contours on the closed edge map
#     contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw contours on image for visualization
#     image_with_coins = image.copy()
#     cv2.drawContours(image_with_coins, contours, -1, (0, 255, 0), 2)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
#     plt.title('Detected Coins using Sobel in ' + color_space)
#     plt.axis('off')
#     # plt.show()
#     plt.savefig(f"{out_dir}/coins.jpg")
#     plt.close()
    
#     return orig, contours


def detect_coins_sobel(orig, image, blurred, color_space,image_name):
    """
    Detect coin-like contours using the Sobel edge detector.
    Filters contours based on area and circularity.
    Returns the original image and a list of filtered coin contours.
    """
    # Compute Sobel gradients in x and y direction
    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude
    sobel_mag = cv2.magnitude(sobelX, sobelY)
    sobel_mag = np.uint8(np.absolute(sobel_mag))
    
    # Threshold the Sobel magnitude image (adjust threshold value as needed)
    _, sobel_edges = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # out_dir = f"Outputs/Part1/{color_space}/Sobel"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/Sobel"
    ensure_dir(out_dir)
    # Save Sobel edges image
    plt.figure(figsize=(10, 10))
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Detected Edges (Sobel) in ' + color_space)
    plt.axis('off')
    plt.savefig(f"{out_dir}/edges.jpg", bbox_inches='tight')
    plt.close()

    # Find contours on the closed edge map
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and circularity.
    filtered_contours = []
    min_area = 500  # Adjust this value based on coin size.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # perimeter = cv2.arcLength(cnt, True)
        # if perimeter == 0:
        #     continue
        # circularity = 4 * math.pi * area / (perimeter * perimeter)
        # # Adjust the circularity threshold as needed (1.0 is perfect circle)
        # if circularity < 0.7:
        #     continue
        filtered_contours.append(cnt)
    
    # Draw filtered contours on the image for visualization
    image_with_coins = image.copy()
    cv2.drawContours(image_with_coins, filtered_contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins using Sobel in ' + color_space)
    plt.axis('off')
    plt.savefig(f"{out_dir}/coins.jpg", bbox_inches='tight')
    plt.close()
    
    return orig, filtered_contours



def detect_coins_roberts(orig, image, blurred, color_space,image_name):
    """
    Detect coin-like contours using the Roberts edge detector.
    Returns the original image and a list of coin contours.
    """
    # Define Roberts operator kernels
    kernel_roberts_x = np.array([[1, 0],
                                 [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1],
                                 [-1, 0]], dtype=np.float32)
    
    # Apply Roberts filters on the blurred image
    roberts_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_roberts_x)
    roberts_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_roberts_y)
    # Compute gradient magnitude
    roberts_mag = cv2.magnitude(roberts_x, roberts_y)
    roberts_mag = np.uint8(np.absolute(roberts_mag))
    
    # Threshold the Roberts magnitude image (adjust threshold as needed)
    _, roberts_edges = cv2.threshold(roberts_mag, 10, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(roberts_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # out_dir = f"Outputs/Part1/{color_space}/Roberts"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/Roberts"
    ensure_dir(out_dir)
    # Display Roberts edges image
    plt.figure(figsize=(10, 10))
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Detected Edges (Roberts) in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/edges.jpg")
    plt.close()

    # Find contours on the closed edge map
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    filtered_contours = []
    min_area = 500  # Adjust this value based on coin size.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # perimeter = cv2.arcLength(cnt, True)
        # if perimeter == 0:
        #     continue
        # circularity = 4 * math.pi * area / (perimeter * perimeter)
        # # Adjust the circularity threshold as needed (1.0 is perfect circle)
        # if circularity < 0.7:
        #     continue
        filtered_contours.append(cnt)

    # Draw contours on a copy of the image
    image_with_coins = image.copy()
    cv2.drawContours(image_with_coins, filtered_contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins using Roberts in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/coins.jpg")
    plt.close()
    
    return orig, filtered_contours
def detect_coins_prewitt(orig, image, blurred, color_space,image_name):
    """
    Detect coin-like contours using the Prewitt edge detector.
    Returns the original image and a list of coin contours.
    """
    # Define Prewitt operator kernels
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]], dtype=np.float32)
    
    # Apply Prewitt filters on the blurred image
    prewitt_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_prewitt_y)
    # Compute gradient magnitude
    prewitt_mag = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_mag = np.uint8(np.absolute(prewitt_mag))
    
    # Threshold the Prewitt magnitude image (adjust threshold as needed)
    _, prewitt_edges = cv2.threshold(prewitt_mag, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(prewitt_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # out_dir = f"Outputs/Part1/{color_space}/Prewitt"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/Prewitt"
    ensure_dir(out_dir)
    # Display Prewitt edges image
    plt.figure(figsize=(10, 10))
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Detected Edges (Prewitt) in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/edges.jpg")
    plt.close()

    # Find contours on the closed edge map
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    min_area = 500  # Adjust this value based on coin size.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # perimeter = cv2.arcLength(cnt, True)
        # if perimeter == 0:
        #     continue
        # circularity = 4 * math.pi * area / (perimeter * perimeter)
        # # Adjust the circularity threshold as needed (1.0 is perfect circle)
        # if circularity < 0.7:
        #     continue
        filtered_contours.append(cnt)
    # Draw contours on a copy of the image for visualization
    image_with_coins = image.copy()
    cv2.drawContours(image_with_coins, filtered_contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins using Prewitt in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/coins.jpg")
    plt.close()
    
    return orig, filtered_contours


def segment_coins(orig_image, coin_contours,method,color_space,image_name):
    """
    Segment each coin from the original image using the detected contours.
    """
    # out_dir = f"Outputs/Part1/{color_space}/{method}/Segmented"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/{method}/Segmented"
    ensure_dir(out_dir)
    gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    for idx, contour in enumerate(coin_contours):
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255,thickness=cv2.FILLED)
        segmented = cv2.bitwise_and(orig_image, orig_image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        segmented_coin = segmented[y:y+h, x:x+w]
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(segmented_coin, cv2.COLOR_BGR2RGB))
        plt.title(f'Segmented Coin {idx + 1} using {method} in {color_space}')
        plt.axis('off')
        # plt.show()
        plt.savefig(f"{out_dir}/coin_{idx + 1}.jpg")
        plt.close()


def count_coins(orig_image, coin_contours,method,color_space,image_name):
    """
    Displays the original image with detected coin contours and writes the number
    of coins in red text on the image.
    """
    # Copy the original image and draw the detected coin contours (in green)
    image_with_coins = orig_image.copy()
    cv2.drawContours(image_with_coins, coin_contours, -1, (0, 255, 0), 2)
    
    # Calculate total coins and prepare the text to display
    total = len(coin_contours)
    text = f"Coins: {total}"
    
    # Choose the text position, font, scale, color (red), and thickness
    position = (10, 30)  # Top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red (BGR)
    thickness = 2
    
    # Overlay the text on the image
    cv2.putText(image_with_coins, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    

    # out_dir = f"Outputs/Part1/{color_space}/{method}"
    out_dir = f"Outputs/Part1/{image_name}/{color_space}/{method}"
    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_coins, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coins with Count using '+method + ' in ' + color_space)
    plt.axis('off')
    # plt.show()
    plt.savefig(f"{out_dir}/coins_with_count.jpg")
    plt.close()

images=['img1.jpg','img2.jpg','img3.jpg']
for image_path in images:
    image_actual_path = f"Inputs/Part-1/{image_path}"
    if not ensure_file(image_actual_path):
        print("Error: Image not found")
    else:
        print(f"Processing {image_path}")
        image_name = image_path.split('.')[0]
        colour_spaces = ['gray','hsv','lab']
        image_path = image_actual_path
        for color_space in colour_spaces:
            print(f"Processing {image_path} in {color_space} color space")
            orig,image,blurred = preprocess_image(image_path,color_space)
            
            print("Processing Canny")
            orig, coin_contours = detect_coins_canny(orig, image, blurred,color_space,image_name)
            segment_coins(orig, coin_contours,'Canny',color_space,image_name)
            count_coins(orig,coin_contours,'Canny',color_space,image_name)

            print("Processing Marr-Hildreth")

            orig, coin_contours = detect_coins_marr_hildreth(orig, image, blurred,color_space,image_name)
            segment_coins(orig, coin_contours,'Marr-Hildreth',color_space,image_name)
            count_coins(orig,coin_contours,'Marr-Hildreth',color_space,image_name)

            print("Processing Sobel")

            orig, coin_contours = detect_coins_sobel(orig, image, blurred,color_space,image_name)
            segment_coins(orig, coin_contours,'Sobel',color_space,image_name)
            count_coins(orig,coin_contours,'Sobel',color_space,image_name)

            print("Processing Roberts")

            orig, coin_contours = detect_coins_roberts(orig, image, blurred,color_space,image_name)
            segment_coins(orig, coin_contours,'Roberts',color_space,image_name)
            count_coins(orig,coin_contours,'Roberts',color_space,image_name)

            print("Processing Prewitt")

            orig, coin_contours = detect_coins_prewitt(orig, image, blurred,color_space,image_name)
            segment_coins(orig, coin_contours,'Prewitt',color_space,image_name)
            count_coins(orig,coin_contours,'Prewitt',color_space,image_name)