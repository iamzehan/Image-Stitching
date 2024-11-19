import os
import cv2
import glob
import json
import numpy as np

class ImageStitching:
    def __init__(self, output_dir='./results'):
        """
        Initializes ImageStiching Class
        :param output_dir: where outputs will be saved
        """
        self.output_dir = output_dir
        self.coordinates_dir = os.path.join(output_dir, "coordinates")
        self.stitched_images_dir = os.path.join(output_dir, "stitched_images")
        os.makedirs(self.coordinates_dir, exist_ok=True) # checking the existance of the output directory for the resulting coordinates
        os.makedirs(self.stitched_images_dir, exist_ok=True) #checking the existance of the output directory for the resulting images

    def save_coordinates(self, matches, keypoints_base, keypoints_next, folder_number, file_name):
        """
        finds the matched points and saves them to designated output folder
        :param matches: contains the match keypoints
        :param key_points_base: contains keypoints for feature_points from base image
        :param key_points_next: contains keypoints for features_points from the next image
        :param folder_number: contains the folder number for storage managibility 
        :param file_name: contains default syntax for the output file_name
        """
        matched_points = []
        for m in matches:
            base_pt = keypoints_base[m.queryIdx].pt  # Point from the base image
            next_pt = keypoints_next[m.trainIdx].pt  # Point from the next image
            matched_points.append({"base_point": base_pt, "next_point": next_pt})

        folder_path = os.path.join(self.coordinates_dir, folder_number)
        os.makedirs(folder_path, exist_ok=True)

        output_path = os.path.join(folder_path, file_name)
        with open(output_path, 'w') as f:
            json.dump(matched_points, f, indent=4)
        print(f"Saved matched keypoint coordinates to {output_path}")

    def stitch_images(self, images, folder_number):
        """
        Creates Panoramic Images and saves them to the designated output folder
        :param images: input images
        :param folder_number: input folder number are carried to better manage the output
        """
        base_img = images[0]

        for i in range(1, len(images)):
            #creating greyscale bases
            gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            gray_next = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

            orb = cv2.ORB_create(5000)
            keypoints_base, descriptors_base = orb.detectAndCompute(gray_base, None)
            keypoints_next, descriptors_next = orb.detectAndCompute(gray_next, None)

            # Using Bruteforce matcher to find matching keypoints
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_base, descriptors_next)
            matches = sorted(matches, key=lambda x: x.distance)

            self.save_coordinates(matches[:50], keypoints_base, keypoints_next, folder_number, f'matches_{i}.json')

            good_matches = matches[:50]

            src_pts = np.float32([keypoints_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_next[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            #Using RANSAC for Homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            height, width, _ = base_img.shape
            warped_img = cv2.warpPerspective(images[i], H, (width + images[i].shape[1], height))

            stitched_image = np.copy(warped_img)
            stitched_image[0:height, 0:width] = base_img

            base_img = stitched_image

        output_folder = os.path.join(self.stitched_images_dir, folder_number)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'panoramic_image_{folder_number}.jpg')
        cv2.imwrite(output_path, base_img)
        print(f"Panoramic image saved to {output_path}")

        return base_img


if __name__ == '__main__':
    # Load the images
    folder = input("Select a folder number (1-6):\n")
    image_folder = os.path.join("./Images", folder)
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.*')))  # Sort for consistent order

    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]

    if not images:
        print(f"No valid images found in {image_folder}")
        exit()

    stitcher = ImageStitching(output_dir='./results')

    panorama = stitcher.stitch_images(images, folder)

    cv2.imshow('Panoramic Image', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()