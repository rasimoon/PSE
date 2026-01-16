import cv2 as cv
import numpy as np 

img_path = input('Enter image path: ')

# BGR NOT RGB

img = cv.imread(img_path)



def resize_image(img, dims):
    resized_image = cv.resize(img, dims, interpolation=cv.INTER_AREA) 
    return resized_image

img = resize_image(img, (400, 400))


objects_dict = {}

def find_adj_pixels(coordinates, img_row_sums, avg_brightness):
    all_coordinates = []
    all_coordinates.append(coordinates)
    i = 0
    while i < len(all_coordinates):
        coordinate = all_coordinates[i]
        adj_coordinates = ((coordinate[0] + 1, coordinate[1]), (coordinate[0], coordinate[1] + 1), (coordinate[0] - 1, coordinate[1]), (coordinate[0], coordinate[1] - 1),
                               (coordinate[0] + 1, coordinate[1] + 1), (coordinate[0] - 1, coordinate[1] - 1), (coordinate[0] - 1, coordinate[1] + 1), (coordinate[0] + 1, coordinate[1] - 1))
        for adj_coordinate in adj_coordinates:
                try:
                    brightness = img_row_sums[adj_coordinate[0], adj_coordinate[1]] 
                    if brightness > avg_brightness and (adj_coordinate not in all_coordinates):
                     all_coordinates.append(adj_coordinate)
                except:
                    pass
        i += 1
    return all_coordinates

def find_object(img_row_sums, objects_dict, avg_brightness):
    brightest = -1
    brightest_y = None
    brightest_x = None

    seen_coordinates = list(objects_dict.values())
    seen_coordinates = set(y for x in seen_coordinates for y in x)
    
    for j in range(len(img_row_sums)):
        brightness_idx = np.argmax(img_row_sums[j])
        brightness = img_row_sums[j][brightness_idx]
        if brightness > brightest and ((j, brightness_idx) not in seen_coordinates) and (brightness > avg_brightness):
            brightest = brightness
            brightest_y = j
            brightest_x = int(brightness_idx)
    if brightest == -1:
        return (False, objects_dict) 
    coordinates = (brightest_y, brightest_x)
    all_object_coordinates = find_adj_pixels(coordinates=coordinates, img_row_sums=img_row_sums, avg_brightness=avg_brightness)
    objects_count = len(list(objects_dict.keys()))
    objects_dict[f"Object {objects_count + 1}"] = all_object_coordinates 
    return (True, objects_dict) 
        
row_sums = np.sum(img, axis=2)

brightness_sum = 0

for row in row_sums:
    brightness_sum = brightness_sum + np.sum(row)

max_y = len(img)
max_x = len(img[0])

edge_coordinates = (max_y - 1, max_x - 1)

avg_amp = 2  # constant for regulating what pixel brightness is determined as being prominent (increase for detecting fainter objects)

avg_brightness = round((brightness_sum / (max_y * max_x)) * avg_amp) #average brightness across all pixels

searching = True 

iteration = 0
while searching:
     results = find_object(row_sums, objects_dict, avg_brightness)
     searching = results[0]
     objects_dict = results[1]
     iteration += 1
     print (iteration)

print('------------------------------------------------------------------------')


def create_bboxes(objects_dict, edge_coordinates):
    bboxes_dict = {}
    i = 0
    for detected_object in objects_dict:
        i += 1
        all_coordinates = objects_dict[detected_object]
        all_x_coordinates = [x[1] for x in all_coordinates]
        all_y_coordinates = [x[0] for x in all_coordinates]
        
     
        max_x = max(all_x_coordinates) + 1
        min_x = min(all_x_coordinates) - 1

        max_y = max(all_y_coordinates) + 1
        min_y = min(all_y_coordinates) - 1
        
        # Right at the edge safeguards 
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > edge_coordinates[1]:
            max_x = edge_coordinates[1]
        if max_y > edge_coordinates[0]:
            max_y = edge_coordinates[0]

        bbox = (min_x, min_y, max_x, max_y)

        bboxes_dict[f"Object {i}"] = bbox
    
    return bboxes_dict

def display_bboxes(bboxes_dict, img, type = 'all'):
    tot_objects = len(bboxes_dict)
    if type == 'all':
        img_cop = img.copy()
    
    i = 0
    for object in bboxes_dict:
        i += 1
        print(i)

        bbox = bboxes_dict[object]
        if type == 'select':
         temp_img = img.copy()
        else:
         temp_img = img_cop 

        min_y = bbox[1]
        min_x = bbox[0]

        max_y = bbox[3]
        max_x = bbox[2]
   

        # Side 1 
        for y in range(max_y - min_y):
            temp_img[min_y + y, min_x] = np.array([0, 0, 255])
        
        # Side 2
        for y in range(max_y - min_y):
            temp_img[min_y + y, max_x] = np.array([0, 0, 255])
        
        # Side 3 
        for x in range(max_x - min_x):
            temp_img[min_y, min_x + x] = np.array([0, 0, 255])

        # Side 4
        for x in range(max_x - min_x):
            temp_img[max_y, min_x + x] = np.array([0, 0, 255])
        
        if type == 'select':
          print(f"{object} / {tot_objects}")
          cv.imshow('Image with Highlighted Object', temp_img)
          cv.waitKey(0)
          cv.destroyAllWindows()

    if type == 'all':
        cv.imshow('Image with Highlighted Objects', img_cop)
        cv.waitKey(0)
        cv.destroyAllWindows()   
        

def add_bboxes(object_1, object_2, bboxes_dict):
    pass



bboxes_dict = create_bboxes(objects_dict, edge_coordinates=edge_coordinates)
display_bboxes(bboxes_dict=bboxes_dict, img=img, type="select")


