import cv2
import numpy as np


class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """
    def __init__(self, pad_result: tuple=None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result




class Face():

    def __init__(self, min_face_size: int=20, steps_threshold: list=None,
                 scale_factor: float=0.709):
  
        self.__min_face_size = min_face_size
        self.__steps_threshold = steps_threshold
        self.__scale_factor = scale_factor
        
        return

    
    def detect_faces(self, img)-> list:
        self.__img = img
        if self.__img is None or not hasattr(self.__img, "shape"):
            raise IOError("Image not valid.")
    
        height, width, _ = self.__img.shape
        stage_status = StageStatus(width=width, height=height)
        
        m = 12 / self.__min_face_size
        min_layer = np.amin([height, width]) * m        

        scales = self.__compute_scale_pyramid(m, min_layer)
        stages = [self.__stage1]
        result = [scales, stage_status]
        
        for stage in stages:
            result = stage(img, result[0], result[1])
        
        [total_boxes, points] = result

        bounding_boxes = []
        
        for bounding_box, keypoints in zip(total_boxes, points.T):

            bounding_boxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                        'left_eye': (int(keypoints[0]), int(keypoints[5])),
                        'right_eye': (int(keypoints[1]), int(keypoints[6])),
                        'nose': (int(keypoints[2]), int(keypoints[7])),
                        'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                        'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }
            )

        return bounding_boxes


        
    def __stage1(self, image, scales: list, stage_status):
        """
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        total_boxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image,scale)

            img_x = np.expand_dims(scaled_image, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            out = self.__pnet.feed(img_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
                                                    out0[0, :, :, :].copy(), scale, self.__steps_threshold[0])

            # inter-scale nms
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numboxes = total_boxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = self.__rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return total_boxes, status

        
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        height, width, _ = image.shape
        
        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))
        
        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)
        
        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125
        
        return im_data_normalized
    
    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.__scale_factor, factor_count)]
            min_layer = min_layer * self.__scale_factor
            factor_count += 1

        return scales
    
  #all face detection main functionalities

###############################################################################



#result = detector.detect_faces(img)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)


cv2.imwrite("ivan_drawn.jpg", img)

print(result)





