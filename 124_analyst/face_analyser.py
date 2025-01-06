from typing import Any, Optional, List, Tuple
import threading
import cv2
import numpy
import onnxruntime

import face.globals
from face.download import conditional_download
from face.face_store import get_static_faces, set_static_faces
from face.face_helper import warp_face, create_static_anchors, distance_to_kps, distance_to_bbox, apply_nms
from face.filesystem import resolve_relative_path
from face.typing import Frame, Face, FaceSet, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, ModelSet, Bbox, Kps, Score, Embedding
from face.vision import resize_frame_dimension

FACE_ANALYSER = None
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : ModelSet =\
{
	'face_detector_retinaface':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/retinaface_10g.onnx',
		'path': resolve_relative_path('../.assets/models/retinaface_10g.onnx')
	},
	'face_detector_yunet':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/yunet_2023mar.onnx',
		'path': resolve_relative_path('../.assets/models/yunet_2023mar.onnx')
	},
	'face_recognizer_arcface_blendswap':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_inswapper':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_simswap':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/arcface_simswap.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_simswap.onnx')
	},
	'gender_age':
	{
		'url': 'https://github.com/quangnhat2428/rp_face_/releases/download/source/gender_age.onnx',
		'path': resolve_relative_path('../.assets/models/gender_age.onnx')
	}
}


def get_face_analyser() -> Any:
	global FACE_ANALYSER

	with THREAD_LOCK:
		if FACE_ANALYSER is None:
			if face.globals.face_detector_model == 'retinaface':
				face_detector = onnxruntime.InferenceSession(MODELS.get('face_detector_retinaface').get('path'), providers = face.globals.execution_providers)
			if face.globals.face_detector_model == 'yunet':
				face_detector = cv2.FaceDetectorYN.create(MODELS.get('face_detector_yunet').get('path'), '', (0, 0))
			if face.globals.face_recognizer_model == 'arcface_blendswap':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_blendswap').get('path'), providers = face.globals.execution_providers)
			if face.globals.face_recognizer_model == 'arcface_inswapper':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_inswapper').get('path'), providers = face.globals.execution_providers)
			if face.globals.face_recognizer_model == 'arcface_simswap':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_simswap').get('path'), providers = face.globals.execution_providers)
			gender_age = onnxruntime.InferenceSession(MODELS.get('gender_age').get('path'), providers = face.globals.execution_providers)
			FACE_ANALYSER =\
			{
				'face_detector': face_detector,
				'face_recognizer': face_recognizer,
				'gender_age': gender_age
			}
	return FACE_ANALYSER


def clear_face_analyser() -> Any:
	global FACE_ANALYSER

	FACE_ANALYSER = None


def pre_check() -> bool:
	if not face.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_urls =\
		[
			MODELS.get('face_detector_retinaface').get('url'),
			MODELS.get('face_detector_yunet').get('url'),
			MODELS.get('face_recognizer_arcface_inswapper').get('url'),
			MODELS.get('face_recognizer_arcface_simswap').get('url'),
			MODELS.get('gender_age').get('url')
		]
		conditional_download(download_directory_path, model_urls)
	return True


def extract_faces(frame: Frame) -> List[Face]:
	face_detector_width, face_detector_height = map(int, face.globals.face_detector_size.split('x'))
	frame_height, frame_width, _ = frame.shape
	temp_frame = resize_frame_dimension(frame, face_detector_width, face_detector_height)
	temp_frame_height, temp_frame_width, _ = temp_frame.shape
	ratio_height = frame_height / temp_frame_height
	ratio_width = frame_width / temp_frame_width
	if face.globals.face_detector_model == 'retinaface':
		bbox_list, kps_list, score_list = detect_with_retinaface(temp_frame, temp_frame_height, temp_frame_width, face_detector_height, face_detector_width, ratio_height, ratio_width)
		return create_faces(frame, bbox_list, kps_list, score_list)
	elif face.globals.face_detector_model == 'yunet':
		bbox_list, kps_list, score_list = detect_with_yunet(temp_frame, temp_frame_height, temp_frame_width, ratio_height, ratio_width)
		return create_faces(frame, bbox_list, kps_list, score_list)
	return []
from numba import jit
import numpy as np
@jit(nopython=True)
def bbox_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    intersection_x = max(x1, x2)
    intersection_y = max(y1, y2)
    intersection_w = min(x1 + w1, x2 + w2) - intersection_x
    intersection_h = min(y1 + h1, y2 + h2) - intersection_y
    if intersection_w <= 0 or intersection_h <= 0:
        return False
    intersection_area = intersection_w * intersection_h
    area1 = w1 * h1
    area2 = w2 * h2
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou >= 0.45

@jit(nopython=True)
def remove_overlapping_bbox(bbox_list, kps_list, score_list):
    bbox_list = bbox_list[::-1]
    kps_list = kps_list[::-1]
    score_list = score_list[::-1]
    indices_to_keep = []
    for i in range(len(bbox_list)):
        keep = True
        for j in range(len(bbox_list)):
            if i != j and bbox_overlap(bbox_list[i], bbox_list[j]):
                if score_list[i] < score_list[j]:
                    keep = False
                    break
        if keep:
            indices_to_keep.append(i)
    
    kept_bboxes = [bbox_list[i] for i in indices_to_keep]
    kept_kps = [kps_list[i] for i in indices_to_keep]
    kept_scores = [score_list[i] for i in indices_to_keep]
    
    return kept_bboxes, kept_kps, kept_scores

@jit(nopython=True)
def rotate_frame(frame, angle):
    if angle == 0:
        return frame
    elif angle == 90:
        return np.ascontiguousarray(np.rot90(frame, k=1))
    elif angle == 180:
        return np.ascontiguousarray(np.rot90(frame, k=2))
    elif angle == 270:
        return np.ascontiguousarray(np.rot90(frame, k=3))
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")

@jit(nopython=True)
def rotate_bbox(bbox, angle, frame_width, frame_height):
    if angle == 0:
        return bbox
    elif angle == 90:
        return np.array([frame_height - bbox[3], bbox[0], frame_height - bbox[1], bbox[2]])
    elif angle == 180:
        return np.array([frame_width - bbox[2], frame_height - bbox[3], frame_width - bbox[0], frame_height - bbox[1]])
    elif angle == 270:
        return np.array([bbox[1], frame_width - bbox[2], bbox[3], frame_width - bbox[0]])
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")

@jit(nopython=True)
def rotate_kps(kps, angle, frame_width, frame_height):
    rotated_kps = np.zeros_like(kps)
    if angle == 0:
        return kps
    elif angle == 90:
        for i in range(kps.shape[0]):
            rotated_kps[i, 0] = frame_height - kps[i, 1]
            rotated_kps[i, 1] = kps[i, 0]
    elif angle == 180:
        for i in range(kps.shape[0]):
            rotated_kps[i, 0] = frame_width - kps[i, 0]
            rotated_kps[i, 1] = frame_height - kps[i, 1]
    elif angle == 270:
        for i in range(kps.shape[0]):
            rotated_kps[i, 0] = kps[i, 1]
            rotated_kps[i, 1] = frame_width - kps[i, 0]
    return rotated_kps


import numpy as np
from numba import jit
import numpy as np
@jit(nopython=True)
def bbox_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = intersection_x * intersection_y
    area1 = w1 * h1
    area2 = w2 * h2
    min_area = min(area1, area2)
    iou = intersection_area / min_area
    return iou >= 0.65

@jit(nopython=True)
def remove_overlapping_bbox(bbox_list, kps_list, score_list):
    indices_to_keep = []
    for i in range(len(bbox_list)):
        keep = True
        for j in range(len(bbox_list)):
            if i != j and bbox_overlap(bbox_list[i], bbox_list[j]):
                if score_list[i] < score_list[j]:
                    keep = False
                    break
        if keep:
            indices_to_keep.append(i)
    
    kept_bboxes = [bbox_list[i] for i in indices_to_keep]
    kept_kps = [kps_list[i] for i in indices_to_keep]
    kept_scores = [score_list[i] for i in indices_to_keep]
    
    return kept_bboxes, kept_kps, kept_scores
import numpy as np

def filter_detections(bbox_lists, kps_lists, score_lists):
    for angle in [0, 90, 180, 270]:
        bbox_list = bbox_lists[angle]
        kps_list = kps_lists[angle]
        score_list = score_lists[angle]
        
        filtered_bbox = []
        filtered_kps = []
        filtered_score = []
        
        for i in range(len(bbox_list)):
            # Lấy tọa độ của các điểm
            eye1 = kps_list[i][0]
            eye2 = kps_list[i][1]
            nose = kps_list[i][2]
            mouth1 = kps_list[i][3]
            mouth2 = kps_list[i][4]

            # Điều kiện kiểm tra các mối quan hệ giữa mắt, mũi và miệng
            eye_nose_condition = False
            nose_mouth_condition = False
            eye_mouth_condition = False

            if angle == 0:
                eye_nose_condition = eye1[1] < nose[1] and eye2[1] < nose[1]
                nose_mouth_condition = nose[1] < mouth1[1] and nose[1] < mouth2[1]
                eye_mouth_condition = eye1[1] < mouth1[1] and eye2[1] < mouth2[1]
            elif angle == 180:
                eye_nose_condition = eye1[1] > nose[1] and eye2[1] > nose[1]
                nose_mouth_condition = nose[1] > mouth1[1] and nose[1] > mouth2[1]
                eye_mouth_condition = eye1[1] > mouth1[1] and eye2[1] > mouth2[1]
            elif angle == 270:
                eye_nose_condition = eye1[0] < nose[0] and eye2[0] < nose[0]
                nose_mouth_condition = nose[0] < mouth1[0] and nose[0] < mouth2[0]
                eye_mouth_condition = eye1[0] < mouth1[0] and eye2[0] < mouth2[0]
            elif angle == 90:
                eye_nose_condition = eye1[0] > nose[0] and eye2[0] > nose[0]
                nose_mouth_condition = nose[0] > mouth1[0] and nose[0] > mouth2[0]
                eye_mouth_condition = eye1[0] > mouth1[0] and eye2[0] > mouth2[0]
            
            # Ưu tiên điều kiện mắt-mũi-miệng
            if eye_nose_condition and nose_mouth_condition and eye_mouth_condition:
                filtered_bbox.append(bbox_list[i])
                filtered_kps.append(kps_list[i])
                filtered_score.append(score_list[i])
            else:
                # Nếu điều kiện trên không thỏa mãn, kiểm tra vị trí của mắt so với miệng
                if angle == 0 and kps_list[i][1][1] < kps_list[i][3][1]:
                    filtered_bbox.append(bbox_list[i])
                    filtered_kps.append(kps_list[i])
                    filtered_score.append(score_list[i])
                elif angle == 180 and kps_list[i][1][1] > kps_list[i][3][1]:
                    filtered_bbox.append(bbox_list[i])
                    filtered_kps.append(kps_list[i])
                    filtered_score.append(score_list[i])
                elif angle == 270 and kps_list[i][0][0] < kps_list[i][3][0]:
                    filtered_bbox.append(bbox_list[i])
                    filtered_kps.append(kps_list[i])
                    filtered_score.append(score_list[i])
                elif angle == 90 and kps_list[i][0][0] > kps_list[i][3][0]:
                    filtered_bbox.append(bbox_list[i])
                    filtered_kps.append(kps_list[i])
                    filtered_score.append(score_list[i])
        
        bbox_lists[angle] = filtered_bbox
        kps_lists[angle] = filtered_kps
        score_lists[angle] = filtered_score



def detect_with_retinaface(temp_frame, temp_frame_height, temp_frame_width,
                           face_detector_height, face_detector_width,
                           ratio_height, ratio_width):
    face_detector = get_face_analyser().get('face_detector')
    bbox_lists = {0: [], 90: [], 180: [], 270: []}
    kps_lists = {0: [], 90: [], 180: [], 270: []}
    score_lists = {0: [], 90: [], 180: [], 270: []}
    feature_strides = [8, 16, 32]
    feature_map_channel = 3
    anchor_total = 2

    directions = [0,90,180,270]  # north, east, south, west

    for direction in directions:
        rotated_frame = rotate_frame(temp_frame, direction)
        prepare_frame = np.zeros((face_detector_height, face_detector_width, 3))
        rotated_height, rotated_width = rotated_frame.shape[:2]
        prepare_frame[:rotated_height, :rotated_width, :] = rotated_frame
        temp_frame_norm = (prepare_frame - 127.5) / 128.0
        temp_frame_norm = np.expand_dims(temp_frame_norm.transpose(2, 0, 1), axis=0).astype(np.float32)

        with THREAD_SEMAPHORE:
            detections = face_detector.run(None, {face_detector.get_inputs()[0].name: temp_frame_norm})

        for index, feature_stride in enumerate(feature_strides):
            keep_indices = np.where(detections[index] >= face.globals.face_detector_score)[0]
            if keep_indices.any():
                stride_height = face_detector_height // feature_stride
                stride_width = face_detector_width // feature_stride
                anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
                bbox_raw = (detections[index + feature_map_channel] * feature_stride)
                kps_raw = detections[index + feature_map_channel * 2] * feature_stride

                for bbox in distance_to_bbox(anchors, bbox_raw)[keep_indices]:
                    rotated_bbox = rotate_bbox(bbox, direction, rotated_width, rotated_height)
                    bbox_lists[direction].append(np.array([
                        rotated_bbox[0] * ratio_width,
                        rotated_bbox[1] * ratio_height,
                        rotated_bbox[2] * ratio_width,
                        rotated_bbox[3] * ratio_height
                    ]))

                for kps in distance_to_kps(anchors, kps_raw)[keep_indices]:
                    rotated_kps = rotate_kps(kps, direction, rotated_width, rotated_height)
                    kps_lists[direction].append(rotated_kps * [ratio_width, ratio_height])

                for score in detections[index][keep_indices]:
                    score_lists[direction].append(score[0])

    filter_detections(bbox_lists, kps_lists, score_lists)

    # Merge lists for all directions
    merged_bbox_list = sum(bbox_lists.values(), [])
    merged_kps_list = sum(kps_lists.values(), [])
    merged_score_list = sum(score_lists.values(), [])
    #merged_bbox_list, merged_kps_list, merged_score_list = remove_overlapping_bbox(merged_bbox_list, merged_kps_list, merged_score_list)
    return merged_bbox_list, merged_kps_list, merged_score_list







def detect_with_yunet(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, ratio_height : float, ratio_width : float) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')
	face_detector.setInputSize((temp_frame_width, temp_frame_height))
	face_detector.setScoreThreshold(face.globals.face_detector_score)
	bbox_list = []
	kps_list = []
	score_list = []
	with THREAD_SEMAPHORE:
		_, detections = face_detector.detect(temp_frame)
	if detections.any():
		for detection in detections:
			bbox_list.append(numpy.array(
			[
				detection[0] * ratio_width,
				detection[1] * ratio_height,
				(detection[0] + detection[2]) * ratio_width,
				(detection[1] + detection[3]) * ratio_height
			]))
			kps_list.append(detection[4:14].reshape((5, 2)) * [ ratio_width, ratio_height])
			score_list.append(detection[14])
	return bbox_list, kps_list, score_list


def create_faces(frame : Frame, bbox_list : List[Bbox], kps_list : List[Kps], score_list : List[Score]) -> List[Face]:
	faces = []
	if face.globals.face_detector_score > 0:
		sort_indices = numpy.argsort(-numpy.array(score_list))
		bbox_list = [ bbox_list[index] for index in sort_indices ]
		kps_list = [ kps_list[index] for index in sort_indices ]
		score_list = [ score_list[index] for index in sort_indices ]
		keep_indices = apply_nms(bbox_list, 0.4)
		for index in keep_indices:
			bbox = bbox_list[index]
			kps = kps_list[index]
			score = score_list[index]
			embedding, normed_embedding = calc_embedding(frame, kps)
			gender, age = detect_gender_age(frame, kps)
			faces.append(Face(
				bbox = bbox,
				kps = kps,
				score = score,
				embedding = embedding,
				normed_embedding = normed_embedding,
				gender = gender,
				age = age
			))
	return faces


def calc_embedding(temp_frame : Frame, kps : Kps) -> Tuple[Embedding, Embedding]:
	face_recognizer = get_face_analyser().get('face_recognizer')
	crop_frame, matrix = warp_face(temp_frame, kps, 'arcface_112_v2', (112, 112))
	crop_frame = crop_frame.astype(numpy.float32) / 127.5 - 1
	crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis = 0)
	embedding = face_recognizer.run(None,
	{
		face_recognizer.get_inputs()[0].name: crop_frame
	})[0]
	embedding = embedding.ravel()
	normed_embedding = embedding / numpy.linalg.norm(embedding)
	return embedding, normed_embedding


def detect_gender_age(frame : Frame, kps : Kps) -> Tuple[int, int]:
	gender_age = get_face_analyser().get('gender_age')
	crop_frame, affine_matrix = warp_face(frame, kps, 'arcface_112_v2', (96, 96))
	crop_frame = numpy.expand_dims(crop_frame, axis = 0).transpose(0, 3, 1, 2).astype(numpy.float32)
	prediction = gender_age.run(None,
	{
		gender_age.get_inputs()[0].name: crop_frame
	})[0][0]
	gender = int(numpy.argmax(prediction[:2]))
	age = int(numpy.round(prediction[2] * 100))
	return gender, age


def get_one_face(frame : Frame, position : int = 0) -> Optional[Face]:
	many_faces = get_many_faces(frame)
	if many_faces:
		try:
			return many_faces[position]
		except IndexError:
			return many_faces[-1]
	return None


def get_average_face(frames : List[Frame], position : int = 0) -> Optional[Face]:
	average_face = None
	faces = []
	embedding_list = []
	normed_embedding_list = []
	for frame in frames:
		face = get_one_face(frame, position)
		if face:
			faces.append(face)
			embedding_list.append(face.embedding)
			normed_embedding_list.append(face.normed_embedding)
	if faces:
		average_face = Face(
			bbox = faces[0].bbox,
			kps = faces[0].kps,
			score = faces[0].score,
			embedding = numpy.mean(embedding_list, axis = 0),
			normed_embedding = numpy.mean(normed_embedding_list, axis = 0),
			gender = faces[0].gender,
			age = faces[0].age
		)
	return average_face


def get_many_faces(frame : Frame) -> List[Face]:
	try:
		faces_cache = get_static_faces(frame)
		if faces_cache:
			faces = faces_cache
		else:
			faces = extract_faces(frame)
			set_static_faces(frame, faces)
		if face.globals.face_analyser_order:
			faces = sort_by_order(faces, face.globals.face_analyser_order)
		if face.globals.face_analyser_age:
			faces = filter_by_age(faces, face.globals.face_analyser_age)
		if face.globals.face_analyser_gender:
			faces = filter_by_gender(faces, face.globals.face_analyser_gender)
		return faces
	except (AttributeError, ValueError):
		return []


def find_similar_faces(frame : Frame, reference_faces : FaceSet, face_distance : float) -> List[Face]:
	similar_faces : List[Face] = []
	many_faces = get_many_faces(frame)

	if reference_faces:
		for reference_set in reference_faces:
			if not similar_faces:
				for reference_face in reference_faces[reference_set]:
					for face in many_faces:
						if compare_faces(face, reference_face, face_distance):
							similar_faces.append(face)
	return similar_faces


def compare_faces(face : Face, reference_face : Face, face_distance : float) -> bool:
	if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
		current_face_distance = 1 - numpy.dot(face.normed_embedding, reference_face.normed_embedding)
		return current_face_distance < face_distance
	return False


def sort_by_order(faces : List[Face], order : FaceAnalyserOrder) -> List[Face]:
	if order == 'left-right':
		return sorted(faces, key = lambda face: face.bbox[0])
	if order == 'right-left':
		return sorted(faces, key = lambda face: face.bbox[0], reverse = True)
	if order == 'top-bottom':
		return sorted(faces, key = lambda face: face.bbox[1])
	if order == 'bottom-top':
		return sorted(faces, key = lambda face: face.bbox[1], reverse = True)
	if order == 'small-large':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
	if order == 'large-small':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse = True)
	if order == 'best-worst':
		return sorted(faces, key = lambda face: face.score, reverse = True)
	if order == 'worst-best':
		return sorted(faces, key = lambda face: face.score)
	return faces


def filter_by_age(faces : List[Face], age : FaceAnalyserAge) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.age < 13 and age == 'child':
			filter_faces.append(face)
		elif face.age < 19 and age == 'teen':
			filter_faces.append(face)
		elif face.age < 60 and age == 'adult':
			filter_faces.append(face)
		elif face.age > 59 and age == 'senior':
			filter_faces.append(face)
	return filter_faces


def filter_by_gender(faces : List[Face], gender : FaceAnalyserGender) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.gender == 0 and gender == 'female':
			filter_faces.append(face)
		if face.gender == 1 and gender == 'male':
			filter_faces.append(face)
	return filter_faces
