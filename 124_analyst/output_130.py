from typing import Tuple, Optional
from time import sleep
import gradio

import face.globals
from face import process_manager, wording
from face.core import conditional_process
from face.memory import limit_system_memory
from face.normalizer import normalize_output_path
from face.uis.components.source import listen
from face.uis.core import get_ui_component
from face.filesystem import clear_temp, is_image, is_video
exec(open("/content/extracted/130/face/uis/components/source.py").read())
OUTPUT_IMAGE : Optional[gradio.Image] = None
OUTPUT_VIDEO : Optional[gradio.Video] = None
OUTPUT_START_BUTTON : Optional[gradio.Button] = None
OUTPUT_CLEAR_BUTTON : Optional[gradio.Button] = None
OUTPUT_STOP_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global OUTPUT_IMAGE
	global OUTPUT_VIDEO
	global OUTPUT_START_BUTTON
	global OUTPUT_STOP_BUTTON
	global OUTPUT_CLEAR_BUTTON

	OUTPUT_IMAGE = gradio.Image(
		label = wording.get('uis.output_image_or_video'),
		visible = False
	)
	OUTPUT_VIDEO = gradio.Video(
		label = wording.get('uis.output_image_or_video')
	)
	OUTPUT_START_BUTTON = gradio.Button(
		value = wording.get('uis.start_button'),
		variant = 'primary',
		size = 'sm'
	)
	OUTPUT_STOP_BUTTON = gradio.Button(
		value = wording.get('uis.stop_button'),
		variant = 'primary',
		size = 'sm',
		visible = False
	)
	OUTPUT_CLEAR_BUTTON = gradio.Button(
		value = wording.get('uis.clear_button'),
		size = 'sm'
	)


def listen() -> None:
	output_path_textbox = get_ui_component('output_path_textbox')
	if output_path_textbox:
		OUTPUT_START_BUTTON.click(start, outputs = [ OUTPUT_START_BUTTON, OUTPUT_STOP_BUTTON ])
		OUTPUT_START_BUTTON.click(process, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO, OUTPUT_START_BUTTON, OUTPUT_STOP_BUTTON ])
	OUTPUT_STOP_BUTTON.click(stop, outputs = [ OUTPUT_START_BUTTON, OUTPUT_STOP_BUTTON ])
	OUTPUT_CLEAR_BUTTON.click(clear, outputs = [ OUTPUT_IMAGE, OUTPUT_VIDEO ])


def start() -> Tuple[gradio.Button, gradio.Button]:
	while not process_manager.is_processing():
		sleep(0.5)
	return gradio.Button(visible = False), gradio.Button(visible = True)
import os
output_path = "/content/drive/MyDrive/output"

# Kiểm tra xem thư mục đã tồn tại hay chưa
if not os.path.exists(output_path):
    # Tạo thư mục
    os.makedirs(output_path)
    print("Đã tạo thư mục thành công")
else:
    print("Thư mục đã tồn tại")
from typing import Tuple
import os
import gradio as gr
def process() -> Tuple[gr.Image, gr.Video, gr.Button, gr.Button]:
    
    output_path = "/content/drive/MyDrive/output"
    output_directory = "/content/output"
    absolute_directory = os.path.abspath(output_directory)
    file_paths = []
    
    # Thu thập các đường dẫn file
    for filename in os.listdir(output_directory):
        file_path = os.path.join(absolute_directory, filename)
        file_paths.append(file_path)
    face.globals.output_path = output_path
    # Xử lý từng đường dẫn file
    for file_path in file_paths:
        face.globals.target_path = file_path
        normed_output_path = normalize_output_path(face.globals.target_path, face.globals.output_path)
        if face.globals.system_memory_limit > 0:
            limit_system_memory(face.globals.system_memory_limit)
        conditional_process()
        
    if is_image(normed_output_path):
            return gr.Image(value=normed_output_path, visible=True), gr.Video(value=None, visible=False), gr.Button(visible=True), gr.Button(visible=False)
        
    if is_video(normed_output_path):
            return gr.Image(value=None, visible=False), gr.Video(value=normed_output_path, visible=True), gr.Button(visible=True), gr.Button(visible=False)
    
    return gr.Image(value=None), gr.Video(value=None), gr.Button(visible=True), gr.Button(visible=False)


def stop() -> Tuple[gradio.Button, gradio.Button]:
	process_manager.stop()
	return gradio.Button(visible = True), gradio.Button(visible = False)

process()
def clear() -> Tuple[gradio.Image, gradio.Video]:
	while process_manager.is_processing():
		sleep(0.5)
	if face.globals.target_path:
		clear_temp(face.globals.target_path)
	return gradio.Image(value = None), gradio.Video(value = None)