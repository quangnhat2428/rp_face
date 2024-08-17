import shutil

# Đường dẫn nguồn
source_path = '/content/rp_face/124_analyst/face_analyser.py'

# Đường dẫn đích
destination_path = '/content/extracted/124/facefusion/face_analyser.py'

# Di chuyển tệp
shutil.move(source_path, destination_path)

print(f'Tệp đã được di chuyển từ {source_path} tới {destination_path}')
