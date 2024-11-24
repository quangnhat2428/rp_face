import shutil

# Đường dẫn nguồn
source_path = '/content/rp_face_/124_analyst/face_analyser.py'

# Đường dẫn đích
destination_path = '/content/extracted/124/face/face_analyser.py'

# Di chuyển tệp
shutil.move(source_path, destination_path)

print(f'Tệp đã được di chuyển từ {source_path} tới {destination_path}')
