import cv2
import mediapipe as mp
import numpy as np
import datetime
import sqlite3
import argparse
import os

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')


# Analizador de argumentos
parser = argparse.ArgumentParser(description='Medir el ángulo de inclinación de los brazos en un video')
parser.add_argument('--input', type=str, required=True, help='Ruta del video de entrada')
#parser.add_argument('--medir_left', action='store_true', help='Medir el ángulo del brazo izquierdo')
# parser.add_argument('--medir_right', action='store_true', help='Medir el ángulo del brazo derecho')
parser.add_argument('-o', '--output', action='store_true', help='Crear archivo de salida')
parser.add_argument('--show', action='store_true', help='Mostrar la inferencia en tiempo real')


# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

video_name = args.input
# medir_left = args.medir_left
# medir_right = args.medir_right
output = args.output
show = args.show

# Inicializar los módulos de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define las conexiones de los brazos
ARM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
]

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(video_name)


# Obtén las dimensiones del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Verificar si el archivo existe
if output:
    output_name = 'output'
    output_ext = '.mp4'
    output_index = 1
    output_dir = 'outputs'
    
    # Verificar si existe el directorio y crearlo si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Verificar si el archivo ya existe
    while os.path.exists(f'outputs\\{output_name}{(" (" + str(output_index) + ")" if output_index > 0 else "")}{output_ext}'):
        output_index += 1

    # Crear un objeto VideoWriter
    out = cv2.VideoWriter(f'outputs\\{output_name}{(" (" + str(output_index) + ")" if output_index > 0 else "")}{output_ext}', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))


def calculate_angle(a, b, c):
    a = np.array(a)  # Primer punto
    b = np.array(b)  # Punto central
    c = np.array(c)  # Segundo punto

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

left_angles = []
right_angles = []

# Conectar a la base de datos SQLite
conn = sqlite3.connect('datos.db')
c = conn.cursor()

# Crear una tabla si no existe
c.execute('''CREATE TABLE IF NOT EXISTS angle_detections
             (fuente TEXT, fecha TEXT, angle_min_left REAL, angle_max_left REAL, delta_angle_left REAL, 
              angle_min_right REAL, angle_max_right REAL, delta_angle_right REAL)''')



while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # Dibuja solo los brazos
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, ARM_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, 
                                mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST])
        
        # Obtener las coordenadas de los puntos de interés
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # print(f'Left Shoulder: ({left_shoulder.x}, {left_shoulder.y}), Left Elbow: ({left_elbow.x}, {left_elbow.y}), Left Wrist: ({left_wrist.x}, {left_wrist.y})')
        # print(f'Right Shoulder: ({right_shoulder.x}, {right_shoulder.y}), Right Elbow: ({right_elbow.x}, {right_elbow.y}), Right Wrist: ({right_wrist.x}, {right_wrist.y})')

        # Verificar si existe el brazo izquierdo
        if left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and left_wrist.visibility > 0.5:
            height, width, _ = frame.shape
            left_shoulder_x = int(left_shoulder.x * width)
            left_shoulder_y = int(left_shoulder.y * height)
            left_elbow_x = int(left_elbow.x * width)
            left_elbow_y = int(left_elbow.y * height)
            left_wrist_x = int(left_wrist.x * width)
            left_wrist_y = int(left_wrist.y * height)

            # Calcular el ángulo de inclinación
            left_angle = calculate_angle((left_shoulder_x, left_shoulder_y), (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y))

            # Almacenar el ángulo en una lista
            left_angles.append(left_angle)

            # Mostrar el ángulo en el frame
            cv2.putText(frame, f'Left Angle: {int(left_angle)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        # Verificar si existe el brazo derecho
        if right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5 and right_wrist.visibility > 0.5:
            height, width, _ = frame.shape
            right_shoulder_x = int(right_shoulder.x * width)
            right_shoulder_y = int(right_shoulder.y * height)
            right_elbow_x = int(right_elbow.x * width)
            right_elbow_y = int(right_elbow.y * height)
            right_wrist_x = int(right_wrist.x * width)
            right_wrist_y = int(right_wrist.y * height)

            # Calcular el ángulo de inclinación
            right_angle = calculate_angle((right_shoulder_x, right_shoulder_y), (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y))

            # Almacenar el ángulo en una lista
            right_angles.append(right_angle)

            # Mostrar el ángulo en el frame
            cv2.putText(frame, f'Right Angle: {int(right_angle)}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        # Escribe el frame en el archivo de salida
        if output:
            out.write(frame)

        # Muestra la inferencia en tiempo real
        if show:
            cv2.imshow('Pose Estimation', frame)

            # Si se presiona la tecla 'q', se rompe el bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Calcular los datos estadísticos
if left_angles:  # Verificar si la lista no está vacía
    angle_min_left = round(min(left_angles), 2)
    angle_max_left = round(max(left_angles), 2)
    delta_angle_left = round(angle_max_left - angle_min_left, 2)
else:
    angle_min_left = angle_max_left = delta_angle_left = None

if right_angles:  # Verificar si la lista no está vacía
    angle_min_right = round(min(right_angles), 2)
    angle_max_right = round(max(right_angles), 2)
    delta_angle_right = round(angle_max_right - angle_min_right, 2)
else:
    angle_min_right = angle_max_right = delta_angle_right = None


# Obtener la fecha y hora actuales
now = datetime.datetime.now()
fecha = now.strftime("%Y-%m-%d %H:%M:%S")

# Insertar los datos en la base de datos
c.execute("INSERT INTO angle_detections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
          (video_name, fecha, angle_min_left, angle_max_left, delta_angle_left,
           angle_min_right, angle_max_right, delta_angle_right))
conn.commit()

cap.release()  # liberar el objeto VideoCapture

if output:
    out.release()  # liberar el objeto VideoWriter

cv2.destroyAllWindows() # cerrar todas las ventanas de OpenCV

# Cerrar la conexión a la base de datos
conn.close()